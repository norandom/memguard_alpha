"""Point-in-time FRED series: values as they stood on a decision date (Requirement 6.5).

FRED keeps vintages (ALFRED). Every observation row carries the real-time period
``[realtime_start, realtime_end]`` during which FRED published that value, so the
period ``realtime_start = realtime_end = as_of`` returns each observation as it
was known on ``as_of``, before any later revision, and omits observations not yet
published then. A series whose earliest vintage date is after ``as_of`` has no
such view: its current vintage is used and every row is flagged ``revised_data``.

Each request is one GET to ``{BASE_URL}/<endpoint>`` with transport retries and
redirects disabled, ambient proxy and netrc settings ignored and an explicit
timeout. At most 120 requests start in any rolling 60 seconds, through the
module-level ``RATE_LIMITER``. Every accepted response body is cached as one JSON
file per request under ``cache_dir`` (the private runs folder,
``DEFAULT_CACHE_DIR`` from the repository root), so a repeated query makes no
request and the study replays offline.

The API key is a query parameter. It is never stored and never part of a cache
file name or content, and it is redacted from exception messages, details and
urllib3's DEBUG request lines (a filter on the urllib3 loggers that print URLs).
"""

import hashlib
import json
import logging
import math
import os
import re
import tempfile
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from threading import Lock

import requests

from recall_guard.core.request_control import RequestError

#: FRED series routes; an endpoint name (``observations``, ``vintagedates``) is appended.
BASE_URL = "https://api.stlouisfed.org/fred/series"
#: The private runs folder for the cache, relative to the repository root.
DEFAULT_CACHE_DIR = Path("runs") / "finance-sim" / "fred"
MAX_REQUESTS_PER_MINUTE = 120
TIMEOUT_S = 30.0
#: FRED's marker for an observation without a value.
MISSING_VALUE = "."
_SERIES_ID = re.compile(r"[A-Za-z0-9_]+")
# FRED keys are 32-character lowercase alphanumeric strings. Alphanumeric text is
# unchanged by URL encoding, so redacting the raw form also covers every URL.
_API_KEY = re.compile(r"[A-Za-z0-9]+")
_KEY_PARAMETER = re.compile(r"(api_key=)[^&\s'\"]+")
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_DETAIL_LIMIT = 2000
_REDACTED = "[redacted]"


def _redact(text: str, secret: str) -> str:
    return text.replace(secret, _REDACTED)


class _KeyParameterFilter(logging.Filter):
    """Redact ``api_key`` query values from a urllib3 record (its DEBUG request line holds the URL)."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except (TypeError, ValueError):  # A filter that raises would break the caller's request.
            message = f"{record.msg} {record.args!r}"
        if "api_key=" in message:
            record.msg, record.args = _KEY_PARAMETER.sub(r"\1" + _REDACTED, message), ()
        return True


for _logger_name in ("urllib3.connectionpool", "urllib3.util.retry"):
    _logger = logging.getLogger(_logger_name)
    if not any(type(existing).__name__ == _KeyParameterFilter.__name__ for existing in _logger.filters):
        _logger.addFilter(_KeyParameterFilter())


def _plain_date(value, name: str) -> date:
    if not isinstance(value, date) or isinstance(value, datetime):
        raise TypeError(f"{name} must be a date, not a datetime")
    return value


def _finite(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    try:
        value = float(value)
    except OverflowError:
        raise ValueError(f"{name} must be a finite number") from None
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _series_id(value) -> str:
    if not isinstance(value, str) or _SERIES_ID.fullmatch(value) is None:
        raise ValueError("series_id must be a nonempty FRED series id of letters, digits and underscores")
    return value


def _iso_date(value, name: str) -> date:
    if not isinstance(value, str) or _ISO_DATE.fullmatch(value) is None:
        raise ValueError(f"{name} must be an ISO date (YYYY-MM-DD)")
    return date.fromisoformat(value)


@dataclass(frozen=True)
class Observation:
    """One FRED observation as it stood on the decision date ``as_of``.

    ``value`` is in the series' own units (an index level for CPIAUCSL, percent
    for T10Y2Y, BAA10Y and DTB3). ``vintage_start`` is the returned row's
    ``realtime_start``, the start of the real-time period that contains
    ``as_of``. It is None exactly when ``revised_data`` is True: the row then
    comes from the current vintage and may hold revisions published after
    ``as_of``.
    """

    series_id: str
    observation_date: date        # the period the value describes
    value: float
    as_of: date                   # the decision date the query was made for
    vintage_start: date | None    # realtime_start of the returned row; None when revised
    revised_data: bool            # True when no vintage existed on as_of

    def __post_init__(self) -> None:
        _series_id(self.series_id)
        _plain_date(self.observation_date, "observation_date")
        _plain_date(self.as_of, "as_of")
        object.__setattr__(self, "value", _finite(self.value, "value"))
        if not isinstance(self.revised_data, bool):
            raise TypeError("revised_data must be a bool")
        if self.observation_date > self.as_of:
            raise ValueError("observation_date must be on or before as_of")
        if self.revised_data != (self.vintage_start is None):
            raise ValueError("vintage_start must be None exactly when revised_data is True")
        if self.vintage_start is not None and _plain_date(self.vintage_start, "vintage_start") > self.as_of:
            raise ValueError("vintage_start must be on or before as_of")


class RollingRateLimiter:
    """Admit at most ``max_requests`` request starts in any rolling ``period_s`` seconds.

    ``clock`` returns monotonic seconds and ``sleep`` waits a number of seconds;
    both are injectable so tests never wait. Admission waits under a lock, so
    concurrent callers are admitted one at a time. A start is recorded at the
    actual wakeup time, so a late wakeup does not compress later starts, and
    never before its turn, even when ``sleep`` returns early.
    """

    def __init__(self, max_requests: int = MAX_REQUESTS_PER_MINUTE, period_s: float = 60.0, *,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep) -> None:
        if isinstance(max_requests, bool) or not isinstance(max_requests, int) or max_requests < 1:
            raise ValueError("max_requests must be a positive integer")
        if _finite(period_s, "period_s") <= 0:
            raise ValueError("period_s must be finite and greater than 0")
        self.max_requests = max_requests
        self.period_s = float(period_s)
        self._clock = clock
        self._sleep = sleep
        self._starts: deque[float] = deque()
        self._lock = Lock()

    def acquire(self) -> float:
        """Wait until a request may start, record its start and return the seconds asked of ``sleep``."""
        with self._lock:
            now = self._clock()
            self._expire(now)
            wait = 0.0
            if len(self._starts) >= self.max_requests:
                turn = self._starts[0] + self.period_s
                wait = max(0.0, turn - now)
                if wait > 0:
                    self._sleep(wait)
                now = max(self._clock(), turn)
                self._starts.popleft()
                self._expire(now)
            self._starts.append(now)
            return wait

    def _expire(self, now: float) -> None:
        while self._starts and self._starts[0] <= now - self.period_s:
            self._starts.popleft()


#: The limiter every FRED request passes; tests replace it with an injected clock and sleep.
RATE_LIMITER = RollingRateLimiter()


def _arguments(series_id, as_of, api_key, cache_dir) -> tuple[str, date, str, Path]:
    """Validate the shared arguments before any request; the key is never echoed."""
    _series_id(series_id)
    _plain_date(as_of, "as_of")
    if not isinstance(api_key, str):
        raise TypeError("api_key must be a string")
    if _API_KEY.fullmatch(api_key) is None:
        raise ValueError("api_key must be a nonempty alphanumeric FRED API key")
    if not isinstance(cache_dir, (str, os.PathLike)) or not os.fspath(cache_dir):
        raise TypeError("cache_dir must be a nonempty path")
    return series_id, as_of, api_key, Path(cache_dir)


def _cache_path(cache_dir: Path, series_id: str, slot: str, params: Mapping[str, str]) -> Path:
    """``<series_id>_<slot>_<sha256 of the canonical parameters>[:16].json``; ``params`` never hold the key."""
    canonical = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return cache_dir / f"{series_id}_{slot}_{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:16]}.json"


def _get(endpoint: str, params: Mapping[str, str], api_key: str) -> tuple[requests.Response | None, RequestError | None]:
    """Send one GET; a failure is returned, not raised, so no exception context (with the URL) is kept."""
    try:
        with requests.Session() as session:
            session.trust_env = False
            session.mount("http://", requests.adapters.HTTPAdapter(max_retries=0))
            session.mount("https://", requests.adapters.HTTPAdapter(max_retries=0))
            return session.get(f"{BASE_URL}/{endpoint}", params={**params, "api_key": api_key},
                               timeout=TIMEOUT_S, allow_redirects=False), None
    except requests.RequestException as error:
        return None, RequestError(f"FRED {endpoint} request for {params['series_id']} failed in transport",
                                  code="transport_error", retryable=True,
                                  detail=_redact(str(error), api_key)[:_DETAIL_LIMIT])


def _cached_body(endpoint: str, params: Mapping[str, str], slot: str, parse: Callable, *,
                 api_key: str, cache_dir: Path):
    """Return ``parse(body)`` from the cache, or from one request whose accepted body is then cached."""
    path = _cache_path(cache_dir, params["series_id"], slot, params)
    if path.is_file():
        return parse(json.loads(path.read_text(encoding="utf-8")))
    RATE_LIMITER.acquire()
    response, failure = _get(endpoint, params, api_key)
    if failure is not None:
        raise failure
    text = _redact(response.content.decode("utf-8", errors="replace"), api_key)
    if response.status_code != 200:
        raise RequestError(f"HTTP {response.status_code}: FRED {endpoint} request for {params['series_id']} "
                           "rejected (fred_http_error)", status_code=response.status_code,
                           code="fred_http_error", detail=text[:_DETAIL_LIMIT])
    try:
        body = json.loads(text)
    except ValueError:
        raise ValueError(f"FRED {endpoint} response for {params['series_id']} is not JSON") from None
    result = parse(body)
    cache_dir.mkdir(parents=True, exist_ok=True)
    handle, partial = tempfile.mkstemp(dir=cache_dir, prefix=path.stem + ".", suffix=".partial")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
        os.replace(partial, path)
    except BaseException:
        Path(partial).unlink(missing_ok=True)
        raise
    return result


def _vintage_dates(body) -> tuple[date, ...]:
    if not isinstance(body, Mapping) or not isinstance(body.get("vintage_dates"), list):
        raise ValueError("FRED vintagedates body must hold a vintage_dates list")
    return tuple(_iso_date(value, "vintage date") for value in body["vintage_dates"])


def _observations(body, series_id: str, as_of: date, point_in_time: bool) -> tuple[Observation, ...]:
    """Parse ``series/observations`` rows dated on or before ``as_of``; ``"."`` rows are skipped."""
    if not isinstance(body, Mapping) or not isinstance(body.get("observations"), list):
        raise ValueError("FRED observations body must hold an observations list")
    rows = body["observations"]
    count = body.get("count")
    if isinstance(count, int) and not isinstance(count, bool) and count > len(rows):
        raise ValueError(f"FRED returned a truncated observation list for {series_id}")
    observations = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("FRED observation rows must be objects")
        observation_date = _iso_date(row.get("date"), "observation date")
        if observation_date > as_of:
            raise ValueError(f"FRED returned a {series_id} observation dated after {as_of.isoformat()}")
        vintage_start = None
        if point_in_time:
            vintage_start = _iso_date(row.get("realtime_start"), "realtime_start")
            if not vintage_start <= as_of <= _iso_date(row.get("realtime_end"), "realtime_end"):
                raise ValueError(f"FRED returned a {series_id} row whose real-time period excludes "
                                 f"{as_of.isoformat()}")
        value = row.get("value")
        if isinstance(value, str) and value.strip() == MISSING_VALUE:
            continue
        try:
            number = float(value) if isinstance(value, str) else math.nan
        except ValueError:
            number = math.nan
        if not math.isfinite(number):
            raise ValueError(f"FRED {series_id} value for {observation_date.isoformat()} must be a finite number")
        observations.append(Observation(series_id, observation_date, number, as_of, vintage_start, not point_in_time))
    observations.sort(key=lambda observation: observation.observation_date)
    if len({observation.observation_date for observation in observations}) != len(observations):
        raise ValueError(f"FRED returned duplicate {series_id} observation dates")
    return tuple(observations)


def vintage_available(series_id: str, as_of: date, *, api_key: str, cache_dir: Path) -> bool:
    """True when the series' earliest vintage date is on or before ``as_of``.

    ``series/vintagedates`` is read once per series (``sort_order=asc``,
    ``limit=1``: the earliest vintage only) and cached as
    ``<series_id>_vintagedates_<hash>.json``, since one list serves every
    decision date. A series with no vintage at all is not available.
    """
    series_id, as_of, api_key, cache_dir = _arguments(series_id, as_of, api_key, cache_dir)
    params = {"series_id": series_id, "file_type": "json", "sort_order": "asc", "limit": "1"}
    dates = _cached_body("vintagedates", params, "vintagedates", _vintage_dates, api_key=api_key, cache_dir=cache_dir)
    return bool(dates) and min(dates) <= as_of


def fetch_series_as_of(series_id: str, as_of: date, *, api_key: str, cache_dir: Path) -> tuple[Observation, ...]:
    """Every observation dated on or before ``as_of`` as it stood on ``as_of``, oldest first.

    With a vintage on ``as_of`` the request is ``series/observations`` with
    ``realtime_start = realtime_end = observation_end = as_of``; otherwise it is
    the current vintage up to ``observation_end = as_of`` and every row has
    ``revised_data = True``. Missing values (``"."``) are skipped. HTTP errors
    raise RequestError ``fred_http_error`` with the status, network failures
    ``transport_error``; a body that is not valid FRED JSON, a non-finite value or
    a row outside ``as_of`` raises ValueError. Nothing invalid is cached, and a
    cached body that fails validation raises ValueError instead of being
    refetched (remove the file to refetch it).
    """
    series_id, as_of, api_key, cache_dir = _arguments(series_id, as_of, api_key, cache_dir)
    point_in_time = vintage_available(series_id, as_of, api_key=api_key, cache_dir=cache_dir)
    day = as_of.isoformat()
    params = {"series_id": series_id, "file_type": "json", "sort_order": "asc", "observation_end": day}
    if point_in_time:
        params |= {"realtime_start": day, "realtime_end": day}
    return _cached_body("observations", params, day,
                        lambda body: _observations(body, series_id, as_of, point_in_time),
                        api_key=api_key, cache_dir=cache_dir)


def fetch_point_in_time(series_id: str, as_of: date, *, api_key: str, cache_dir: Path) -> Observation | None:
    """The latest observation available on ``as_of`` (the last of ``fetch_series_as_of``), or None."""
    observations = fetch_series_as_of(series_id, as_of, api_key=api_key, cache_dir=cache_dir)
    return observations[-1] if observations else None
