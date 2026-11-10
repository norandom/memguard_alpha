"""Study item assembly for the memorization-adjusted finance simulation (task 7.1).

Builds the study's items and keeps every raw licensed value under the private
runs folder ``runs/finance-sim/``:

- Macro items (6.5): one per decision month, the last calendar day of each month
  from 2016-01 to 2026-06. The row holds CPI year-on-year, T10Y2Y and BAA10Y as
  they stood on the decision date (the FRED loader's point-in-time vectors), their
  60-month rolling z-scores and the per-series revised-data flags. A month whose
  z-score window is too short, or whose CPI base month is missing, is excluded
  with the reason.
- Realized next-month returns (6.6): FMP dividend-adjusted closes for SWDA.L, XLK
  and IAU, and DTB3 from the current FRED vintage as the cash leg. A month whose
  next month is not complete keeps ``realized = None`` with the reason; nothing is
  substituted. The returns go to a private file next to the raw FMP bodies.
- Cutoff classes (6.2, 6.4): each decision month is before or after each model's
  sourced cutoff in ``data/cutoffs.yaml`` (``pre_cutoff = decision_date <=
  cutoff``); a model without a sourced cutoff is excluded with the reason.
- EDGAR items: the frozen calibration rows (memorized and control) and the
  negative-control rows, checked against the corpus manifest's per-file SHA-256,
  each with its split, filing quarter and outlook prompt hash.

FMP and DTB3 are fetched through an injectable ``transport`` so tests replay
recorded bodies; the macro series go through the FRED loader and its cache. Keys
come from the process environment by name (``FRED_API_KEY``, ``FMP_API_TOKEN``),
travel only as query parameters and never enter a file, cache name, log line,
exception or repr. Nothing here starts a model host or sends a model request.
"""

from __future__ import annotations

import calendar
import hashlib
import json
import logging
import math
import os
import re
import statistics
import tempfile
from bisect import bisect_right
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import partial
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeVar

import requests

from recall_guard.core.loader import load_cutoffs
from recall_guard.core.request_control import RequestError
from recall_guard.dataset import fred
from recall_guard.dataset.edgar_corpora import Quarter
from recall_guard.dataset.edgar_outlook import prompt_sha256, render_outlook_prompt
from recall_guard.dataset.fred import Observation
from recall_guard.dataset.macro_regime import (
    ASSETS,
    MacroRow,
    RealizedReturns,
    render_anonymised,
    render_identifying,
)

T = TypeVar("T")

ROOT = Path(__file__).resolve().parents[2]
#: The private runs folder, resolved against the repository root; the FRED loader's cache is its ``fred`` folder.
PRIVATE_ROOT = ROOT / fred.DEFAULT_CACHE_DIR.parent
DEFAULT_CUTOFFS = ROOT / "data" / "cutoffs.yaml"
DEFAULT_EDGAR_DIR = ROOT / "data" / "edgar"
ITEMS_SCHEMA = "finance-sim-items/1"
RETURNS_SCHEMA = "finance-sim-realized-returns/1"

#: First and last decision months, (year, month), inclusive.
FIRST_MONTH = (2016, 1)
LAST_MONTH = (2026, 6)
#: The macro series of a row: FRED series id -> (raw key, z-score key) of ``MacroRow``.
MACRO_SERIES: Mapping[str, tuple[str, str]] = MappingProxyType({
    "CPIAUCSL": ("cpi_yoy", "cpi_yoy_z"),
    "T10Y2Y": ("t10y2y", "t10y2y_z"),
    "BAA10Y": ("baa10y", "baa10y_z"),
})
CPI_SERIES = "CPIAUCSL"
#: The cash leg of the realized returns; not a model input.
CASH_SERIES = "DTB3"
ZSCORE_WINDOW_MONTHS = 60
ZSCORE_MIN_MONTHS = 30
CASH_CATEGORY = "short_treasury_cash"
#: FMP symbol per risky asset category, from ``macro_regime.ASSETS``.
PRICE_SYMBOLS: Mapping[str, str] = MappingProxyType(
    {category: real for _, category, real in ASSETS if category != CASH_CATEGORY})
FMP_URL = "https://financialmodelingprep.com/stable/historical-price-eod/dividend-adjusted"
FMP_KEY_NAME = "FMP_API_TOKEN"
FRED_KEY_NAME = "FRED_API_KEY"
TIMEOUT_S = 30.0
#: A month-end close or rate older than this many days before the month's end is not month-end data.
MONTH_END_TOLERANCE_DAYS = 7
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
LICENSING_NOTE = (
    "Private research data. Realized returns derive from FMP dividend-adjusted closes (licensed) and FRED DTB3; "
    "BAA10Y levels are Moody's data, cited and never published. Keep under runs/finance-sim; publish derived "
    "statistics only.")

#: The EDGAR corpus files and their role in the study.
EDGAR_FILES: Mapping[str, str] = MappingProxyType({
    "calibration/is_memorized.jsonl": "calibration",
    "calibration/oos_control.jsonl": "calibration",
    "held_out/oos_control.jsonl": "negative_control",
    "held_out/is_memorized.jsonl": "unused",
})
EDGAR_UNUSED_REASON = ("unused: the negative control needs releases the model cannot have memorised, so the "
                       "held-out memorized releases take no part in the study")

_KEY = re.compile(r"[A-Za-z0-9]+")
_SYMBOL = re.compile(r"[A-Za-z0-9][A-Za-z0-9.\-]*")
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_ACCESSION = re.compile(r"\d{10}-\d{2}-\d{6}")
_URL = re.compile(r"https?://\S+")
_APIKEY_PARAMETER = re.compile(r"(apikey=)[^&\s'\"]+")
_DETAIL_LIMIT = 2000
_REDACTED = "[redacted]"

class _ApikeyParameterFilter(logging.Filter):
    """Redact FMP ``apikey`` query values from urllib3's DEBUG request lines (fred.py covers ``api_key``)."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except (TypeError, ValueError):  # A filter that raises would break the caller's request.
            message = f"{record.msg} {record.args!r}"
        if "apikey=" in message:
            record.msg, record.args = _APIKEY_PARAMETER.sub(r"\1" + _REDACTED, message), ()
        return True


for _logger_name in ("urllib3.connectionpool", "urllib3.util.retry"):
    _logger = logging.getLogger(_logger_name)
    if not any(type(existing).__name__ == _ApikeyParameterFilter.__name__ for existing in _logger.filters):
        _logger.addFilter(_ApikeyParameterFilter())


# ---------- errors ----------

class ExcludedMonth(Exception):
    """A decision month without a usable macro row; ``reason`` says why."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class ItemSourceError(RuntimeError):
    """A data source failed; ``code`` and ``retryable`` classify it, and no key is ever part of it.

    HTTP and transport failures keep the source's code (``fred_http_error``,
    ``fmp_http_error``, ``transport_error``) with the status; a body that is not
    valid source data is ``invalid_body`` and is never retryable.
    """

    def __init__(self, message: str, *, source: str, code: str, retryable: bool,
                 status_code: int | None = None, detail: str | None = None) -> None:
        super().__init__(message)
        self.source = source
        self.code = code
        self.retryable = retryable
        self.status_code = status_code
        self.detail = detail


# ---------- small helpers ----------

def _redact(text: str, secret: str) -> str:
    return text.replace(secret, _REDACTED) if secret else text


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _plain_date(value: Any, name: str) -> date:
    if not isinstance(value, date) or isinstance(value, datetime):
        raise TypeError(f"{name} must be a date, not a datetime")
    return value


def _iso_date(value: Any, name: str) -> date:
    if not isinstance(value, str) or _ISO_DATE.fullmatch(value) is None:
        raise ValueError(f"{name} must be an ISO date (YYYY-MM-DD); got {value!r}")
    return date.fromisoformat(value)


def _key(value: Any, name: str) -> str:
    """Validate a key without ever echoing it; alphanumeric text is unchanged by URL encoding."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if _KEY.fullmatch(value) is None:
        raise ValueError(f"{name} must be a nonempty alphanumeric API key")
    return value


def _directory(value: Any, name: str) -> Path:
    if not isinstance(value, (str, os.PathLike)) or not os.fspath(value):
        raise TypeError(f"{name} must be a nonempty path")
    return Path(value)


def _month_index(day: date) -> int:
    return day.year * 12 + day.month - 1


def _month_label(index: int) -> str:
    return f"{index // 12:04d}-{index % 12 + 1:02d}"


def _month_start(index: int) -> date:
    return date(index // 12, index % 12 + 1, 1)


def _month_end(index: int) -> date:
    year, month = index // 12, index % 12 + 1
    return date(year, month, calendar.monthrange(year, month)[1])


def _write_atomic(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` through a temporary file in the same folder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, partial_path = tempfile.mkstemp(dir=path.parent, prefix=path.stem + ".", suffix=".partial")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
        os.replace(partial_path, path)
    except BaseException:
        Path(partial_path).unlink(missing_ok=True)
        raise


def api_key_from_environment(name: str, environ: Mapping[str, str] | None = None) -> str:
    """The value of ``FRED_API_KEY`` or ``FMP_API_TOKEN`` from the process environment only.

    No file is read. A missing or empty variable raises ``ValueError`` naming the
    variable, never a value.
    """
    if name not in (FRED_KEY_NAME, FMP_KEY_NAME):
        raise ValueError(f"name must be {FRED_KEY_NAME} or {FMP_KEY_NAME}")
    value = (os.environ if environ is None else environ).get(name, "")
    if not value:
        raise ValueError(f"{name} is not set in the process environment")
    return value


@dataclass(frozen=True)
class PrivatePaths:
    """Where the assembly keeps its private files under one root (``runs/finance-sim`` by default)."""

    root: Path

    @property
    def fred_cache(self) -> Path:
        """The FRED loader's cache, which also holds the DTB3 current-vintage body."""
        return self.root / fred.DEFAULT_CACHE_DIR.name

    @property
    def fmp_cache(self) -> Path:
        """Raw FMP adjusted-close bodies: licensed, never published."""
        return self.root / "data" / "fmp"

    @property
    def realized_returns(self) -> Path:
        return self.root / "data" / "realized_returns.json"

    @property
    def items(self) -> Path:
        return self.root / "plan" / "items.json"


# ---------- decision months ----------

def _month_tuple(value: Any, name: str) -> int:
    if (not isinstance(value, tuple) or len(value) != 2
            or any(isinstance(part, bool) or not isinstance(part, int) for part in value)
            or not 1 <= value[1] <= 12 or value[0] < 1):
        raise ValueError(f"{name} must be a (year, month) pair of integers")
    return value[0] * 12 + value[1] - 1


def decision_months(first: tuple[int, int] = FIRST_MONTH, last: tuple[int, int] = LAST_MONTH) -> tuple[date, ...]:
    """The last calendar day of every month from ``first`` to ``last``, inclusive (126 by default)."""
    start, end = _month_tuple(first, "first"), _month_tuple(last, "last")
    if end < start:
        raise ValueError("last must not be before first")
    return tuple(_month_end(index) for index in range(start, end + 1))


def _validated_months(months: Any) -> tuple[date, ...]:
    if isinstance(months, (str, bytes)) or not isinstance(months, Sequence):
        raise TypeError("months must be a sequence of dates")
    months = tuple(months)
    if not months:
        raise ValueError("months must not be empty")
    for day in months:
        _plain_date(day, "decision month")
        if day != _month_end(_month_index(day)):
            raise ValueError(f"decision month {day.isoformat()} must be the last calendar day of its month")
    if any(later <= earlier for earlier, later in pairwise(months)):
        raise ValueError("decision months must be strictly increasing")
    return months


def price_window(months: Sequence[date]) -> tuple[date, date]:
    """The FMP and DTB3 window: the month before the first decision month to the month after the last."""
    months = _validated_months(months)
    return _month_start(_month_index(months[0]) - 1), _month_end(_month_index(months[-1]) + 1)


# ---------- macro rows ----------

def _vector(vectors: Mapping[str, Sequence[Observation]], series_id: str, decision_date: date) -> tuple[Observation, ...]:
    if series_id not in vectors:
        raise ValueError(f"vectors must hold the {series_id} observations")
    vector = tuple(vectors[series_id])
    for observation in vector:
        if not isinstance(observation, Observation):
            raise TypeError(f"{series_id} observations must be fred.Observation instances")
        if observation.series_id != series_id or observation.as_of != decision_date:
            raise ValueError(f"{series_id} observations must be the series as of {decision_date.isoformat()}")
    if any(later.observation_date <= earlier.observation_date for earlier, later in pairwise(vector)):
        raise ValueError(f"{series_id} observations must be in strictly increasing date order")
    if not vector:
        raise ExcludedMonth(f"{series_id} has no observation available on {decision_date.isoformat()}")
    return vector


def _trailing(monthly: Mapping[int, float], latest: int) -> list[float]:
    """The monthly values of the 60 calendar months that end with ``latest``, the latest included."""
    return [monthly[index] for index in range(latest - ZSCORE_WINDOW_MONTHS + 1, latest + 1) if index in monthly]


def _cpi_yoy(vector: tuple[Observation, ...]) -> tuple[float, list[float], int]:
    """Latest CPI year-on-year in percent, its monthly window and the latest month.

    The base is the level of the same calendar month one year earlier, which is
    the observation twelve places earlier on a complete monthly series; matching
    by month keeps a month FRED left without a value from shifting the base.
    """
    levels: dict[int, float] = {}
    for observation in vector:
        if observation.value <= 0:
            raise ExcludedMonth(f"{CPI_SERIES} level for {observation.observation_date.isoformat()} is not positive")
        levels[_month_index(observation.observation_date)] = observation.value
    if len(levels) != len(vector):
        raise ValueError(f"{CPI_SERIES} must hold one observation per month")
    yoy = {index: (level / levels[index - 12] - 1.0) * 100.0 for index, level in levels.items() if index - 12 in levels}
    latest = _month_index(vector[-1].observation_date)
    if latest not in yoy:
        raise ExcludedMonth(f"{CPI_SERIES} has no level for {_month_label(latest - 12)}, twelve months before its "
                            f"latest month {_month_label(latest)}")
    return yoy[latest], _trailing(yoy, latest), latest


def _daily(vector: tuple[Observation, ...]) -> tuple[float, list[float], int]:
    """Latest daily value, the window of month-end values (last observation of each month) and the latest month."""
    monthly: dict[int, float] = {}
    for observation in vector:  # Oldest first, so the last observation of each month wins.
        monthly[_month_index(observation.observation_date)] = observation.value
    latest = _month_index(vector[-1].observation_date)
    return vector[-1].value, _trailing(monthly, latest), latest


def _zscore(series_id: str, current: float, window: list[float], latest: int) -> float:
    label = _month_label(latest)
    if len(window) < ZSCORE_MIN_MONTHS:
        raise ExcludedMonth(f"{series_id} has {len(window)} monthly values in the {ZSCORE_WINDOW_MONTHS} months to "
                            f"{label}; its z-score needs at least {ZSCORE_MIN_MONTHS}")
    spread = statistics.stdev(window)
    if not math.isfinite(spread) or spread <= 0:
        raise ExcludedMonth(f"{series_id} does not vary over its {len(window)} monthly values to {label}; "
                            "its z-score is undefined")
    return (current - statistics.fmean(window)) / spread


def build_macro_row(decision_date: date, vectors: Mapping[str, Sequence[Observation]]) -> MacroRow:
    """The macro row of one decision month from the FRED vectors as of that date alone.

    ``vectors`` maps each of :data:`MACRO_SERIES` to ``fred.fetch_series_as_of``
    on ``decision_date``. ``cpi_yoy`` is the latest CPI level over the level of
    the same month a year earlier, minus one, in percent; ``t10y2y`` and
    ``baa10y`` are the latest daily values. Each z-score is the current value
    against the trailing 60 calendar months of monthly values (CPI year-on-year,
    or the last daily value of each month), the current month included, with
    ddof 1 and at least 30 values. ``revised`` holds each series'
    ``Observation.revised_data``. A month the rule cannot compute raises
    :class:`ExcludedMonth` with the reason; malformed vectors raise ``ValueError``.
    """
    _plain_date(decision_date, "decision_date")
    if not isinstance(vectors, Mapping):
        raise TypeError("vectors must map FRED series ids to observations")
    raw: dict[str, float] = {}
    zscores: dict[str, float] = {}
    revised: dict[str, bool] = {}
    for series_id, (raw_key, z_key) in MACRO_SERIES.items():
        vector = _vector(vectors, series_id, decision_date)
        current, window, latest = (_cpi_yoy if series_id == CPI_SERIES else _daily)(vector)
        raw[raw_key] = current
        zscores[z_key] = _zscore(series_id, current, window, latest)
        revised[series_id] = any(observation.revised_data for observation in vector)
    return MacroRow(decision_date=decision_date, raw=raw, zscores=zscores, revised=revised)


def row_to_dict(row: MacroRow) -> dict[str, Any]:
    """Plain JSON form of a macro row; its canonical JSON is what ``row_sha256`` hashes."""
    return {"decision_date": row.decision_date.isoformat(), "raw": dict(row.raw), "zscores": dict(row.zscores),
            "revised": dict(row.revised)}


# ---------- FMP and DTB3 through an injectable transport ----------

#: ``transport(url, params, timeout_s) -> (status code, body bytes)``; ``params`` carry the key.
Transport = Callable[[str, Mapping[str, str], float], tuple[int, bytes]]


def requests_transport(url: str, params: Mapping[str, str], timeout_s: float) -> tuple[int, bytes]:
    """One GET with transport retries and redirects disabled and ambient proxy and netrc settings ignored."""
    with requests.Session() as session:
        session.trust_env = False
        session.mount("http://", requests.adapters.HTTPAdapter(max_retries=0))
        session.mount("https://", requests.adapters.HTTPAdapter(max_retries=0))
        response = session.get(url, params=dict(params), timeout=timeout_s, allow_redirects=False)
        return response.status_code, response.content


def _send(transport: Transport, url: str, params: Mapping[str, str], secret: str, source: str) -> tuple[int, str]:
    """Call the transport; a failure is raised after the handler, so no exception context (with the URL) is kept."""
    failure = None
    try:
        status, body = transport(url, params, TIMEOUT_S)
    except requests.RequestException as error:
        failure = _redact(f"{type(error).__name__}: {error}", secret)[:_DETAIL_LIMIT]
    if failure is not None:
        raise RequestError(f"{source} request failed in transport", code="transport_error", retryable=True,
                           detail=failure)
    if isinstance(status, bool) or not isinstance(status, int) or not isinstance(body, bytes):
        raise TypeError("transport must return (status code, body bytes)")
    return status, _redact(body.decode("utf-8", errors="replace"), secret)


def _json_body(text: str, what: str) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        raise ValueError(f"{what} is not JSON") from None


def _cached(path: Path, parse: Callable[[Any], T], fetch: Callable[[], tuple[int, str]], source: str,
            http_code: str) -> T:
    """``parse`` of the cached body, or of one fetched body that is cached once it parses.

    A cached body that fails to parse raises ``ValueError`` instead of being
    refetched (remove the file to refetch it). Nothing invalid is cached.
    """
    if path.is_file():
        return parse(_json_body(path.read_text(encoding="utf-8"), f"cached {source} body {path.name}"))
    status, text = fetch()
    if status != 200:
        raise RequestError(f"HTTP {status}: {source} request rejected ({http_code})", status_code=status,
                           code=http_code, retryable=status in RETRYABLE_STATUS, detail=text[:_DETAIL_LIMIT])
    result = parse(_json_body(text, f"{source} response"))
    _write_atomic(path, text)
    return result


def _window(start: Any, end: Any) -> tuple[date, date]:
    _plain_date(start, "start")
    _plain_date(end, "end")
    if end < start:
        raise ValueError("end must not be before start")
    return start, end


def _parse_closes(payload: Any, symbol: str, start: date, end: date) -> tuple[tuple[date, float], ...]:
    if not isinstance(payload, list):
        message = payload.get("Error Message") if isinstance(payload, Mapping) else None
        suffix = f": {message[:200]}" if isinstance(message, str) else ""
        raise ValueError(f"FMP adjusted-close body for {symbol} must be a list of rows{suffix}")
    closes: dict[date, float] = {}
    for row in payload:
        if not isinstance(row, Mapping):
            raise ValueError(f"FMP adjusted-close rows for {symbol} must be objects")
        if "symbol" in row and row["symbol"] != symbol:
            raise ValueError(f"FMP returned a row for another symbol in the {symbol} body")
        day = _iso_date(row.get("date"), f"FMP {symbol} date")
        if not start <= day <= end:
            raise ValueError(f"FMP returned a {symbol} close dated {day.isoformat()}, outside {start.isoformat()} "
                             f"to {end.isoformat()}")
        value = row.get("adjClose")
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                or value <= 0):
            raise ValueError(f"FMP {symbol} adjClose for {day.isoformat()} must be a positive finite number")
        if day in closes:
            raise ValueError(f"FMP returned duplicate {symbol} closes for {day.isoformat()}")
        closes[day] = float(value)
    if not closes:
        raise ValueError(f"FMP returned no adjusted closes for {symbol} from {start.isoformat()} to {end.isoformat()}")
    return tuple(sorted(closes.items()))


def fetch_adjusted_closes(symbol: str, *, start: date, end: date, api_key: str, cache_dir: Path,
                          transport: Transport = requests_transport) -> tuple[tuple[date, float], ...]:
    """FMP dividend-adjusted closes (``adjClose``) of ``symbol`` from ``start`` to ``end``, oldest first.

    One GET to :data:`FMP_URL` with ``symbol``, ``from``, ``to`` and ``apikey``.
    The body is cached as ``<symbol>_<start>_<end>.json`` under ``cache_dir`` (the
    private FMP folder; licensed data), so a repeated call makes no request. HTTP
    errors raise ``RequestError`` ``fmp_http_error`` with the status (retryable for
    429 and 5xx), transport failures ``transport_error``; a body that is not a list
    of rows with ISO dates inside the window and positive finite ``adjClose``
    values, or that holds no row, raises ``ValueError``.
    """
    if not isinstance(symbol, str) or _SYMBOL.fullmatch(symbol) is None:
        raise ValueError("symbol must be a ticker of letters, digits, dots and hyphens")
    start, end = _window(start, end)
    api_key = _key(api_key, FMP_KEY_NAME)
    cache_dir = _directory(cache_dir, "cache_dir")
    params = {"symbol": symbol, "from": start.isoformat(), "to": end.isoformat()}
    source = f"FMP adjusted closes for {symbol}"
    return _cached(cache_dir / f"{symbol}_{start.isoformat()}_{end.isoformat()}.json",
                   partial(_parse_closes, symbol=symbol, start=start, end=end),
                   partial(_send, transport, FMP_URL, {**params, "apikey": api_key}, api_key, source),
                   source, "fmp_http_error")


def _parse_rates(body: Any, start: date, end: date) -> tuple[tuple[date, float], ...]:
    if not isinstance(body, Mapping) or not isinstance(body.get("observations"), list):
        raise ValueError(f"FRED {CASH_SERIES} body must hold an observations list")
    rows = body["observations"]
    count = body.get("count")
    if isinstance(count, int) and not isinstance(count, bool) and count > len(rows):
        raise ValueError(f"FRED returned a truncated {CASH_SERIES} observation list")
    rates: dict[date, float] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"FRED {CASH_SERIES} observation rows must be objects")
        day = _iso_date(row.get("date"), f"{CASH_SERIES} observation date")
        if not start <= day <= end:
            raise ValueError(f"FRED returned a {CASH_SERIES} observation dated {day.isoformat()}, outside the window")
        value = row.get("value")
        if isinstance(value, str) and value.strip() == fred.MISSING_VALUE:
            continue
        try:
            number = float(value) if isinstance(value, str) else math.nan
        except ValueError:
            number = math.nan
        if not math.isfinite(number):
            raise ValueError(f"FRED {CASH_SERIES} value for {day.isoformat()} must be a finite number")
        if day in rates:
            raise ValueError(f"FRED returned duplicate {CASH_SERIES} observations for {day.isoformat()}")
        rates[day] = number
    if not rates:
        raise ValueError(f"FRED returned no {CASH_SERIES} values from {start.isoformat()} to {end.isoformat()}")
    return tuple(sorted(rates.items()))


def _fetch_rates(transport: Transport, params: Mapping[str, str], api_key: str, source: str) -> tuple[int, str]:
    fred.RATE_LIMITER.acquire()
    return _send(transport, f"{fred.BASE_URL}/observations", {**params, "api_key": api_key}, api_key, source)


def fetch_cash_rates(*, start: date, end: date, api_key: str, cache_dir: Path,
                     transport: Transport = requests_transport) -> tuple[tuple[date, float], ...]:
    """DTB3 in percent from the current FRED vintage, from ``start`` to ``end``, oldest first.

    One ``series/observations`` GET without a real-time period (the current
    vintage), counted against the FRED loader's rate limiter and cached under
    ``cache_dir`` as ``DTB3_current_<hash>.json`` in the loader's naming scheme.
    Missing values (``"."``) are skipped. Errors follow
    :func:`fetch_adjusted_closes`, with the code ``fred_http_error``.
    """
    start, end = _window(start, end)
    api_key = _key(api_key, FRED_KEY_NAME)
    cache_dir = _directory(cache_dir, "cache_dir")
    params = {"series_id": CASH_SERIES, "file_type": "json", "sort_order": "asc",
              "observation_start": start.isoformat(), "observation_end": end.isoformat()}
    digest = hashlib.sha256(json.dumps(params, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    source = f"FRED {CASH_SERIES} observations"
    return _cached(cache_dir / f"{CASH_SERIES}_current_{digest[:16]}.json",
                   partial(_parse_rates, start=start, end=end),
                   partial(_fetch_rates, transport, params, api_key, source), source, "fred_http_error")


# ---------- realized returns ----------

@dataclass(frozen=True)
class RealizedOutcome:
    """The realized next-month returns of one decision month, or None with the reason.

    ``dates`` records the data used: per risky category the close dates in the
    decision month and the next month, and for the cash leg the DTB3 date.
    """

    decision_date: date
    realized: RealizedReturns | None
    reason: str | None
    dates: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        _plain_date(self.decision_date, "decision_date")
        if (self.realized is None) == (self.reason is None):
            raise ValueError("a realized outcome holds either returns or the reason they are missing")

    def to_dict(self) -> dict[str, Any]:
        realized = None if self.realized is None else {
            name: getattr(self.realized, name) for name in ("world_equity", "tech_sector", "gold_commodity",
                                                            CASH_CATEGORY)}
        return {"decision_date": self.decision_date.isoformat(), "realized": realized, "reason": self.reason,
                "dates": {name: list(days) for name, days in sorted(self.dates.items())}}


def _series(values: Any, name: str) -> tuple[tuple[date, float], ...]:
    series = tuple(values)
    for entry in series:
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise TypeError(f"{name} must hold (date, value) pairs")
        _plain_date(entry[0], f"{name} date")
        if isinstance(entry[1], bool) or not isinstance(entry[1], (int, float)) or not math.isfinite(entry[1]):
            raise ValueError(f"{name} values must be finite numbers")
    if any(later[0] <= earlier[0] for earlier, later in pairwise(series)):
        raise ValueError(f"{name} must be in strictly increasing date order")
    return series


def _stale(day: date, index: int) -> bool:
    return (_month_end(index) - day).days > MONTH_END_TOLERANCE_DAYS


def _outcome(decision_date: date, last_close: Mapping[str, Mapping[int, tuple[date, float]]],
             cash: tuple[tuple[date, float], ...], cash_dates: list[date], complete_through: date) -> RealizedOutcome:
    index = _month_index(decision_date)
    if _month_end(index + 1) > complete_through:
        return RealizedOutcome(decision_date, None, f"the next month {_month_label(index + 1)} is not complete: data "
                               f"through {complete_through.isoformat()}", {})
    dates: dict[str, tuple[str, ...]] = {}
    returns: dict[str, float] = {}
    for category, symbol in PRICE_SYMBOLS.items():
        pair = []
        for month in (index, index + 1):
            close = last_close[category].get(month)
            if close is None or _stale(close[0], month):
                return RealizedOutcome(decision_date, None, f"{symbol} has no adjusted close in the last "
                                       f"{MONTH_END_TOLERANCE_DAYS} days of {_month_label(month)}", dates)
            pair.append(close)
        dates[category] = (pair[0][0].isoformat(), pair[1][0].isoformat())
        returns[category] = pair[1][1] / pair[0][1] - 1.0
    position = bisect_right(cash_dates, decision_date)
    if position == 0 or _stale(cash_dates[position - 1], index):
        return RealizedOutcome(decision_date, None, f"{CASH_SERIES} has no value in the last "
                               f"{MONTH_END_TOLERANCE_DAYS} days of {_month_label(index)}", dates)
    rate_day, rate = cash[position - 1]
    dates[CASH_CATEGORY] = (rate_day.isoformat(),)
    returns[CASH_CATEGORY] = rate / 100.0 / 12.0
    return RealizedOutcome(decision_date, RealizedReturns(**returns), None, dates)


def realized_returns(months: Sequence[date], closes: Mapping[str, Sequence[tuple[date, float]]],
                     cash: Sequence[tuple[date, float]], *, complete_through: date) -> tuple[RealizedOutcome, ...]:
    """Realized next-month returns per decision month M, as decimals.

    Per risky category ``adjClose(last trading day of M+1) / adjClose(last trading
    day of M) - 1``, where a month's last trading day is the symbol's last close
    in that month; the cash leg is ``DTB3(last observation on or before the end of
    M) / 100 / 12``. M keeps ``realized = None`` with the reason when M+1 ends
    after ``complete_through``, or when a close or the rate is missing or more
    than :data:`MONTH_END_TOLERANCE_DAYS` days before its month's end. Nothing is
    substituted.
    """
    months = _validated_months(months)
    _plain_date(complete_through, "complete_through")
    if not isinstance(closes, Mapping) or set(closes) != set(PRICE_SYMBOLS):
        raise ValueError(f"closes must hold exactly the categories {sorted(PRICE_SYMBOLS)}")
    last_close: dict[str, dict[int, tuple[date, float]]] = {}
    for category in PRICE_SYMBOLS:
        by_month: dict[int, tuple[date, float]] = {}
        for day, value in _series(closes[category], f"closes[{category!r}]"):
            if value <= 0:
                raise ValueError(f"closes[{category!r}] must be positive")
            by_month[_month_index(day)] = (day, float(value))  # Oldest first: the month's last close wins.
        last_close[category] = by_month
    cash_series = _series(cash, "cash")
    cash_dates = [day for day, _ in cash_series]
    return tuple(_outcome(decision_date, last_close, cash_series, cash_dates, complete_through)
                 for decision_date in months)


# ---------- cutoff classes ----------

def sourced_cutoffs(path: Path | str = DEFAULT_CUTOFFS) -> dict[str, tuple[date, str | None]]:
    """``{model_id: (cutoff, source URL or None)}`` from the cutoff registry.

    The cutoff comes from ``load_cutoffs``; the source is the first URL in the
    comment on the model's own registry line, as the registry rules require.
    """
    path = Path(path)
    cutoffs = load_cutoffs(path)
    text = path.read_text(encoding="utf-8")
    out: dict[str, tuple[date, str | None]] = {}
    for model_id, cutoff in cutoffs.items():
        line = re.search(rf"^[ \t]*[\"']?{re.escape(model_id)}[\"']?[ \t]*:[^#\n]*#(?P<comment>[^\n]*)$", text,
                         re.MULTILINE)
        url = _URL.search(line.group("comment")) if line else None
        out[model_id] = (cutoff, url.group(0) if url else None)
    return out


@dataclass(frozen=True)
class ModelCutoff:
    """A model with a sourced cutoff and the class of every decision month."""

    model_id: str
    cutoff: date
    source: str
    pre_cutoff: Mapping[date, bool]   # decision_date -> decision_date <= cutoff

    @property
    def pre_count(self) -> int:
        return sum(self.pre_cutoff.values())

    @property
    def post_count(self) -> int:
        return len(self.pre_cutoff) - self.pre_count

    def to_dict(self) -> dict[str, Any]:
        return {"model_id": self.model_id, "cutoff": self.cutoff.isoformat(), "source": self.source,
                "rule": "pre_cutoff = decision_date <= cutoff", "pre_cutoff_months": self.pre_count,
                "post_cutoff_months": self.post_count}


@dataclass(frozen=True)
class CutoffClasses:
    """Models with a sourced cutoff (``included``) and the others with the reason (``excluded``)."""

    included: Mapping[str, ModelCutoff]
    excluded: Mapping[str, str]


def _model_ids(model_ids: Any) -> tuple[str, ...]:
    if isinstance(model_ids, (str, bytes)) or not isinstance(model_ids, Sequence):
        raise TypeError("model_ids must be a sequence of model ids")
    ids = tuple(model_ids)
    if not ids or any(not isinstance(model_id, str) or not model_id.strip() for model_id in ids):
        raise ValueError("model_ids must hold at least one nonempty model id")
    if len(set(ids)) != len(ids):
        raise ValueError("model_ids must not repeat a model")
    return ids


def classify_cutoffs(months: Sequence[date], model_ids: Sequence[str], *,
                     registry: Path | str = DEFAULT_CUTOFFS) -> CutoffClasses:
    """Classify each decision month before (on or before the cutoff) or after each model's sourced cutoff.

    A model missing from the registry, or whose registry line cites no source, is
    excluded with the reason (6.4) and gets no classes.
    """
    months = _validated_months(months)
    ids = _model_ids(model_ids)
    registered = sourced_cutoffs(registry)
    name = Path(registry).name
    included: dict[str, ModelCutoff] = {}
    excluded: dict[str, str] = {}
    for model_id in ids:
        entry = registered.get(model_id)
        if entry is None:
            excluded[model_id] = (f"{model_id} has no training cutoff in the sourced cutoff registry {name}, so "
                                  "its decision months cannot be classified before or after the cutoff")
        elif entry[1] is None:
            excluded[model_id] = (f"the {name} entry for {model_id} cites no source, so its cutoff is not a "
                                  "sourced cutoff")
        else:
            included[model_id] = ModelCutoff(model_id, entry[0], entry[1],
                                             {day: day <= entry[0] for day in months})
    return CutoffClasses(included=included, excluded=excluded)


# ---------- EDGAR items ----------

@dataclass(frozen=True)
class EdgarItem:
    """One SEC EDGAR release of the frozen corpus, with its split and filing quarter."""

    item_id: str
    split: str            # calibration | negative_control
    source_file: str      # path relative to the corpus folder
    line: int             # 1-based line in that file
    label: int            # 1 memorized, 0 control
    published_at: date
    quarter: str
    accession: str
    cik: str
    company: str
    release_text: str     # the record's ``prompt`` field, unchanged
    row_sha256: str       # SHA-256 of the record's canonical JSON

    @property
    def prompt(self) -> str:
        return render_outlook_prompt(self.release_text)

    @property
    def prompt_sha256(self) -> str:
        return prompt_sha256(self.prompt)

    def to_dict(self) -> dict[str, Any]:
        return {"item_id": self.item_id, "kind": "edgar", "split": self.split, "source_file": self.source_file,
                "line": self.line, "label": self.label, "published_at": self.published_at.isoformat(),
                "quarter": self.quarter, "accession": self.accession, "cik": self.cik, "company": self.company,
                "release_text": self.release_text, "row_sha256": self.row_sha256,
                "prompt_sha256": self.prompt_sha256}


@dataclass(frozen=True)
class EdgarItems:
    """The calibration and negative-control releases, the corpus file hashes and the unused file."""

    calibration: tuple[EdgarItem, ...]
    negative_control: tuple[EdgarItem, ...]
    files: Mapping[str, Mapping[str, Any]]   # file -> role, sha256, rows, label
    unused: Mapping[str, str]                # file -> reason


def _jsonl_rows(data: bytes, name: str) -> list[tuple[int, Mapping[str, Any]]]:
    rows = []
    # Split on "\n" only: str.splitlines would also split on U+2028 or U+0085 inside a JSON string.
    for number, line in enumerate(data.decode("utf-8").split("\n"), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            raise ValueError(f"EDGAR file {name} line {number} is not JSON") from None
        if not isinstance(row, Mapping):
            raise ValueError(f"EDGAR file {name} line {number} must be an object")
        rows.append((number, row))
    return rows


def _edgar_item(row: Mapping[str, Any], name: str, line: int, split: str, label: int) -> EdgarItem:
    where = f"EDGAR file {name} line {line}"
    release = row.get("prompt")
    if not isinstance(release, str) or not release.strip():
        raise ValueError(f"{where} has no release text")
    if isinstance(row.get("label"), bool) or row.get("label") != label:
        raise ValueError(f"{where} has label {row.get('label')!r}, but the manifest gives the file label {label}")
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{where} has no metadata object")
    published = _iso_date(metadata.get("published_at"), f"{where} published_at")
    accession = metadata.get("accession")
    if not isinstance(accession, str) or _ACCESSION.fullmatch(accession) is None:
        raise ValueError(f"{where} has no SEC accession number")
    for field_name in ("cik", "company"):
        if not isinstance(metadata.get(field_name), str) or not metadata[field_name].strip():
            raise ValueError(f"{where} has no {field_name}")
    return EdgarItem(item_id=f"edgar-{split}-{accession}", split=split, source_file=name, line=line, label=label,
                     published_at=published, quarter=Quarter.of(published).label, accession=accession,
                     cik=metadata["cik"], company=metadata["company"], release_text=release,
                     row_sha256=_sha256_text(_canonical(row)))


def load_edgar_items(edgar_dir: Path | str = DEFAULT_EDGAR_DIR) -> EdgarItems:
    """The EDGAR study items from the frozen corpus in ``edgar_dir``.

    Calibration rows are ``calibration/is_memorized.jsonl`` (label 1) and
    ``calibration/oos_control.jsonl`` (label 0); the negative-control items are
    ``held_out/oos_control.jsonl``; ``held_out/is_memorized.jsonl`` is recorded as
    unused. Every file must match the SHA-256, row count and label of
    ``manifest.json``, so a changed row set is refused with ``ValueError`` naming
    the file. Items keep the file order, which is by publication date.
    """
    edgar_dir = _directory(edgar_dir, "edgar_dir")
    manifest = _json_body((edgar_dir / "manifest.json").read_text(encoding="utf-8"), "EDGAR manifest")
    entries = manifest.get("files") if isinstance(manifest, Mapping) else None
    if not isinstance(entries, Mapping):
        raise ValueError("EDGAR manifest must hold a files mapping")
    splits: dict[str, list[EdgarItem]] = {"calibration": [], "negative_control": []}
    files: dict[str, dict[str, Any]] = {}
    unused: dict[str, str] = {}
    for name, role in EDGAR_FILES.items():
        entry = entries.get(name)
        if not isinstance(entry, Mapping) or entry.get("label") not in (0, 1) or isinstance(entry.get("label"), bool):
            raise ValueError(f"EDGAR manifest has no labelled entry for {name}")
        data = (edgar_dir / name).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != entry.get("sha256"):
            raise ValueError(f"EDGAR file {name} does not match its manifest sha256, so the frozen row set changed")
        rows = _jsonl_rows(data, name)
        if len(rows) != entry.get("rows"):
            raise ValueError(f"EDGAR file {name} holds {len(rows)} rows, but the manifest gives {entry.get('rows')!r}")
        files[name] = {"role": role, "sha256": digest, "rows": len(rows), "label": entry["label"]}
        if role == "unused":
            unused[name] = EDGAR_UNUSED_REASON
            continue
        splits[role].extend(_edgar_item(row, name, line, role, entry["label"]) for line, row in rows)
    items = splits["calibration"] + splits["negative_control"]
    if len({item.item_id for item in items}) != len(items):
        raise ValueError("EDGAR items repeat an accession within one split")
    return EdgarItems(calibration=tuple(splits["calibration"]), negative_control=tuple(splits["negative_control"]),
                      files=files, unused=unused)


# ---------- assembly ----------

@dataclass(frozen=True)
class MacroItem:
    """One included decision month: its macro row, the latest observation used per series and its outcome."""

    item_id: str
    row: MacroRow
    observed_through: Mapping[str, str]   # series -> ISO date of the latest observation in its as-of vector
    outcome: RealizedOutcome

    @property
    def decision_date(self) -> date:
        return self.row.decision_date

    def prompts(self) -> dict[str, str]:
        """Both renderings of the row (6.1)."""
        return {"anonymised": render_anonymised(self.row), "identifying": render_identifying(self.row)}

    def to_dict(self, pre_cutoff: Mapping[str, bool]) -> dict[str, Any]:
        row = row_to_dict(self.row)
        outcome = self.outcome.to_dict()
        return {"item_id": self.item_id, "kind": "macro", "decision_date": self.decision_date.isoformat(),
                "row": row, "row_sha256": _sha256_text(_canonical(row)),
                "prompt_sha256": {name: prompt_sha256(text) for name, text in self.prompts().items()},
                "observed_through": dict(self.observed_through), "realized": outcome["realized"],
                "realized_missing_reason": outcome["reason"], "pre_cutoff": dict(pre_cutoff)}


@dataclass(frozen=True)
class ItemAssembly:
    """Every study item with its sources, exclusions and private files."""

    months: tuple[date, ...]
    macro: tuple[MacroItem, ...]
    excluded_months: Mapping[date, str]
    outcomes: tuple[RealizedOutcome, ...]
    edgar: EdgarItems
    cutoffs: CutoffClasses
    price_window: tuple[date, date]
    complete_through: date
    paths: PrivatePaths

    def to_dict(self) -> dict[str, Any]:
        """The private ``items.json`` content; paths are relative to the private root, so it is machine independent."""
        included = self.cutoffs.included
        macro = [item.to_dict({model_id: cutoff.pre_cutoff[item.decision_date]
                               for model_id, cutoff in included.items()}) for item in self.macro]
        revised = {series: sum(item.row.revised[series] for item in self.macro) for series in MACRO_SERIES}
        start, end = self.price_window
        return {
            "schema": ITEMS_SCHEMA,
            "decision_months": {"first": self.months[0].isoformat(), "last": self.months[-1].isoformat(),
                                "count": len(self.months), "rule": "last calendar day of each month"},
            "sources": {
                "macro": {"series": list(MACRO_SERIES), "loader": "recall_guard.dataset.fred.fetch_series_as_of",
                          "view": "as available on the decision date where FRED has vintages, else the current "
                                  "vintage flagged revised"},
                "prices": {"provider": "FMP", "endpoint": "historical-price-eod/dividend-adjusted",
                           "field": "adjClose", "symbols": dict(PRICE_SYMBOLS), "window": [start.isoformat(),
                                                                                         end.isoformat()]},
                "cash": {"series": CASH_SERIES, "vintage": "current", "rule": "DTB3 / 100 / 12"},
                "complete_through": self.complete_through.isoformat(),
            },
            "licensing": LICENSING_NOTE,
            "models": {"included": {model_id: cutoff.to_dict() for model_id, cutoff in included.items()},
                       "excluded": dict(self.cutoffs.excluded)},
            "macro_items": macro,
            "excluded_months": [{"decision_date": day.isoformat(), "reason": reason}
                                for day, reason in sorted(self.excluded_months.items())],
            "edgar": {"calibration": [item.to_dict() for item in self.edgar.calibration],
                      "negative_control": [item.to_dict() for item in self.edgar.negative_control],
                      "files": {name: dict(entry) for name, entry in self.edgar.files.items()},
                      "unused": dict(self.edgar.unused)},
            "private_files": {name: getattr(self.paths, name).relative_to(self.paths.root).as_posix()
                              for name in ("realized_returns", "fmp_cache", "fred_cache")},
            "summary": {"decision_months": len(self.months), "macro_items": len(self.macro),
                        "excluded_months": len(self.excluded_months),
                        "realized_missing": sum(item.outcome.realized is None for item in self.macro),
                        "revised_months": revised, "models_included": sorted(included),
                        "models_excluded": sorted(self.cutoffs.excluded),
                        "edgar_calibration": len(self.edgar.calibration),
                        "edgar_negative_control": len(self.edgar.negative_control)},
        }


def _classified(source: str, call: Callable[[], T]) -> T:
    """Run one source fetch and classify its failure as an :class:`ItemSourceError` without exception context."""
    try:
        return call()
    except RequestError as error:
        failure = (error.code or "request_error", bool(error.retryable) or error.status_code in RETRYABLE_STATUS,
                   error.status_code, str(error), error.detail)
    except ValueError as error:
        failure = ("invalid_body", False, None, str(error), None)
    code, retryable, status_code, message, detail = failure
    raise ItemSourceError(f"{source}: {message}", source=source, code=code, retryable=retryable,
                          status_code=status_code, detail=detail)


def _returns_text(outcomes: Sequence[RealizedOutcome], window: tuple[date, date], complete_through: date) -> str:
    document = {"schema": RETURNS_SCHEMA, "licensing": LICENSING_NOTE,
                "sources": {"prices": "FMP historical-price-eod/dividend-adjusted adjClose",
                            "symbols": dict(PRICE_SYMBOLS), "cash": f"FRED {CASH_SERIES} current vintage / 100 / 12",
                            "window": [window[0].isoformat(), window[1].isoformat()]},
                "complete_through": complete_through.isoformat(),
                "months": [outcome.to_dict() for outcome in outcomes]}
    return json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"


def assemble_items(*, fred_api_key: str, fmp_api_key: str, model_ids: Sequence[str],
                   private_root: Path | str = PRIVATE_ROOT, months: Sequence[date] | None = None,
                   cutoffs_path: Path | str = DEFAULT_CUTOFFS, edgar_dir: Path | str = DEFAULT_EDGAR_DIR,
                   today: date | None = None, transport: Transport = requests_transport) -> ItemAssembly:
    """Assemble every study item; raw licensed data and the realized returns stay under ``private_root``.

    Order: inputs are validated before any request; the cutoff classes and the
    EDGAR items need no network; then FMP closes and DTB3 (through
    ``transport``), whose realized returns are written to the private returns
    file; then one FRED loader call per series and decision month (cached under
    the private FRED folder). The next month counts as complete when it ends on or
    before ``today`` minus one day and inside the price window. A failed fetch
    raises :class:`ItemSourceError`; an uncomputable month is excluded with the
    reason. ``items.json`` is written by :func:`write_items`.
    """
    months = _validated_months(decision_months() if months is None else months)
    fred_key = _key(fred_api_key, FRED_KEY_NAME)
    fmp_key = _key(fmp_api_key, FMP_KEY_NAME)
    today = date.today() if today is None else _plain_date(today, "today")
    paths = PrivatePaths(_directory(private_root, "private_root"))
    cutoffs = classify_cutoffs(months, model_ids, registry=cutoffs_path)
    edgar = load_edgar_items(edgar_dir)

    window = price_window(months)
    complete_through = min(window[1], today - timedelta(days=1))
    closes = {category: _classified(f"FMP {symbol}", partial(
        fetch_adjusted_closes, symbol, start=window[0], end=window[1], api_key=fmp_key, cache_dir=paths.fmp_cache,
        transport=transport)) for category, symbol in PRICE_SYMBOLS.items()}
    cash = _classified(f"FRED {CASH_SERIES}", partial(
        fetch_cash_rates, start=window[0], end=window[1], api_key=fred_key, cache_dir=paths.fred_cache,
        transport=transport))
    outcomes = realized_returns(months, closes, cash, complete_through=complete_through)
    _write_atomic(paths.realized_returns, _returns_text(outcomes, window, complete_through))

    macro: list[MacroItem] = []
    excluded: dict[date, str] = {}
    for decision_date, outcome in zip(months, outcomes, strict=True):
        vectors = {series_id: _classified(f"FRED {series_id}", partial(
            fred.fetch_series_as_of, series_id, decision_date, api_key=fred_key, cache_dir=paths.fred_cache))
            for series_id in MACRO_SERIES}
        try:
            row = build_macro_row(decision_date, vectors)
        except ExcludedMonth as exclusion:
            excluded[decision_date] = exclusion.reason
            continue
        macro.append(MacroItem(item_id=f"macro-{decision_date:%Y-%m}", row=row, outcome=outcome,
                               observed_through={series_id: vector[-1].observation_date.isoformat()
                                                 for series_id, vector in vectors.items()}))
    return ItemAssembly(months=months, macro=tuple(macro), excluded_months=excluded, outcomes=outcomes,
                        edgar=edgar, cutoffs=cutoffs, price_window=window, complete_through=complete_through,
                        paths=paths)


def write_items(assembly: ItemAssembly, path: Path | str | None = None) -> str:
    """Write ``assembly.to_dict()`` as private JSON (``plan/items.json`` by default) and return its SHA-256."""
    if not isinstance(assembly, ItemAssembly):
        raise TypeError("assembly must be an ItemAssembly")
    target = assembly.paths.items if path is None else _directory(path, "path")
    text = json.dumps(assembly.to_dict(), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    _write_atomic(target, text)
    return _sha256_text(text)
