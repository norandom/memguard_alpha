import dataclasses

import pytest
import requests

from recall_guard.core.nvidia_lm import CompletionResult, NvidiaLM, TokenLogprob


def _build_mock_response(mocker, content="Bullish", with_logprobs=True):
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    payload = {
        "choices": [
            {
                "message": {"content": content},
            }
        ]
    }
    if with_logprobs:
        payload["choices"][0]["logprobs"] = {
            "content": [
                {
                    "token": "Bull",
                    "logprob": -0.1,
                    "top_logprobs": [
                        {"token": "Bull", "logprob": -0.1},
                        {"token": "Bear", "logprob": -1.5},
                    ],
                },
                {
                    "token": "ish",
                    "logprob": -0.2,
                    "top_logprobs": [
                        {"token": "ish", "logprob": -0.2},
                        {"token": "y", "logprob": -2.0},
                    ],
                },
            ]
        }
    mock_response.json.return_value = payload
    return mock_response


def test_request_body_contains_temperature_and_top_logprobs(mocker):
    mock_post = mocker.patch("requests.post")
    mock_post.return_value = _build_mock_response(mocker)

    lm = NvidiaLM(api_key="test_key", model="nvidia/nemotron-3-super-120b-a12b")
    lm.generate("Predict the stock direction.")

    called_json = mock_post.call_args.kwargs["json"]
    assert called_json["temperature"] == 0.0
    assert called_json["logprobs"] is True
    assert called_json["top_logprobs"] == 20


def test_response_parsed_into_completion_result(mocker):
    mock_post = mocker.patch("requests.post")
    mock_post.return_value = _build_mock_response(mocker)

    lm = NvidiaLM(api_key="test_key", model="nvidia/nemotron-3-super-120b-a12b")
    result = lm.generate("Predict the stock direction.")

    assert isinstance(result, CompletionResult)
    assert result.content == "Bullish"
    assert len(result.logprobs) == 2
    first = result.logprobs[0]
    assert isinstance(first, TokenLogprob)
    assert first.token == "Bull"
    assert first.logprob == -0.1
    assert len(first.top_logprobs) == 2
    assert first.top_logprobs[0]["token"] == "Bull"


def test_completion_result_is_frozen_dataclass():
    assert dataclasses.is_dataclass(CompletionResult)
    params = CompletionResult.__dataclass_params__
    assert params.frozen is True


def test_token_logprob_is_frozen_dataclass():
    assert dataclasses.is_dataclass(TokenLogprob)
    params = TokenLogprob.__dataclass_params__
    assert params.frozen is True


def test_temperature_override_passed_through(mocker):
    mock_post = mocker.patch("requests.post")
    mock_post.return_value = _build_mock_response(mocker)

    lm = NvidiaLM(api_key="test_key", model="nvidia/nemotron-3-super-120b-a12b")
    lm.generate("Predict the stock direction.", temperature=0.7)

    called_json = mock_post.call_args.kwargs["json"]
    assert called_json["temperature"] == 0.7


def test_timeout_raises_timeout_error(mocker):
    mock_post = mocker.patch("requests.post")
    mock_post.side_effect = requests.exceptions.Timeout("boom")

    lm = NvidiaLM(api_key="test_key", model="nvidia/nemotron-3-super-120b-a12b")
    with pytest.raises(TimeoutError):
        lm.generate("Predict.")


def test_default_timeout_is_fifteen_seconds(mocker):
    mock_post = mocker.patch("requests.post")
    mock_post.return_value = _build_mock_response(mocker)

    lm = NvidiaLM(api_key="test_key", model="nvidia/nemotron-3-super-120b-a12b")
    lm.generate("Predict.")

    assert mock_post.call_args.kwargs["timeout"] == 15.0


def test_missing_top_logprobs_raises_clear_error(mocker):
    mock_post = mocker.patch("requests.post")
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {"content": "Bullish"},
                "logprobs": {
                    "content": [
                        {"token": "Bull", "logprob": -0.1},
                    ]
                },
            }
        ]
    }
    mock_post.return_value = mock_response

    lm = NvidiaLM(api_key="test_key", model="nvidia/nemotron-3-super-120b-a12b")
    with pytest.raises(RuntimeError, match=r"top_logprobs"):
        lm.generate("Predict.")


def test_missing_logprobs_section_raises_clear_error(mocker):
    mock_post = mocker.patch("requests.post")
    mock_post.return_value = _build_mock_response(mocker, with_logprobs=False)

    lm = NvidiaLM(api_key="test_key", model="nvidia/nemotron-3-super-120b-a12b")
    with pytest.raises(RuntimeError):
        lm.generate("Predict.")


def test_choices_zero_none_raises_runtime_error(mocker):
    mock_post = mocker.patch("requests.post")
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"choices": [None]}
    mock_post.return_value = mock_response

    lm = NvidiaLM(api_key="test_key", model="nvidia/nemotron-3-super-120b-a12b")
    with pytest.raises(RuntimeError, match=r"choices\[0\]"):
        lm.generate("Predict.")


def test_logprobs_content_none_raises_runtime_error(mocker):
    mock_post = mocker.patch("requests.post")
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {"content": "Bullish"},
                "logprobs": {"content": None},
            }
        ]
    }
    mock_post.return_value = mock_response

    lm = NvidiaLM(api_key="test_key", model="nvidia/nemotron-3-super-120b-a12b")
    with pytest.raises(RuntimeError, match=r"logprobs.content"):
        lm.generate("Predict.")


def test_scalar_token_entry_raises_runtime_error(mocker):
    mock_post = mocker.patch("requests.post")
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {"content": "Bullish"},
                "logprobs": {"content": [7]},
            }
        ]
    }
    mock_post.return_value = mock_response

    lm = NvidiaLM(api_key="test_key", model="nvidia/nemotron-3-super-120b-a12b")
    with pytest.raises(RuntimeError, match=r"malformed"):
        lm.generate("Predict.")


def test_missing_realized_logprob_raises_runtime_error(mocker):
    mock_post = mocker.patch("requests.post")
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {"content": "Bullish"},
                "logprobs": {
                    "content": [
                        {
                            "token": "Bull",
                            "top_logprobs": [{"token": "Bull", "logprob": -0.1}],
                        }
                    ]
                },
            }
        ]
    }
    mock_post.return_value = mock_response

    lm = NvidiaLM(api_key="test_key", model="nvidia/nemotron-3-super-120b-a12b")
    with pytest.raises(RuntimeError, match=r"missing 'logprob'"):
        lm.generate("Predict.")


def test_concurrent_pacing_enforces_min_interval(mocker):
    """4 workers sharing one paced instance must not burst; request starts
    are spaced by at least min_call_interval_s."""
    import time

    from recall_guard.core.nvidia_lm import generate_many

    starts: list[float] = []

    def _recording_post(*args, **kwargs):
        starts.append(time.monotonic())
        return _build_mock_response(mocker)

    mocker.patch("requests.post", side_effect=_recording_post)

    interval = 0.05
    lm = NvidiaLM(
        api_key="test_key",
        model="nvidia/nemotron-3-super-120b-a12b",
        min_call_interval_s=interval,
    )
    results = generate_many(lm, ["p1", "p2", "p3", "p4"], max_workers=4)

    assert all(isinstance(r, CompletionResult) for r in results)
    assert len(starts) == 4
    ordered = sorted(starts)
    gaps = [b - a for a, b in zip(ordered, ordered[1:], strict=False)]
    # Allow a small scheduling tolerance below the nominal interval.
    assert all(gap >= interval * 0.8 for gap in gaps), gaps


def test_concurrent_calls_actually_overlap(mocker):
    """Requests through one client must be in flight simultaneously.

    The pacing lock previously wrapped the blocking POST, so every concurrent
    call serialised -- and because the lock was taken unconditionally, this
    happened even here, with pacing disabled. Counting peak in-flight requests
    detects that directly, without depending on machine speed.

    Deliberately not a ``threading.Barrier``: under the serialised client a
    barrier deadlocks (hangs) rather than failing.
    """
    import threading
    import time

    from recall_guard.core.nvidia_lm import generate_many

    lock = threading.Lock()
    in_flight = 0
    peak = 0

    def _tracking_post(*args, **kwargs):
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        try:
            time.sleep(0.2)
            return _build_mock_response(mocker)
        finally:
            with lock:
                in_flight -= 1

    mocker.patch("requests.post", side_effect=_tracking_post)

    # Default min_call_interval_s=0.0 -- pacing disabled is the point.
    lm = NvidiaLM(api_key="test_key", model="nvidia/nemotron-3-super-120b-a12b")
    results = generate_many(lm, ["p"] * 8, max_workers=8)

    assert all(isinstance(r, CompletionResult) for r in results)
    assert peak >= 4, f"only {peak} concurrent POST(s); calls are serialised"


def test_concurrent_wall_clock_scales_with_workers(mocker):
    """Duration must track the worker count, not the request count.

    Serialised, 8 requests at 0.2s each take ~1.6s; genuinely concurrent they
    take ~0.2s. The 0.8s threshold sits a comfortable factor of two from both,
    so the assertion is not a machine-speed measurement.
    """
    import time

    from recall_guard.core.nvidia_lm import generate_many

    def _slow_post(*args, **kwargs):
        time.sleep(0.2)
        return _build_mock_response(mocker)

    mocker.patch("requests.post", side_effect=_slow_post)

    lm = NvidiaLM(api_key="test_key", model="nvidia/nemotron-3-super-120b-a12b")
    started = time.monotonic()
    generate_many(lm, ["p"] * 8, max_workers=8)
    elapsed = time.monotonic() - started

    assert elapsed < 0.8, f"8 requests over 8 workers took {elapsed:.2f}s; serialised"


def test_pacing_spaces_starts_independent_of_latency(mocker):
    """Pacing must space request *starts*, not completion-to-start.

    With the lock held across the POST the effective spacing was
    ``interval + round_trip_time``. Reserving the slot makes it exactly
    ``interval``, independent of how slow the endpoint is.
    """
    import time

    from recall_guard.core.nvidia_lm import generate_many

    starts: list[float] = []
    lock = __import__("threading").Lock()

    def _slow_recording_post(*args, **kwargs):
        with lock:
            starts.append(time.monotonic())
        time.sleep(0.2)
        return _build_mock_response(mocker)

    mocker.patch("requests.post", side_effect=_slow_recording_post)

    interval = 0.05
    lm = NvidiaLM(
        api_key="test_key",
        model="nvidia/nemotron-3-super-120b-a12b",
        min_call_interval_s=interval,
    )
    generate_many(lm, ["p"] * 6, max_workers=6)

    ordered = sorted(starts)
    gaps = [b - a for a, b in zip(ordered, ordered[1:], strict=False)]

    # Pacing preserved: every gap clears the configured interval. This is the
    # half that rejects a reservation scheme handing the same slot out twice.
    assert all(gap >= interval * 0.8 for gap in gaps), gaps

    # Not latency-inflated: before the repair every gap was interval + latency
    # (0.25s here), so the median sits an order of magnitude away from either
    # answer. Asserted on the median rather than on every gap because a single
    # thread waking late under load stretches one gap without saying anything
    # about the pacing contract -- and this test previously flaked exactly that
    # way under suite load while passing in isolation.
    median_gap = sorted(gaps)[len(gaps) // 2]
    assert median_gap <= interval + 0.07, gaps


def test_failed_attempt_still_consumes_a_pacing_slot(mocker):
    """A transport failure must not let its retry fire unpaced.

    Only exceptions raised by ``requests.post`` itself skip the pacing stamp;
    an HTTP 429 is a *successful* POST and was already paced. A fast-failing
    connection error is the genuinely exposed case -- a real timeout costs
    ``timeout_s`` of wall clock and so self-paces.
    """
    import time

    starts: list[float] = []
    calls = {"n": 0}

    def _failing_then_ok(*args, **kwargs):
        starts.append(time.monotonic())
        calls["n"] += 1
        if calls["n"] <= 2:
            raise requests.exceptions.ConnectionError("connection reset")
        return _build_mock_response(mocker)

    mocker.patch("requests.post", side_effect=_failing_then_ok)

    interval = 0.05
    lm = NvidiaLM(
        api_key="test_key",
        model="nvidia/nemotron-3-super-120b-a12b",
        min_call_interval_s=interval,
        retry_backoff_s=0.0,
    )
    lm.generate("Predict.")

    assert len(starts) == 3
    gaps = [b - a for a, b in zip(starts, starts[1:], strict=False)]
    assert all(gap >= interval * 0.8 for gap in gaps), gaps


def test_retry_after_header_is_honoured(mocker):
    """A rate-limited response carrying Retry-After must wait that long."""
    slept: list[float] = []
    calls = {"n": 0}

    def _rate_limited_then_ok(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            response = mocker.Mock()
            response.status_code = 429
            response.headers = {"Retry-After": "0.3"}
            response.raise_for_status.side_effect = requests.exceptions.HTTPError(
                "429 Too Many Requests", response=response
            )
            return response
        return _build_mock_response(mocker)

    mocker.patch("requests.post", side_effect=_rate_limited_then_ok)
    mocker.patch("time.sleep", side_effect=slept.append)

    lm = NvidiaLM(
        api_key="test_key",
        model="nvidia/nemotron-3-super-120b-a12b",
        retry_backoff_s=2.0,
    )
    lm.generate("Predict.")

    assert any(abs(s - 0.3) < 1e-9 for s in slept), slept


def test_retry_backoff_is_jittered(mocker):
    """Concurrent rate-limited draws must not retry in unison."""
    slept: list[float] = []

    def _always_rate_limited(*args, **kwargs):
        response = mocker.Mock()
        response.status_code = 503
        response.headers = {}
        response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "503 Service Unavailable", response=response
        )
        return response

    mocker.patch("requests.post", side_effect=_always_rate_limited)
    mocker.patch("time.sleep", side_effect=slept.append)

    lm = NvidiaLM(
        api_key="test_key",
        model="nvidia/nemotron-3-super-120b-a12b",
        retry_backoff_s=2.0,
    )
    for _ in range(12):
        slept.clear()
        with pytest.raises(RuntimeError):
            lm.generate("Predict.")
        if len({round(s, 6) for s in slept}) == len(slept) and slept[0] != 2.0:
            break
    else:  # pragma: no cover - only on a degenerate RNG
        pytest.fail(f"backoff never varied across 12 runs: {slept}")

    assert all(0.0 <= s <= 4.0 for s in slept), slept


def test_http_error_carries_status_code(mocker):
    """Credential rejection must be classifiable from the status, not the text."""
    from recall_guard.core.nvidia_lm import LMHTTPError

    def _unauthorized(*args, **kwargs):
        response = mocker.Mock()
        response.status_code = 401
        response.headers = {}
        response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "Client Error", response=response
        )
        return response

    mocker.patch("requests.post", side_effect=_unauthorized)

    lm = NvidiaLM(api_key="test_key", model="nvidia/nemotron-3-super-120b-a12b")
    with pytest.raises(LMHTTPError) as excinfo:
        lm.generate("Predict.")

    assert excinfo.value.status_code == 401
    assert isinstance(excinfo.value, RuntimeError)


def test_empty_top_logprobs_raises_runtime_error(mocker):
    mock_post = mocker.patch("requests.post")
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {"content": "Bullish"},
                "logprobs": {
                    "content": [
                        {"token": "Bull", "logprob": -0.1, "top_logprobs": []}
                    ]
                },
            }
        ]
    }
    mock_post.return_value = mock_response

    lm = NvidiaLM(api_key="test_key", model="nvidia/nemotron-3-super-120b-a12b")
    with pytest.raises(RuntimeError, match=r"top_logprobs"):
        lm.generate("Predict.")
