"""Leakage, missing-data, parsing and scoring checks for the S&P mechanics pilot."""

import hashlib
import json
import math
from dataclasses import replace
from datetime import date, timedelta
from decimal import Decimal

import pytest

from scripts.prepare_completion_inputs import sessions
from scripts.prepare_sp500_pilot import build, freeze, validate
from scripts.sp500_inputs import (
    Observation,
    PriceWindow,
    cohort_ranges,
    parse_snapshot,
    render_prompts,
    select_matched_dates,
    window_record,
    windows,
)
from scripts.sp500_scoring import paired_advantage, parse_answer, score_answer


def sample_window():
    days = sessions("2024-01-02", "2024-02-15")[:21]
    observations = tuple(Observation(date.fromisoformat(day), Decimal(100 + index))
                         for index, day in enumerate(days))
    return PriceWindow(observations[-1], observations[:-1])


def test_source_missing_stays_missing_and_invalid_order_is_rejected():
    raw = b"observation_date,SP500\n2024-01-02,100.00\n2024-01-03,.\n2024-01-04,102.00\n"
    rows = parse_snapshot(raw, date(2024, 1, 4))
    assert rows[1].close is None
    with pytest.raises(ValueError, match="strictly increasing"):
        parse_snapshot(raw + b"2024-01-04,103.00\n", date(2024, 1, 4))
    with pytest.raises(ValueError, match="boundary"):
        parse_snapshot(raw, date(2024, 1, 3))


@pytest.mark.parametrize("value", ["NaN", "-1", "0", "Infinity", "invalid"])
def test_invalid_prices_fail_closed(value):
    with pytest.raises(ValueError):
        parse_snapshot(f"observation_date,SP500\n2024-01-02,{value}\n".encode(), date(2024, 1, 2))


def test_context_does_not_bridge_missing_exchange_session():
    window = sample_window()
    observations = list((*window.history, window.target))
    observations[4] = replace(observations[4], close=None)
    observations.insert(0, Observation(date(2023, 12, 29), Decimal(99)))
    candidates, rejected = windows(tuple(observations), window.target.day, window.target.day)
    assert not candidates
    assert rejected == [window.target.day.isoformat()]


def test_target_value_does_not_enter_prompts_or_volatility():
    window = sample_window()
    mutated = replace(window, target=replace(window.target, close=Decimal("999999")))
    assert render_prompts(window) == render_prompts(mutated)
    assert window.volatility == mutated.volatility
    assert "999999" not in "".join(render_prompts(mutated).values())


def test_anonymous_and_identified_numeric_inputs_are_identical():
    window = sample_window()
    prompts = render_prompts(window)
    marker = "observation,value\n"
    def numbers(arm):
        return [row.split(",")[1] for row in prompts[arm].split(marker)[1].splitlines()]
    assert numbers("identified") == numbers("anonymous")
    assert len(numbers("anonymous")) == 20
    assert numbers("anonymous")[0] == "100.00000000"
    assert "S&P" not in prompts["anonymous"]
    assert window.target.day.isoformat() not in prompts["anonymous"]
    assert all(row.day.isoformat() not in prompts["anonymous"] for row in window.history)


def test_cohort_ranges_exclude_cutoff_month_and_extra_buffer():
    ranges = cohort_ranges(date(2024, 6, 30), date(2026, 11, 3))
    assert ranges["pre"][1] == date(2024, 5, 1)
    assert ranges["post"][0] == date(2024, 7, 31)
    assert all((end - start).days == 89 for start, end in ranges.values())


def test_selection_is_deterministic_and_ignores_target_close():
    base = sample_window()
    candidates = [
        PriceWindow(replace(base.target, day=base.target.day + timedelta(days=index)),
                    tuple(replace(row, close=row.close + Decimal(index)) for row in base.history))
        for index in range(20)
    ]
    selected = select_matched_dates(candidates, candidates)
    mutated = [replace(window, target=replace(window.target, close=Decimal("987654"))) for window in candidates]
    selected_mutated = select_matched_dates(mutated, mutated)
    assert [(quartile, before.target.day, after.target.day) for quartile, before, after in selected] == [
        (quartile, before.target.day, after.target.day) for quartile, before, after in selected_mutated]
    assert len({after.target.day for _, _, after in selected}) == 4


@pytest.mark.parametrize("content", [
    '{"value": true}', '{"value": NaN}', '{"value": -1}', '{"value": 0}',
    '{"value": "123"}', '{"value": 1, "value": 2}', '{"value": 1, "extra": 2}',
    "123", "null", '[123]', 'Commentary: {"value":123}',
    '{"value":1e9999}', '{"value":1e-9999}',
])
def test_invalid_answers_are_not_silent_predictions(content):
    assert parse_answer(content)["status"] == "invalid"


def test_abstention_remains_distinct_from_invalid():
    assert parse_answer('{"value":null}') == {"status": "abstained", "value": None}
    assert parse_answer('{"value":123.45}') == {"status": "parsed", "value": "123.45"}


def test_exact_continuation_and_flat_baseline_have_expected_losses():
    item = window_record("test", "pre", 0, sample_window())
    perfect = score_answer(item, "identified", '{"value":120}')
    unchanged = score_answer(item, "anonymous", '{"value":119}')
    assert perfect["normalized_absolute_error"] == pytest.approx(0, abs=1e-10)
    assert unchanged["normalized_absolute_error"] == pytest.approx(unchanged["unchanged_close_normalized_error"])
    assert paired_advantage(perfect, unchanged) > 0
    assert perfect["observed_log_return"] == pytest.approx(math.log(120 / 119))


def test_missing_pair_is_unavailable_not_zero_and_mismatch_fails():
    item = window_record("test", "post", 0, sample_window())
    identified = score_answer(item, "identified", '{"value":120}')
    anonymous = score_answer(item, "anonymous", '{"value":null}')
    assert paired_advantage(identified, anonymous) is None
    with pytest.raises(ValueError, match="same item"):
        paired_advantage(identified, anonymous | {"item_id": "other"})


def test_recall_precision_uses_half_up():
    item = window_record("test", "pre", 0, sample_window())
    item["target_close"] = "120.01"
    assert score_answer(item, "recall", '{"value":120.005}')["exact_at_two_decimals"]
    assert not score_answer(item, "recall", '{"value":120.004}')["exact_at_two_decimals"]


def test_local_bundle_roundtrip_and_tamper_detection(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    days = sessions("2023-12-01", "2026-11-03")
    raw = ("observation_date,SP500\n" + "".join(
        f"{day},{4000 + index + (index % 7) * 0.3:.2f}\n" for index, day in enumerate(days))).encode()
    (source / "SP500.csv").write_bytes(raw)
    (source / "source.json").write_text(json.dumps({"series": "SP500", "sha256": hashlib.sha256(raw).hexdigest()}))
    plan, items, requests = build(source, date(2026, 11, 4))
    assert len(items) == 16 and len(requests) == 48
    assert plan["budget"]["maximum_physical_requests"] == 50
    assert all(item["split"] == "development_mechanics_only" for item in items)
    output = tmp_path / "pilot"
    freeze(source, output, date(2026, 11, 4))
    validate(source, output, date(2026, 11, 4))
    with pytest.raises(FileExistsError):
        freeze(source, output, date(2026, 11, 4))
    (output / "requests.json").write_text("[]")
    with pytest.raises(ValueError, match="mismatch"):
        validate(source, output, date(2026, 11, 4))


def test_summary_retains_failed_unperformed_and_common_usable_counts():
    from scripts.sp500_scoring import summarize_scores

    pre = window_record("test", "pre", 0, sample_window())
    post = window_record("test", "post", 0, sample_window())
    scores = [
        score_answer(pre, "identified", '{"value":120}'),
        score_answer(pre, "anonymous", '{"value":119}'),
        score_answer(pre, "recall", '{"value":null}'),
        score_answer(post, "identified", '{"value":120}'),
        {"item_id": post["item_id"], "arm": "anonymous", "status": "failed"},
    ]
    result = summarize_scores([pre, post], scores)["test"]
    assert result["cohorts"]["pre"]["common_usable_pairs"] == 1
    assert result["cohorts"]["post"]["common_usable_pairs"] == 0
    assert result["cohorts"]["post"]["arms"]["recall"]["status_counts"] == {"unperformed": 1}
    assert result["cohorts"]["pre"]["arms"]["recall"]["status_counts"] == {"abstained": 1}
    assert result["cutoff_contrast"] is None
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_scores([pre, post], scores + [scores[0]])


def test_cutoff_contrast_sign_matches_the_pre_cutoff_advantage():
    from scripts.sp500_scoring import summarize_scores

    pre = window_record("test", "pre", 0, sample_window())
    post = window_record("test", "post", 0, sample_window())
    scores = [score_answer(pre, "identified", '{"value":120}'),
              score_answer(pre, "anonymous", '{"value":119}'),
              score_answer(post, "identified", '{"value":119}'),
              score_answer(post, "anonymous", '{"value":119}')]
    assert summarize_scores([pre, post], scores)["test"]["cutoff_contrast"] > 0
