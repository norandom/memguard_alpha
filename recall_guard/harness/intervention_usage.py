"""Physical-event resource summaries for offline intervention analysis.

Rows here are physical requests, keyed by (execution_id, call_id), never model
responses or condition rows. Condition views overlap because singles reuse
draw zero and rewritten conditions share rewriting/verification overhead.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from math import fsum

from recall_guard.harness.intervention_types import ExperimentCheckpoint, RequestRecord


def _counter(values, denominator):
    """Missing values stay unknown; a known subtotal is never a complete total."""
    known = [value for value in values if value is not None]
    subtotal = (sum(known) if all(isinstance(value, int) for value in known) else fsum(known)) if known else None
    return {"total": subtotal if denominator and len(known) == denominator else None,
            "known_subtotal": subtotal, "recorded_requests": len(known), "requests": denominator}


def _usage_sources(checkpoint):
    """Use a response/diagnostic only when it identifies exactly one physical call.

    Event usage is authoritative, including failed-call usage. With retries or
    multiple callback hooks, raw response usage cannot safely identify a call;
    it is not added to the ledger. No counter sets are merged or estimated.
    """
    associations = {}
    for request in checkpoint.requests:
        key = (request.item_id, request.model_role, request.arm, request.draw_index)
        associations.setdefault(key, []).append(request)
    draws = {draw.identity: draw for draw in checkpoint.draws}
    rewrites = {rewrite.item_id: rewrite for rewrite in checkpoint.rewrites}
    results = {}
    for key, requests in associations.items():
        if len(requests) != 1 or requests[0].event.usage is not None:
            continue
        item_id, role, arm, index = key
        request = requests[0]
        response, diagnostics = None, None
        if role == "target" and (draw := draws.get((item_id, arm, index))) is not None:
            response, diagnostics = draw.response, draw.metadata.get("diagnostics")
        elif role != "target" and (rewrite := rewrites.get(item_id)) is not None:
            response = rewrite.rewrite_response if role == "rewriter" else rewrite.verification_response
            metadata = rewrite.fidelity_checks.get("generation" if role == "rewriter" else "verification", {})
            diagnostics = metadata.get("diagnostics")
        if request.event.status == "succeeded" and response is not None and response.usage is not None:
            results[request.identity] = (response.usage, "unique_response")
        elif request.event.status == "failed" and diagnostics and diagnostics.get("usage") is not None:
            usage = diagnostics["usage"]
            if isinstance(usage, Mapping) and all(isinstance(name, str) and name.strip()
                                                 and type(count) is int and count >= 0 for name, count in usage.items()):
                results[request.identity] = (usage, "unique_failure_diagnostics")
    return results


def summarize_usage(checkpoint: ExperimentCheckpoint, requests: Sequence[RequestRecord]) -> dict:
    """Sum each reported token key independently, grouped by endpoint/model/role.

    Latency is in seconds, summed request work; concurrent elapsed wall time is
    not identified by this sum. RequestEvent.cost has no currency field, so its
    recorded units are retained without inventing a currency or a price table.
    """
    fallback = _usage_sources(checkpoint)
    groups = {}
    for request in sorted(requests, key=lambda value: value.identity):
        config = getattr(checkpoint.protocol, request.model_role + "_model")
        key = (request.model_role, config.provider, config.protocol.value, config.endpoint, config.model)
        groups.setdefault(key, []).append(request)
    summaries = []
    for (role, provider, protocol, endpoint, model), records in groups.items():
        counters, sources, usages = set(), Counter(), []
        for request in records:
            usage, source = ((request.event.usage, "physical_event") if request.event.usage is not None
                             else fallback.get(request.identity, (None, "unavailable")))
            usages.append(usage)
            sources[source] += 1
            if usage is not None:
                counters.update(usage)
        latencies = [record.event.latency_s for record in records if record.event.latency_s is not None]
        summaries.append({
            "role": role, "provider": provider, "protocol": protocol, "endpoint": endpoint, "model": model,
            "physical_requests": len(records), "statuses": dict(sorted(Counter(
                record.event.status for record in records).items())),
            "usage_sources": dict(sorted(sources.items())),
            "counters": {name: _counter([None if usage is None else usage.get(name) for usage in usages], len(records))
                         for name in sorted(counters)},
            "latency_s": {"sum": fsum(latencies) if latencies else None,
                          "mean": fsum(latencies) / len(latencies) if latencies else None,
                          "recorded_requests": len(latencies), "requests": len(records)},
            "cost": _counter([record.event.cost for record in records], len(records)),
        })
    return {"physical_requests": len(requests), "groups": summaries,
            "cost": _counter([request.event.cost for request in requests], len(requests)),
            "cost_unit": "recorded_units_currency_unspecified",
            "latency_unit": "seconds_of_physical_request_work_not_wall_time",
            "counting": "unique_execution_id_and_call_id; overlapping_condition_views_must_not_be_added",
            "usage_policy": "reported_keys_separately; missing_counters_unknown; no_token_counter_cross_sums"}
