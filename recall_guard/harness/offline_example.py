"""An invented offline setup for the front, so that its use can be shown without a model.

:func:`ask` needs four things that a person without a running model does not
have: a catalogue entry whose modes are qualified, a recall calibrator per
mode, replies with their token evidence, and a host for the logits mode.
:func:`offline_example` returns all four, built in process.

Everything here is invented. The entry :data:`EXAMPLE_ENTRY` names no model,
its quality record is written by hand so that the qualification rule passes,
the calibrators are fitted on sixteen made-up prompts answered by a stand-in
host, and the replies and their evidence are fixed values. The numbers a run
returns show the shape of a result and say nothing about any model or route.

On a real route none of this is needed: the entry comes from the shipped
catalogue, the client and the logits host are built from the entry's provider
profile, and the calibrator is fitted on that model with
:class:`~recall_guard.harness.scorer.MemoryGuardedScorer`.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from recall_guard.core.catalogue import CATALOGUE_SCHEMA, Catalogue, Mode, load_catalogue
from recall_guard.core.completion import (
    CapabilityEvidence,
    CapabilityReport,
    CapabilityStatus,
    FeatureChoices,
    ModelConfig,
    ModelResponse,
    TokenAlternative,
    TokenEvidence,
)
from recall_guard.core.full_distribution import FullDistributionEvidence
from recall_guard.harness.front import WORKED_EXAMPLE_REPLY, FixedEvidence, FixedReplies, fixed_reply
from recall_guard.harness.guarded_decision import SourceCalibrator
from recall_guard.harness.scorer import MemoryGuardedScorer
from recall_guard.mia.estimate_source import EstimateSource

#: The invented catalogue entry of the offline example. It names no model.
EXAMPLE_ENTRY = "example-model"
#: The calibrator fingerprint the invented entry records for both modes.
EXAMPLE_FINGERPRINT = "sha256:offline-example"
#: Alternatives per output token in the fixed evidence of each mode.
EXAMPLE_K: Mapping[Mode, int] = MappingProxyType({Mode.LOGPROBS: 3, Mode.LOGITS: 2})
#: Replies per answer: the ensemble the example asks for, all of them needed.
EXAMPLE_DRAWS = 8

_SETTINGS = {"max_tokens": 512, "temperature": 0.0, "logprobs": True, "top_logprobs": EXAMPLE_K[Mode.LOGPROBS]}
#: Five output tokens with three alternatives each: what the logprobs mode reads from a reply.
_TOKENS = [("x", [("x", -0.2), ("y", -1.2), ("z", -2.1)])] * 5

_CALIBRATION = {
    "boundary": {
        "domain": "financial_time_series", "exposed": "months inside the training range",
        "fenced": "months after the training cutoff", "date_range": "invented for the offline example",
    },
    "quality": {
        "holdout_auc": 0.7, "interval_low": 0.63, "interval_high": 0.77,
        "interval_method": "prompt_clustered_bootstrap", "confidence": 0.95, "n_exposed": 120, "n_fenced": 110,
        "n_prompts": 40, "permutation_p": 0.001, "n_valid_exposed": 480, "n_valid_fenced": 440,
    },
    "cutoff": {
        "inside_mean": 0.6, "after_mean": 0.4, "difference": 0.2, "interval_low": 0.1, "interval_high": 0.3,
        "p_value": 0.004, "n_inside": 120, "n_after": 110,
    },
    "calibrator_fingerprint": EXAMPLE_FINGERPRINT,
    # The catalogue requires a register id of the form R<digits>. R0 is no report of the research register.
    "evidence": {"register_ids": ["R0"], "method": "reply_features"},
}


def _mode(requires: str | None) -> dict[str, Any]:
    return {"available": True, "unavailable_reason": None, "requires": requires,
            "features": ["entropy", "top1_probability"], "features_unavailable": [],
            "calibrations": [dict(_CALIBRATION)]}


_CATALOGUE = {
    "schema": CATALOGUE_SCHEMA,
    "rule": {"min_auc": 0.6, "min_valid": 50, "alpha": 0.05},
    "entries": [{
        "entry_id": EXAMPLE_ENTRY, "profile_id": None, "model": "invented example model",
        "route": "in process, no model", "weights_identity": None, "weights_group": None,
        "modes": {
            Mode.LOGPROBS.value: _mode(None),
            Mode.LOGITS.value: _mode("a host that returns the next-token vector (fixed values here)"),
        },
    }],
}


class _StandInHost:
    """Answers the calibration prompts in process with evidence that separates the two invented corpora."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    def generate(self, prompt: str, **parameters: Any) -> ModelResponse:
        base = -0.2 if prompt.startswith("seen") else -5.0 if prompt.startswith("unseen") else -1.0
        jitter = (int(hashlib.sha256(prompt.encode()).hexdigest(), 16) % 1000) / 20000
        evidence = tuple(TokenEvidence("x", base + jitter + index * 0.01, (
            TokenAlternative("x", base + jitter), TokenAlternative("y", base - 1), TokenAlternative("z", base + 0.1),
        )) for index in range(5))
        return ModelResponse("Direction: 1\nConfidence: 0.8", evidence,
                             requested_settings=dict(self.config.parameters) | parameters)

    def probe(self) -> CapabilityReport:
        self.generate("probe")
        features = FeatureChoices(mia_scoring=True)
        required = self.config.inspect(features).capabilities
        return CapabilityReport(
            {name: CapabilityEvidence(CapabilityStatus.SUPPORTED, "in-process generation", observed=True)
             for name in required}, features)


def _logits_evidence(base: float) -> FullDistributionEvidence:
    """What a logits host returns for one reply: three positions, two kept alternatives each."""
    positions = range(3)
    return FullDistributionEvidence(
        model_digest="sha256:" + "0" * 64, n_vocab=50, response_token_ids=tuple(positions),
        logprob=tuple(base - 0.01 * index for index in positions), entropy=(1.0,) * 3, rank=(1,) * 3,
        top_logprobs=tuple(((index, base - 0.01 * index), (40, base - 1.0)) for index in positions),
    )


@dataclass(frozen=True)
class OfflineExample:
    """What :func:`offline_example` returns: an invented entry and what :func:`ask` needs to run it offline.

    ``entry`` is the entry id, ``catalogue`` the one-entry catalogue that holds
    it, and ``calibrators`` the recall calibrator of each mode. :meth:`arguments`
    gives the keyword arguments of one :func:`ask` call for a mode.
    """

    entry: str
    catalogue: Catalogue
    calibrators: Mapping[Mode, SourceCalibrator]
    config: ModelConfig

    def arguments(self, mode: Mode | str) -> dict[str, Any]:
        """The keyword arguments of :func:`ask` for ``mode``: a fresh fixed client and host, and the rest.

        On a real route only ``entry``, ``mode`` and ``calibrator`` are passed;
        the client and the logits host come from the entry's provider profile.
        """
        mode = Mode(mode)
        replies = [fixed_reply(WORKED_EXAMPLE_REPLY, _TOKENS)] * EXAMPLE_DRAWS
        host = FixedEvidence([_logits_evidence(-2.0 - 0.1 * index) for index in range(EXAMPLE_DRAWS)])
        return {
            "entry": self.entry, "mode": mode, "catalogue": self.catalogue, "calibrator": dict(self.calibrators),
            "client": FixedReplies(replies, config=self.config), "full_distribution": host,
            "k": EXAMPLE_K[mode], "draws": EXAMPLE_DRAWS, "min_parsed": EXAMPLE_DRAWS,
        }


def offline_example() -> OfflineExample:
    """The invented offline setup: one entry qualified in both modes, with a calibrator for each.

    Opens no connection and reads no file. The calibrators are fitted on
    sixteen made-up prompts answered by a stand-in host; their fingerprint is
    set to the one the invented entry records, which is what a recorded
    calibrator of a real entry has. See the module docstring: every number is
    invented.
    """
    config = ModelConfig(protocol="openai_chat", endpoint="https://example.invalid/chat", model="example",
                         parameters=_SETTINGS)
    scorer = MemoryGuardedScorer.calibrate_client(
        client=_StandInHost(config),
        is_memorized=[f"seen-{index}" for index in range(8)],
        oos_control=[f"unseen-{index}" for index in range(8)],
        min_valid=4, max_workers=1,
    )
    sources = {Mode.LOGPROBS: EstimateSource.TOP_K, Mode.LOGITS: EstimateSource.FULL_DISTRIBUTION}
    calibrators = {
        mode: dataclasses.replace(SourceCalibrator.from_scorer(scorer, source=source, corpus_sha256=None),
                                  fingerprint=EXAMPLE_FINGERPRINT)
        for mode, source in sources.items()
    }
    return OfflineExample(entry=EXAMPLE_ENTRY, catalogue=load_catalogue(_CATALOGUE),
                          calibrators=MappingProxyType(calibrators), config=config)


__all__ = ["EXAMPLE_DRAWS", "EXAMPLE_ENTRY", "EXAMPLE_FINGERPRINT", "EXAMPLE_K", "OfflineExample", "offline_example"]
