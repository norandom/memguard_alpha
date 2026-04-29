# Design Document: Recall Guard

## Overview
This feature delivers an automated look-ahead bias mitigation pipeline in DSPy for quantitative agents. It executes queries through the NVIDIA API, analyzes the resulting logprobs, and calculates MIA penalties.
**Users**: Quantitative researchers and developers building LLM financial agents.
**Impact**: Eliminates artificially inflated backtest performance by applying the MemGuard-Alpha output-level filter, without destroying signal through input masking.

### Goals
- Process Look-Ahead-Bench data as unmasked inputs.
- Call the NVIDIA API with logprobs enabled.
- Calculate MIA metrics (Loss, Min-K%).
- Discount model confidence dynamically based on the MIA penalty.

### Non-Goals
- Live execution or trading.
- Retraining foundation models.

## Boundary Commitments
### This Spec Owns
- The DSPy signature and module definitions for financial prediction.
- The MIA scoring algorithms (Loss, Min-K%) computed from logprobs.
- The dataset loader for Look-Ahead-Bench.
- The evaluation loop comparing Raw vs. Penalized performance.

### Out of Boundary
- Generating the original Look-Ahead-Bench dataset.
- Training shadow models for other MIA methods.

## Architecture
**Architecture Integration**:
- Selected pattern: DSPy Pipeline with an Interceptor/Scorer pattern for MIA.
- The DSPy module will handle the structured prediction.
- An independent scoring module will take the raw API response containing logprobs and apply the MIA penalty to the final confidence.

## File Structure Plan
### Directory Structure
```text
src/
├── dataset/
│   └── lookahead_loader.py    # Loads Look-Ahead-Bench
├── models/
│   └── nvidia_lm.py           # Custom DSPy LM for NVIDIA API with logprobs
├── pipeline/
│   ├── signature.py           # DSPy Signatures
│   ├── predict_module.py      # Core DSPy Module
│   └── mia_scorer.py          # Calculates Loss, Min-K%, and applies penalty
└── evaluate/
    └── metrics.py             # Evaluation for Raw vs MemGuard
```

## Components and Interfaces

### models/nvidia_lm.py
| Field | Detail |
|-------|--------|
| Intent | Interfaces with the NVIDIA API to request and extract logprobs |
| Requirements | 2 |

**Responsibilities**
- Implement the `dspy.LM` base class.
- Support `include_logprobs=true` and extracting them into the output trace.

### pipeline/mia_scorer.py
| Field | Detail |
|-------|--------|
| Intent | Computes the MIA penalty |
| Requirements | 4, 5 |

**Responsibilities**
- Takes an array of log-probabilities.
- Computes `Loss` and `Min_K_Percent`.
- Applies the penalty function `f(confidence, mia_score) -> penalized_confidence`.

### pipeline/predict_module.py
| Field | Detail |
|-------|--------|
| Intent | Main DSPy module tying the LLM and the Scorer together |
| Requirements | 3, 5 |
