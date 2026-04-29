# Requirements Document

## Introduction
The Recall Guard pipeline is a DSPy-based integration designed to detect and penalize look-ahead bias (memorization) in LLM-generated financial forecasts. Based on the MemGuard-Alpha thesis, this pipeline explicitly rejects input masking (like Z-scores or anonymization) and instead feeds models real, unmasked historical data. It mitigates look-ahead bias at the output layer by retrieving log-probabilities via the NVIDIA API and applying MIA (Membership Inference Attack) penalties (Loss and Min-K%) to the model's confidence scores.

## Requirements

### Requirement 1: Dataset Loading
**Objective:** As a quantitative researcher, I want to load Look-Ahead-Bench data directly into DSPy, so that I can evaluate the models on structured, realistic historical contexts without having to scrape data myself.

#### Acceptance Criteria
1. When the pipeline initializes, the system shall load Look-Ahead-Bench data containing financial contexts and temporal timestamps.
2. The system shall process these contexts as unmasked, real data without stripping entities or dates.
3. The system shall partition the data into DSPy train and dev sets.

### Requirement 2: NVIDIA API Integration
**Objective:** As an evaluator, I want to execute prompts via the NVIDIA API and retrieve token-level log-probabilities, so that I can calculate MIA metrics.

#### Acceptance Criteria
1. When querying the model, the system shall use the NVIDIA API (`https://integrate.api.nvidia.com/v1`).
2. The system shall set `include_logprobs=true` in the API request.
3. The system shall successfully extract and parse the log-probabilities from the NVIDIA API response.

### Requirement 3: Raw Baseline Generation
**Objective:** As a researcher, I want to run a baseline pass without any penalties, so that I can measure the model's unmitigated look-ahead bias on historical data.

#### Acceptance Criteria
1. The system shall run the DSPy pipeline on the dev set without applying MIA filters.
2. The system shall record the model's directional predictions and raw confidence levels.
3. The system shall output the baseline accuracy metrics demonstrating the in-sample vs out-of-sample gap.

### Requirement 4: MIA Scoring Engine
**Objective:** As the system, I want to compute MIA scores (Loss, Min-K%) for a given prediction, so that I can quantify the likelihood that the model memorized the text.

#### Acceptance Criteria
1. The system shall calculate the Average Negative Log-Likelihood (Loss) across the generated tokens.
2. The system shall calculate the Min-K% Prob by averaging the log-probabilities of the K% (e.g., K=20) tokens with the lowest probabilities.
3. The system shall normalize these metrics into a continuous penalty score.

### Requirement 5: MemGuard Penalty Application
**Objective:** As a portfolio manager, I want the system to discount a model's prediction confidence when a memorization spike is detected, so that the signal is mathematically flattened to realistic out-of-sample levels.

#### Acceptance Criteria
1. The system shall apply the computed MIA penalty score directly to the raw confidence output of the model.
2. If the penalty indicates a high probability of memorization, the system shall reduce the confidence of the prediction proportionally.
3. The system shall generate a finalized "MIA-Penalized" signal and calculate its overall accuracy metrics for comparison against the Raw baseline.
