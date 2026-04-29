# Brief: recall-guard

## Problem
Financial quantitative agents suffer from "recall" (look-ahead bias) during backtesting because modern Large Language Models (LLMs) memorize historical financial data (e.g., stock prices, news) from their training sets. This leads to artificially inflated predictive accuracy on historical data that collapses in out-of-sample (future) trading environments.

## Current State
Research has proven that LLMs memorize past financial data, causing severe look-ahead bias in backtests. Traditional attempts to fix this usually rely on altering inputs to break the temporal anchor (e.g., Z-Score Standardization + Temporal Masking, feeding Abstract Factor Spreads instead of price action, or using the BlindTrade Framework with anonymized tickers). However, the core philosophy of MemGuard-Alpha is that altering the input is a dead end because it destroys the very signal needed for accurate forecasting. There is currently no out-of-the-box, lightweight, signal-level filter in DSPy that operates strictly on the output level.

## Desired Outcome
A DSPy-based pipeline that automatically detects and mitigates look-ahead bias in LLM-generated financial forecasts without modifying the input data. The pipeline will successfully reproduce the massive in-sample vs. out-of-sample accuracy gap (the "Raw" baseline) and then mathematically flatten that curve using Membership Inference Attacks (MIA) penalties (the "MemGuard" pass).

## Approach
Implement the pipeline using DSPy and NVIDIA API services (`https://integrate.api.nvidia.com/v1`). Following the MemGuard-Alpha thesis, we feed the model 100% real, unmasked data and catch the "cheating" at the output level.
1. **Dataset**: Treat `Look-Ahead-Bench` as the dataset for DSPy train/dev sets, feeding explicit historical contexts without input abstraction.
2. **"Raw" Baseline Pass**: Run prompts through the model ensemble without filtering to establish baseline look-ahead bias.
3. **"MemGuard" Pass**: Run prompts with `include_logprobs=true` via the NVIDIA API. Extract Loss and Min-K% metrics from the response to calculate a continuous MIA penalty. Discount the model's prediction confidence if a memorization spike is detected.

## Scope
- **In**: DSPy project setup, Look-Ahead-Bench dataset integration, Raw baseline evaluation, MemGuard pass with MIA penalty (Loss, Min-K%), Nvidia API integration.
- **Out**: Live market execution, building custom LLMs from scratch, scraping new ETF data, complex portfolio backtesting engines beyond the MIA signal evaluation.

## Boundary Candidates
- **Dataset Integration**: Loading and structuring Look-Ahead-Bench for DSPy.
- **MIA Scoring Engine**: Extracting logprobs and calculating Loss/Min-K% penalties.
- **DSPy Module**: The core pipeline applying the MIA penalty to the raw model confidence.
- **Evaluation**: Comparing Raw vs. MemGuard performance metrics.

## Out of Boundary
- Live trading infrastructure.
- Alternative bias mitigation strategies (e.g., entity neutering, retraining).

## Upstream / Downstream
- **Upstream**: Look-Ahead-Bench dataset, NVIDIA API.
- **Downstream**: Future quantitative trading strategies and agents that will use this guarded signal.

## Existing Spec Touchpoints
- **Extends**: None (greenfield project).
- **Adjacent**: None.

## Constraints
- Models must be hosted in NVIDIA API services.
- The API must support returning logprobs (e.g., `include_logprobs=true`) to calculate MIA scores.
- Code must use a `.venv` with Python 3.14 (already initialized) and modern DSPy.
