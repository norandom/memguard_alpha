Sources: macro rows from FRED (Federal Reserve Bank of St. Louis) as known on each decision date, including BAA10Y, Moody's Seasoned Baa Corporate Bond Yield Relative to Yield on 10-Year Treasury Constant Maturity, copyright Moody's, cited and not reproduced; realized next-month returns from Financial Modeling Prep (FMP) end-of-day adjusted closes (licensed, not published) with DTB3 from FRED as the cash leg; EDGAR outlook prompts from SEC EDGAR filings. Replayed offline from the recorded draws and scores by scripts/finance_sim/replay.py (task 8.4) with no model request; the JSON document with the per-item rows stays private under runs/finance-sim/analysis/. Derived statistics only: no reply text, no realized return and no raw licensed series.

# Top-k versus full-distribution convergence

Arm sampling, stage negative_control (edgar items), model gemma3:4b, weights sha256:f06e91e18fc74295d0c781bf91622508c75655e2c18ed869d52447ffe1e667e1, plan f7603eeb8efdbf74645d658f2bd733a35b3834ba7c90b74d00c6ac2e6b5332fe.

Values compare only within one model and one set of weights: gemma3:4b with weights sha256:f06e91e18fc74295d0c781bf91622508c75655e2c18ed869d52447ffe1e667e1. Values from different builds or model families are not comparable (Requirement 9.4).

Full-distribution features use the kept top-100 alternatives taken from the full next-token vector plus the exact actual-token logprob; Min-K%++ uses moments of the kept alternatives, not of the whole vocabulary. Agreement at k = 100 is identical by construction.

Host top-k rows truncate each draw's full-distribution evidence to its first k alternatives and keep the full-distribution calibrator, so only the evidence changes. The served row uses the generation host's own top-k logprobs through the top-k calibrator, the study's top_k estimate source. Agreement is the mean and maximum absolute difference to the full-distribution value and the share of draws within 1e-06; mass is the probability the first k alternatives cover, per position. Decisions are recomputed from the recorded draws with decide_from_draws; no request is sent.

Items compared: 40 (0 planned items missing); draws compared: 320 (0 draws missing). Kept alternatives: 100 of a 262145-token vocabulary, covering a mean mass of 0.999989 (minimum 0.999241).

## Feature agreement

| Evidence | k | Feature | Draws | Mean abs diff | Max abs diff | Share within 1e-06 | Missing |
|---|---|---|---|---|---|---|---|
| host_top_k | 1 | loss | 320 | 0 | 0 | 1 | none |
| host_top_k | 1 | min_k | 320 | 0 | 0 | 1 | none |
| host_top_k | 1 | min_k_pp | 320 | 8.29956e+06 | 1.18857e+07 | 0 | none |
| host_top_k | 1 | zlib_ratio | 320 | 0 | 0 | 1 | none |
| host_top_k | 1 | estimate | 320 | 0.499091 | 0.606979 | 0 | none |
| host_top_k | 5 | loss | 320 | 0 | 0 | 1 | none |
| host_top_k | 5 | min_k | 320 | 0 | 0 | 1 | none |
| host_top_k | 5 | min_k_pp | 320 | 2.24208 | 2.90565 | 0 | none |
| host_top_k | 5 | zlib_ratio | 320 | 0 | 0 | 1 | none |
| host_top_k | 5 | estimate | 320 | 0.262063 | 0.330463 | 0 | none |
| host_top_k | 10 | loss | 320 | 0 | 0 | 1 | none |
| host_top_k | 10 | min_k | 320 | 0 | 0 | 1 | none |
| host_top_k | 10 | min_k_pp | 320 | 1.89843 | 2.19774 | 0 | none |
| host_top_k | 10 | zlib_ratio | 320 | 0 | 0 | 1 | none |
| host_top_k | 10 | estimate | 320 | 0.228678 | 0.273637 | 0 | none |
| host_top_k | 20 | loss | 320 | 0 | 0 | 1 | none |
| host_top_k | 20 | min_k | 320 | 0 | 0 | 1 | none |
| host_top_k | 20 | min_k_pp | 320 | 1.72182 | 1.95992 | 0 | none |
| host_top_k | 20 | zlib_ratio | 320 | 0 | 0 | 1 | none |
| host_top_k | 20 | estimate | 320 | 0.21057 | 0.245349 | 0 | none |
| host_top_k | 50 | loss | 320 | 0 | 0 | 1 | none |
| host_top_k | 50 | min_k | 320 | 0 | 0 | 1 | none |
| host_top_k | 50 | min_k_pp | 320 | 0.950581 | 1.09784 | 0 | none |
| host_top_k | 50 | zlib_ratio | 320 | 0 | 0 | 1 | none |
| host_top_k | 50 | estimate | 320 | 0.121236 | 0.143441 | 0 | none |
| host_top_k | 100 | loss | 320 | 0 | 0 | 1 | none |
| host_top_k | 100 | min_k | 320 | 0 | 0 | 1 | none |
| host_top_k | 100 | min_k_pp | 320 | 0 | 0 | 1 | none |
| host_top_k | 100 | zlib_ratio | 320 | 0 | 0 | 1 | none |
| host_top_k | 100 | estimate | 320 | 0 | 0 | 1 | none |
| served_top_k | 5 | loss | 320 | 1.27952 | 2.37404 | 0 | none |
| served_top_k | 5 | min_k | 320 | 8.30826 | 11.981 | 0 | none |
| served_top_k | 5 | min_k_pp | 320 | 0.328392 | 1.45064 | 0 | none |
| served_top_k | 5 | zlib_ratio | 320 | 0.46588 | 0.83584 | 0 | none |
| served_top_k | 5 | estimate | 320 | 0.0631755 | 0.423397 | 0 | none |

## Mass coverage

| Evidence | k | Draws | Mean mass | Min mass | Missing |
|---|---|---|---|---|---|
| host_top_k | 1 | 320 | 0.977514 | 0.395332 | none |
| host_top_k | 5 | 320 | 0.999416 | 0.932908 | none |
| host_top_k | 10 | 320 | 0.999873 | 0.994095 | none |
| host_top_k | 20 | 320 | 0.999934 | 0.996568 | none |
| host_top_k | 50 | 320 | 0.999977 | 0.998581 | none |
| host_top_k | 100 | 320 | 0.999989 | 0.999241 | none |
| served_top_k | 5 | 320 | 0.99999 | 0.995542 | none |

## Adjusted decision change

| Evidence | k | Mode | Items | Mean abs change p_memorized | Max abs change p_memorized | Mean abs change adjusted summary | Max abs change adjusted summary | Share category changed | Share status changed | Missing |
|---|---|---|---|---|---|---|---|---|---|---|
| host_top_k | 1 | ensemble | 40 | 0.499091 | 0.606979 | 0 | 0 | 0 | 0 | none |
| host_top_k | 1 | single | 40 | 0.498405 | 0.606979 | 0 | 0 | 0 | 0 | none |
| host_top_k | 5 | ensemble | 40 | 0.262063 | 0.330463 | 0 | 0 | 0 | 0 | none |
| host_top_k | 5 | single | 40 | 0.261904 | 0.330463 | 0 | 0 | 0 | 0 | none |
| host_top_k | 10 | ensemble | 40 | 0.228678 | 0.273637 | 0 | 0 | 0 | 0 | none |
| host_top_k | 10 | single | 40 | 0.228422 | 0.273637 | 0 | 0 | 0 | 0 | none |
| host_top_k | 20 | ensemble | 40 | 0.21057 | 0.245349 | 0 | 0 | 0 | 0 | none |
| host_top_k | 20 | single | 40 | 0.210403 | 0.245349 | 0 | 0 | 0 | 0 | none |
| host_top_k | 50 | ensemble | 40 | 0.121236 | 0.143441 | 0 | 0 | 0 | 0 | none |
| host_top_k | 50 | single | 40 | 0.121143 | 0.143441 | 0 | 0 | 0 | 0 | none |
| host_top_k | 100 | ensemble | 40 | 0 | 0 | 0 | 0 | 0 | 0 | none |
| host_top_k | 100 | single | 40 | 0 | 0 | 0 | 0 | 0 | 0 | none |
| served_top_k | 5 | ensemble | 40 | 0.0576569 | 0.225028 | 0 | 0 | 0 | 0 | none |
| served_top_k | 5 | single | 40 | 0.0576403 | 0.246504 | 0 | 0 | 0 | 0 | none |
