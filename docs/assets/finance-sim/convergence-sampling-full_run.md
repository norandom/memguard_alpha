Sources: macro rows from FRED (Federal Reserve Bank of St. Louis) as known on each decision date, including BAA10Y, Moody's Seasoned Baa Corporate Bond Yield Relative to Yield on 10-Year Treasury Constant Maturity, copyright Moody's, cited and not reproduced; realized next-month returns from Financial Modeling Prep (FMP) end-of-day adjusted closes (licensed, not published) with DTB3 from FRED as the cash leg; EDGAR outlook prompts from SEC EDGAR filings. Replayed offline from the recorded draws and scores by scripts/finance_sim/replay.py (task 8.4) with no model request; the JSON document with the per-item rows stays private under runs/finance-sim/analysis/. Derived statistics only: no reply text, no realized return and no raw licensed series.

# Top-k versus full-distribution convergence

Arm sampling, stage full_run (macro items), model gemma3:4b, weights sha256:f06e91e18fc74295d0c781bf91622508c75655e2c18ed869d52447ffe1e667e1, plan f7603eeb8efdbf74645d658f2bd733a35b3834ba7c90b74d00c6ac2e6b5332fe.

Values compare only within one model and one set of weights: gemma3:4b with weights sha256:f06e91e18fc74295d0c781bf91622508c75655e2c18ed869d52447ffe1e667e1. Values from different builds or model families are not comparable (Requirement 9.4).

Full-distribution features use the kept top-100 alternatives taken from the full next-token vector plus the exact actual-token logprob; Min-K%++ uses moments of the kept alternatives, not of the whole vocabulary. Agreement at k = 100 is identical by construction.

Host top-k rows truncate each draw's full-distribution evidence to its first k alternatives and keep the full-distribution calibrator, so only the evidence changes. The served row uses the generation host's own top-k logprobs through the top-k calibrator, the study's top_k estimate source. Agreement is the mean and maximum absolute difference to the full-distribution value and the share of draws within 1e-06; mass is the probability the first k alternatives cover, per position. Decisions are recomputed from the recorded draws with decide_from_draws; no request is sent.

Items compared: 40 (0 planned items missing); draws compared: 186 (0 draws missing). Kept alternatives: 100 of a 262145-token vocabulary, covering a mean mass of 0.999978 (minimum 0.990052).

## Feature agreement

| Evidence | k | Feature | Draws | Mean abs diff | Max abs diff | Share within 1e-06 | Missing |
|---|---|---|---|---|---|---|---|
| host_top_k | 1 | loss | 186 | 0 | 0 | 1 | none |
| host_top_k | 1 | min_k | 186 | 0 | 0 | 1 | none |
| host_top_k | 1 | min_k_pp | 186 | 2.21323e+06 | 4.22793e+06 | 0 | none |
| host_top_k | 1 | zlib_ratio | 186 | 0 | 0 | 1 | none |
| host_top_k | 1 | estimate | 186 | 0.531211 | 0.999866 | 0 | none |
| host_top_k | 5 | loss | 186 | 0 | 0 | 1 | none |
| host_top_k | 5 | min_k | 186 | 0 | 0 | 1 | none |
| host_top_k | 5 | min_k_pp | 186 | 5.68259 | 20.6742 | 0 | none |
| host_top_k | 5 | zlib_ratio | 186 | 0 | 0 | 1 | none |
| host_top_k | 5 | estimate | 186 | 0.106243 | 0.780089 | 0 | none |
| host_top_k | 10 | loss | 186 | 0 | 0 | 1 | none |
| host_top_k | 10 | min_k | 186 | 0 | 0 | 1 | none |
| host_top_k | 10 | min_k_pp | 186 | 3.78192 | 11.2322 | 0 | none |
| host_top_k | 10 | zlib_ratio | 186 | 0 | 0 | 1 | none |
| host_top_k | 10 | estimate | 186 | 0.0875078 | 0.647406 | 0 | none |
| host_top_k | 20 | loss | 186 | 0 | 0 | 1 | none |
| host_top_k | 20 | min_k | 186 | 0 | 0 | 1 | none |
| host_top_k | 20 | min_k_pp | 186 | 2.08387 | 2.27733 | 0 | none |
| host_top_k | 20 | zlib_ratio | 186 | 0 | 0 | 1 | none |
| host_top_k | 20 | estimate | 186 | 0.0533318 | 0.18841 | 0 | none |
| host_top_k | 50 | loss | 186 | 0 | 0 | 1 | none |
| host_top_k | 50 | min_k | 186 | 0 | 0 | 1 | none |
| host_top_k | 50 | min_k_pp | 186 | 1.01023 | 1.11366 | 0 | none |
| host_top_k | 50 | zlib_ratio | 186 | 0 | 0 | 1 | none |
| host_top_k | 50 | estimate | 186 | 0.0255269 | 0.0922372 | 0 | none |
| host_top_k | 100 | loss | 186 | 0 | 0 | 1 | none |
| host_top_k | 100 | min_k | 186 | 0 | 0 | 1 | none |
| host_top_k | 100 | min_k_pp | 186 | 0 | 0 | 1 | none |
| host_top_k | 100 | zlib_ratio | 186 | 0 | 0 | 1 | none |
| host_top_k | 100 | estimate | 186 | 0 | 0 | 1 | none |
| served_top_k | 5 | loss | 186 | 0.463156 | 0.791405 | 0 | none |
| served_top_k | 5 | min_k | 186 | 2.4138 | 4.15131 | 0 | none |
| served_top_k | 5 | min_k_pp | 186 | 2.00432 | 2.57447 | 0 | none |
| served_top_k | 5 | zlib_ratio | 186 | 0.301874 | 0.519031 | 0 | none |
| served_top_k | 5 | estimate | 186 | 0.361409 | 0.826424 | 0 | none |

## Mass coverage

| Evidence | k | Draws | Mean mass | Min mass | Missing |
|---|---|---|---|---|---|
| host_top_k | 1 | 186 | 0.946211 | 0.122532 | none |
| host_top_k | 5 | 186 | 0.991803 | 0.505019 | none |
| host_top_k | 10 | 186 | 0.999161 | 0.89529 | none |
| host_top_k | 20 | 186 | 0.999922 | 0.981819 | none |
| host_top_k | 50 | 186 | 0.999965 | 0.986934 | none |
| host_top_k | 100 | 186 | 0.999978 | 0.990052 | none |
| served_top_k | 5 | 186 | 0.999255 | 0.697739 | none |

## Adjusted decision change

| Evidence | k | Mode | Items | Mean abs change p_memorized | Max abs change p_memorized | Mean abs change adjusted summary | Max abs change adjusted summary | Share category changed | Share status changed | Missing |
|---|---|---|---|---|---|---|---|---|---|---|
| host_top_k | 1 | ensemble | 20 | 0.51738 | 0.984643 | 1.44668 | 4.29304 | 0.875 | 0 | no decision: fewer parsed draws than the frozen minimum (the stored result gives the counts): 20 |
| host_top_k | 1 | single | 21 | 0.544485 | 0.996919 | 1.82002 | 5.37309 | 0.761905 | 0 | no decision: fewer parsed draws than the frozen minimum (the stored result gives the counts): 19 |
| host_top_k | 5 | ensemble | 20 | 0.119611 | 0.583467 | 0.324323 | 1.23922 | 0 | 0 | no decision: fewer parsed draws than the frozen minimum (the stored result gives the counts): 20 |
| host_top_k | 5 | single | 21 | 0.0987922 | 0.776235 | 0.333827 | 2.96289 | 0 | 0 | no decision: fewer parsed draws than the frozen minimum (the stored result gives the counts): 19 |
| host_top_k | 10 | ensemble | 20 | 0.0984664 | 0.475201 | 0.272747 | 1.06849 | 0 | 0 | no decision: fewer parsed draws than the frozen minimum (the stored result gives the counts): 20 |
| host_top_k | 10 | single | 21 | 0.0803751 | 0.647406 | 0.275821 | 2.47115 | 0 | 0 | no decision: fewer parsed draws than the frozen minimum (the stored result gives the counts): 19 |
| host_top_k | 20 | ensemble | 20 | 0.0591341 | 0.171706 | 0.19473 | 0.798646 | 0 | 0 | no decision: fewer parsed draws than the frozen minimum (the stored result gives the counts): 20 |
| host_top_k | 20 | single | 21 | 0.0441803 | 0.176601 | 0.143716 | 0.760269 | 0 | 0 | no decision: fewer parsed draws than the frozen minimum (the stored result gives the counts): 19 |
| host_top_k | 50 | ensemble | 20 | 0.0282179 | 0.0835791 | 0.0940803 | 0.379359 | 0 | 0 | no decision: fewer parsed draws than the frozen minimum (the stored result gives the counts): 20 |
| host_top_k | 50 | single | 21 | 0.0205062 | 0.0855532 | 0.0674178 | 0.368307 | 0 | 0 | no decision: fewer parsed draws than the frozen minimum (the stored result gives the counts): 19 |
| host_top_k | 100 | ensemble | 20 | 0 | 0 | 0 | 0 | 0 | 0 | no decision: fewer parsed draws than the frozen minimum (the stored result gives the counts): 20 |
| host_top_k | 100 | single | 21 | 0 | 0 | 0 | 0 | 0 | 0 | no decision: fewer parsed draws than the frozen minimum (the stored result gives the counts): 19 |
| served_top_k | 5 | ensemble | 20 | 0.301957 | 0.536157 | 0.773284 | 2.33536 | 0 | 0 | no decision: fewer parsed draws than the frozen minimum (the stored result gives the counts): 20 |
| served_top_k | 5 | single | 21 | 0.388632 | 0.690601 | 1.02966 | 2.7551 | 0 | 0 | no decision: fewer parsed draws than the frozen minimum (the stored result gives the counts): 19 |
