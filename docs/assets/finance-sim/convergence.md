Figure F4. Top-k versus full-distribution agreement as k grows. Data: the stored convergence tables (task 7.12; read, not recomputed) of the sampling arm (probe role, a small paired check), model gemma3:4b, plan f7603eeb8efd; per feature and k, the mean and the maximum over the compared draws of the absolute difference between the value from each draw's full-distribution evidence truncated to its first k alternatives and the value from the full distribution, and the probability mass the first k alternatives cover (mean and minimum over draws). Estimator: Host top-k rows truncate each draw's full-distribution evidence to its first k alternatives and keep the full-distribution calibrator, so only the evidence changes. The served row uses the generation host's own top-k logprobs through the top-k calibrator, the study's top_k estimate source. Agreement is the mean and maximum absolute difference to the full-distribution value and the share of draws within 1e-06; mass is the probability the first k alternatives cover, per position. Decisions are recomputed from the recorded draws with decide_from_draws; no request is sent. Scope: Full-distribution features use the kept top-100 alternatives taken from the full next-token vector plus the exact actual-token logprob; Min-K%++ uses moments of the kept alternatives, not of the whole vocabulary. Agreement at k = 100 is identical by construction. The served row (the generation host's own top-5 logprobs through the top-k calibrator, a different calibrator) is a separate open marker at k = 5, not joined to the lines. Sample size: full run: 186 draws of 40 items (0 draws and 0 planned items missing), 100 kept alternatives of a 262145-token vocabulary covering a mean mass of 0.999978 (minimum 0.990052); negative control: 320 draws of 40 items (0 draws and 0 planned items missing), 100 kept alternatives of a 262145-token vocabulary covering a mean mass of 0.999989 (minimum 0.999241). Interval: none; the tables report the mean and the maximum absolute difference over draws and the share of draws within 1e-06 (in the table), and no interval. Values compare only within one model and one set of weights: gemma3:4b with weights sha256:f06e91e18fc74295d0c781bf91622508c75655e2c18ed869d52447ffe1e667e1. Values from different builds or model families are not comparable (Requirement 9.4). Encoding: one colour, marker shape and direct label per line (stage and statistic, palette slots 1 to 4 in order): macro is the full run (the macro prompts of the positive control), EDGAR the negative control (the EDGAR releases); solid lines are means, dashed lines the maximum (max, agreement) or the minimum (min, mass); identical lines share one label; open markers are the served row; the muted line marks no difference (agreement) or the whole vocabulary (mass); the x axis is logarithmic. Sources: macro rows from FRED (Federal Reserve Bank of St. Louis) as known on each decision date, including BAA10Y, Moody's Seasoned Baa Corporate Bond Yield Relative to Yield on 10-Year Treasury Constant Maturity, copyright Moody's, cited and not reproduced; EDGAR outlook prompts from SEC EDGAR filings; the study's outcomes come from Financial Modeling Prep (FMP) end-of-day adjusted closes (licensed, not published), and no FMP value enters this figure. Derived statistics only: no reply text, no realized returns and no raw licensed series.

| arm | role | stage | kind | quantity | evidence | k | n | mean_abs_diff | max_abs_diff | share_within | mean_mass | min_mass | drawn | missing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sampling | probe | full_run | macro | loss | host_top_k | 1 | 186 | 0 | 0 | 1 |  |  | yes |  |
| sampling | probe | full_run | macro | min_k | host_top_k | 1 | 186 | 0 | 0 | 1 |  |  | yes |  |
| sampling | probe | full_run | macro | min_k_pp | host_top_k | 1 | 186 | 2.21323e+06 | 4.22793e+06 | 0 |  |  | yes |  |
| sampling | probe | full_run | macro | zlib_ratio | host_top_k | 1 | 186 | 0 | 0 | 1 |  |  | yes |  |
| sampling | probe | full_run | macro | estimate | host_top_k | 1 | 186 | 0.531211 | 0.999866 | 0 |  |  | yes |  |
| sampling | probe | full_run | macro | loss | host_top_k | 5 | 186 | 0 | 0 | 1 |  |  | yes |  |
| sampling | probe | full_run | macro | min_k | host_top_k | 5 | 186 | 0 | 0 | 1 |  |  | yes |  |
| sampling | probe | full_run | macro | min_k_pp | host_top_k | 5 | 186 | 5.68259 | 20.6742 | 0 |  |  | yes |  |
| sampling | probe | full_run | macro | zlib_ratio | host_top_k | 5 | 186 | 0 | 0 | 1 |  |  | yes |  |
| sampling | probe | full_run | macro | estimate | host_top_k | 5 | 186 | 0.106243 | 0.780089 | 0 |  |  | yes |  |
| sampling | probe | full_run | macro | loss | host_top_k | 10 | 186 | 0 | 0 | 1 |  |  | yes |  |
| sampling | probe | full_run | macro | min_k | host_top_k | 10 | 186 | 0 | 0 | 1 |  |  | yes |  |
| sampling | probe | full_run | macro | min_k_pp | host_top_k | 10 | 186 | 3.78192 | 11.2322 | 0 |  |  | yes |  |
| sampling | probe | full_run | macro | zlib_ratio | host_top_k | 10 | 186 | 0 | 0 | 1 |  |  | yes |  |
| sampling | probe | full_run | macro | estimate | host_top_k | 10 | 186 | 0.0875078 | 0.647406 | 0 |  |  | yes |  |
| sampling | probe | full_run | macro | loss | host_top_k | 20 | 186 | 0 | 0 | 1 |  |  | yes |  |
| sampling | probe | full_run | macro | min_k | host_top_k | 20 | 186 | 0 | 0 | 1 |  |  | yes |  |
| sampling | probe | full_run | macro | min_k_pp | host_top_k | 20 | 186 | 2.08387 | 2.27733 | 0 |  |  | yes |  |
| sampling | probe | full_run | macro | zlib_ratio | host_top_k | 20 | 186 | 0 | 0 | 1 |  |  | yes |  |
| sampling | probe | full_run | macro | estimate | host_top_k | 20 | 186 | 0.0533318 | 0.18841 | 0 |  |  | yes |  |
| sampling | probe | full_run | macro | loss | host_top_k | 50 | 186 | 0 | 0 | 1 |  |  | yes |  |
| sampling | probe | full_run | macro | min_k | host_top_k | 50 | 186 | 0 | 0 | 1 |  |  | yes |  |
| sampling | probe | full_run | macro | min_k_pp | host_top_k | 50 | 186 | 1.01023 | 1.11366 | 0 |  |  | yes |  |
| sampling | probe | full_run | macro | zlib_ratio | host_top_k | 50 | 186 | 0 | 0 | 1 |  |  | yes |  |
| sampling | probe | full_run | macro | estimate | host_top_k | 50 | 186 | 0.0255269 | 0.0922372 | 0 |  |  | yes |  |
| sampling | probe | full_run | macro | loss | host_top_k | 100 | 186 | 0 | 0 | 1 |  |  | yes |  |
| sampling | probe | full_run | macro | min_k | host_top_k | 100 | 186 | 0 | 0 | 1 |  |  | yes |  |
| sampling | probe | full_run | macro | min_k_pp | host_top_k | 100 | 186 | 0 | 0 | 1 |  |  | yes |  |
| sampling | probe | full_run | macro | zlib_ratio | host_top_k | 100 | 186 | 0 | 0 | 1 |  |  | yes |  |
| sampling | probe | full_run | macro | estimate | host_top_k | 100 | 186 | 0 | 0 | 1 |  |  | yes |  |
| sampling | probe | full_run | macro | loss | served_top_k | 5 | 186 | 0.463156 | 0.791405 | 0 |  |  | yes |  |
| sampling | probe | full_run | macro | min_k | served_top_k | 5 | 186 | 2.4138 | 4.15131 | 0 |  |  | yes |  |
| sampling | probe | full_run | macro | min_k_pp | served_top_k | 5 | 186 | 2.00432 | 2.57447 | 0 |  |  | yes |  |
| sampling | probe | full_run | macro | zlib_ratio | served_top_k | 5 | 186 | 0.301874 | 0.519031 | 0 |  |  | yes |  |
| sampling | probe | full_run | macro | estimate | served_top_k | 5 | 186 | 0.361409 | 0.826424 | 0 |  |  | yes |  |
| sampling | probe | full_run | macro | mass | host_top_k | 1 | 186 |  |  |  | 0.946211 | 0.122532 | yes |  |
| sampling | probe | full_run | macro | mass | host_top_k | 5 | 186 |  |  |  | 0.991803 | 0.505019 | yes |  |
| sampling | probe | full_run | macro | mass | host_top_k | 10 | 186 |  |  |  | 0.999161 | 0.89529 | yes |  |
| sampling | probe | full_run | macro | mass | host_top_k | 20 | 186 |  |  |  | 0.999922 | 0.981819 | yes |  |
| sampling | probe | full_run | macro | mass | host_top_k | 50 | 186 |  |  |  | 0.999965 | 0.986934 | yes |  |
| sampling | probe | full_run | macro | mass | host_top_k | 100 | 186 |  |  |  | 0.999978 | 0.990052 | yes |  |
| sampling | probe | full_run | macro | mass | served_top_k | 5 | 186 |  |  |  | 0.999255 | 0.697739 | yes |  |
| sampling | probe | negative_control | edgar | loss | host_top_k | 1 | 320 | 0 | 0 | 1 |  |  | yes |  |
| sampling | probe | negative_control | edgar | min_k | host_top_k | 1 | 320 | 0 | 0 | 1 |  |  | yes |  |
| sampling | probe | negative_control | edgar | min_k_pp | host_top_k | 1 | 320 | 8.29956e+06 | 1.18857e+07 | 0 |  |  | yes |  |
| sampling | probe | negative_control | edgar | zlib_ratio | host_top_k | 1 | 320 | 0 | 0 | 1 |  |  | yes |  |
| sampling | probe | negative_control | edgar | estimate | host_top_k | 1 | 320 | 0.499091 | 0.606979 | 0 |  |  | yes |  |
| sampling | probe | negative_control | edgar | loss | host_top_k | 5 | 320 | 0 | 0 | 1 |  |  | yes |  |
| sampling | probe | negative_control | edgar | min_k | host_top_k | 5 | 320 | 0 | 0 | 1 |  |  | yes |  |
| sampling | probe | negative_control | edgar | min_k_pp | host_top_k | 5 | 320 | 2.24208 | 2.90565 | 0 |  |  | yes |  |
| sampling | probe | negative_control | edgar | zlib_ratio | host_top_k | 5 | 320 | 0 | 0 | 1 |  |  | yes |  |
| sampling | probe | negative_control | edgar | estimate | host_top_k | 5 | 320 | 0.262063 | 0.330463 | 0 |  |  | yes |  |
| sampling | probe | negative_control | edgar | loss | host_top_k | 10 | 320 | 0 | 0 | 1 |  |  | yes |  |
| sampling | probe | negative_control | edgar | min_k | host_top_k | 10 | 320 | 0 | 0 | 1 |  |  | yes |  |
| sampling | probe | negative_control | edgar | min_k_pp | host_top_k | 10 | 320 | 1.89843 | 2.19774 | 0 |  |  | yes |  |
| sampling | probe | negative_control | edgar | zlib_ratio | host_top_k | 10 | 320 | 0 | 0 | 1 |  |  | yes |  |
| sampling | probe | negative_control | edgar | estimate | host_top_k | 10 | 320 | 0.228678 | 0.273637 | 0 |  |  | yes |  |
| sampling | probe | negative_control | edgar | loss | host_top_k | 20 | 320 | 0 | 0 | 1 |  |  | yes |  |
| sampling | probe | negative_control | edgar | min_k | host_top_k | 20 | 320 | 0 | 0 | 1 |  |  | yes |  |
| sampling | probe | negative_control | edgar | min_k_pp | host_top_k | 20 | 320 | 1.72182 | 1.95992 | 0 |  |  | yes |  |
| sampling | probe | negative_control | edgar | zlib_ratio | host_top_k | 20 | 320 | 0 | 0 | 1 |  |  | yes |  |
| sampling | probe | negative_control | edgar | estimate | host_top_k | 20 | 320 | 0.21057 | 0.245349 | 0 |  |  | yes |  |
| sampling | probe | negative_control | edgar | loss | host_top_k | 50 | 320 | 0 | 0 | 1 |  |  | yes |  |
| sampling | probe | negative_control | edgar | min_k | host_top_k | 50 | 320 | 0 | 0 | 1 |  |  | yes |  |
| sampling | probe | negative_control | edgar | min_k_pp | host_top_k | 50 | 320 | 0.950581 | 1.09784 | 0 |  |  | yes |  |
| sampling | probe | negative_control | edgar | zlib_ratio | host_top_k | 50 | 320 | 0 | 0 | 1 |  |  | yes |  |
| sampling | probe | negative_control | edgar | estimate | host_top_k | 50 | 320 | 0.121236 | 0.143441 | 0 |  |  | yes |  |
| sampling | probe | negative_control | edgar | loss | host_top_k | 100 | 320 | 0 | 0 | 1 |  |  | yes |  |
| sampling | probe | negative_control | edgar | min_k | host_top_k | 100 | 320 | 0 | 0 | 1 |  |  | yes |  |
| sampling | probe | negative_control | edgar | min_k_pp | host_top_k | 100 | 320 | 0 | 0 | 1 |  |  | yes |  |
| sampling | probe | negative_control | edgar | zlib_ratio | host_top_k | 100 | 320 | 0 | 0 | 1 |  |  | yes |  |
| sampling | probe | negative_control | edgar | estimate | host_top_k | 100 | 320 | 0 | 0 | 1 |  |  | yes |  |
| sampling | probe | negative_control | edgar | loss | served_top_k | 5 | 320 | 1.27952 | 2.37404 | 0 |  |  | yes |  |
| sampling | probe | negative_control | edgar | min_k | served_top_k | 5 | 320 | 8.30826 | 11.981 | 0 |  |  | yes |  |
| sampling | probe | negative_control | edgar | min_k_pp | served_top_k | 5 | 320 | 0.328392 | 1.45064 | 0 |  |  | yes |  |
| sampling | probe | negative_control | edgar | zlib_ratio | served_top_k | 5 | 320 | 0.46588 | 0.83584 | 0 |  |  | yes |  |
| sampling | probe | negative_control | edgar | estimate | served_top_k | 5 | 320 | 0.0631755 | 0.423397 | 0 |  |  | yes |  |
| sampling | probe | negative_control | edgar | mass | host_top_k | 1 | 320 |  |  |  | 0.977514 | 0.395332 | yes |  |
| sampling | probe | negative_control | edgar | mass | host_top_k | 5 | 320 |  |  |  | 0.999416 | 0.932908 | yes |  |
| sampling | probe | negative_control | edgar | mass | host_top_k | 10 | 320 |  |  |  | 0.999873 | 0.994095 | yes |  |
| sampling | probe | negative_control | edgar | mass | host_top_k | 20 | 320 |  |  |  | 0.999934 | 0.996568 | yes |  |
| sampling | probe | negative_control | edgar | mass | host_top_k | 50 | 320 |  |  |  | 0.999977 | 0.998581 | yes |  |
| sampling | probe | negative_control | edgar | mass | host_top_k | 100 | 320 |  |  |  | 0.999989 | 0.999241 | yes |  |
| sampling | probe | negative_control | edgar | mass | served_top_k | 5 | 320 |  |  |  | 0.99999 | 0.995542 | yes |  |
