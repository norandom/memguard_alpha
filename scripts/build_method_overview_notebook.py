"""One-shot generator for notebooks/method_overview.ipynb.

Companion notebook explaining how the project uses the IS and OOS calibration
corpora. Reads on-disk data only (no API calls):
  - data/cutoffs.yaml           (per-model training cutoffs)
  - data/calibration/is_memorized.jsonl  (the earlier-date corpus)
  - data/calibration/oos_control.jsonl   (the later-date control corpus)

Produces a cutoff timeline diagram, year-distribution histograms, a sample-row
table, and a length-distribution comparison. All figures save to
notebooks/figures/method_*.pdf.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NB = nbf.v4.new_notebook()


def md(src: str) -> dict:
    return nbf.v4.new_markdown_cell(src)


def code(src: str) -> dict:
    return nbf.v4.new_code_cell(src)


NB.cells = [
    md(
        "# Method Overview: Why Two Calibration Corpora\n"
        "\n"
        "This notebook explains the data setup behind the honest-model-ranking method,\n"
        "for paper reviewers and anyone who hasn't read the spec end-to-end.\n"
        "\n"
        "**The core idea in one paragraph.** Every LLM in the registry has a documented\n"
        "training cutoff. Material published before that cutoff could have been seen by\n"
        "the model; material published after it was published later than the stated cutoff. We construct two\n"
        "labelled corpora drawn from real, dated FMP news articles:\n"
        "\n"
        "- **IS-memorized** — articles published *before* the earliest model cutoff,\n"
        "  so they sit in the earlier-date bucket for every model in the registry.\n"
        "- **OOS-control** — articles published *after* the latest model cutoff, so they sit\n"
        "  in the later-date control bucket.\n"
        "\n"
        "We then ask each model to score those articles and observe its log-probability\n"
        "signature on each. The MCS classifier (Membership inference Contamination Score)\n"
        "learns to tell the two apart per model. That trained classifier is what flags\n"
        "memorisation on the actual eval prompts.\n"
        "\n"
        "Without these two corpora, the classifier has nothing to learn from. Every\n"
        "downstream claim depends on this setup making sense, which is why this notebook\n"
        "exists separately from the equations / figures one (`qualification.ipynb`)."
    ),
    md("## 1. Imports & paper style"),
    code(
        "import sys\n"
        "from pathlib import Path\n"
        "import json\n"
        "from collections import Counter\n"
        "from datetime import date\n"
        "import yaml\n"
        "\n"
        "# Make the repository root importable regardless of where the notebook runs from.\n"
        "ROOT = Path.cwd()\n"
        "if not (ROOT / 'pyproject.toml').exists():\n"
        "    ROOT = ROOT.parent\n"
        "if str(ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(ROOT))\n"
        "\n"
        "import matplotlib.pyplot as plt\n"
        "import matplotlib.patches as mpatches\n"
        "import numpy as np\n"
        "\n"
        "from recall_guard.harness import configure_paper_style\n"
        "configure_paper_style()\n"
        "\n"
        "FIGURES_DIR = ROOT / 'notebooks' / 'figures'\n"
        "FIGURES_DIR.mkdir(parents=True, exist_ok=True)"
    ),
    md("## 2. Load the data on disk"),
    code(
        "def _load_jsonl(path: Path) -> list[dict]:\n"
        "    with path.open() as f:\n"
        "        return [json.loads(l) for l in f if l.strip()]\n"
        "\n"
        "with (ROOT / 'data' / 'cutoffs.yaml').open() as f:\n"
        "    cutoffs = {m: d for m, d in yaml.safe_load(f)['models'].items()}\n"
        "\n"
        "is_rows = _load_jsonl(ROOT / 'data' / 'calibration' / 'is_memorized.jsonl')\n"
        "oos_rows = _load_jsonl(ROOT / 'data' / 'calibration' / 'oos_control.jsonl')\n"
        "\n"
        "earliest_cutoff = min(cutoffs.values())\n"
        "latest_cutoff   = max(cutoffs.values())\n"
        "\n"
        "print(f'Active models in registry: {len(cutoffs)}')\n"
        "print(f'  earliest cutoff: {earliest_cutoff}')\n"
        "print(f'  latest cutoff:   {latest_cutoff}')\n"
        "print(f'IS-memorized rows: {len(is_rows)} (label=1)')\n"
        "print(f'OOS-control rows: {len(oos_rows)} (label=0)')"
    ),
    md(
        "## 3. The cutoff timeline\n"
        "\n"
        "Every model in the registry has a known training cutoff. The IS-memorized\n"
        "window is everything *before* the earliest cutoff; the OOS-control window is\n"
        "everything *after* the latest cutoff. Articles dated *between* the two cutoffs are\n"
        "memorisable for some models but not others — this harness drops them entirely\n"
        "to keep the labels honest."
    ),
    code(
        "fig, ax = plt.subplots(figsize=(7.0, 3.0))\n"
        "\n"
        "# Span of years covered by both corpora plus the cutoffs.\n"
        "all_dates = (\n"
        "    [date.fromisoformat(r['metadata']['published_at']) for r in is_rows]\n"
        "    + [date.fromisoformat(r['metadata']['published_at']) for r in oos_rows]\n"
        ")\n"
        "x_min = min([min(all_dates), earliest_cutoff]).year - 1\n"
        "x_max = max([max(all_dates), latest_cutoff]).year + 1\n"
        "\n"
        "# Shade the three regions: IS, gap, OOS.\n"
        "ax.axvspan(date(x_min, 1, 1), earliest_cutoff, alpha=0.18, color='#0072B2',\n"
        "           label='IS window (earlier than every cutoff)')\n"
        "ax.axvspan(earliest_cutoff, latest_cutoff, alpha=0.18, color='#999999',\n"
        "           label='gap (model-specific eligibility differs)')\n"
        "ax.axvspan(latest_cutoff, date(x_max, 12, 31), alpha=0.18, color='#D55E00',\n"
        "           label='OOS window (later than every cutoff)')\n"
        "\n"
        "# One row per model, vertical line at its cutoff.\n"
        "for i, (model, cutoff) in enumerate(sorted(cutoffs.items(), key=lambda kv: kv[1])):\n"
        "    short = model.split('/')[-1][:40]\n"
        "    ax.scatter([cutoff], [i], marker='|', s=120, color='black', zorder=3)\n"
        "    ax.text(cutoff, i, f'  {short}', va='center', ha='left', fontsize=6)\n"
        "\n"
        "# Tiny markers for actual corpus rows (helps the reader see they exist).\n"
        "is_dates = [date.fromisoformat(r['metadata']['published_at']) for r in is_rows]\n"
        "oos_dates = [date.fromisoformat(r['metadata']['published_at']) for r in oos_rows]\n"
        "n_models = len(cutoffs)\n"
        "ax.scatter(is_dates, [n_models + 0.4]*len(is_dates), marker='.', s=10,\n"
        "           color='#0072B2', alpha=0.7)\n"
        "ax.scatter(oos_dates, [n_models + 0.4]*len(oos_dates), marker='.', s=10,\n"
        "           color='#D55E00', alpha=0.7)\n"
        "ax.text(date(x_min + 1, 1, 1), n_models + 0.4, 'corpus rows: ',\n"
        "        va='center', ha='right', fontsize=6)\n"
        "\n"
        "ax.set_xlim(date(x_min, 1, 1), date(x_max, 12, 31))\n"
        "ax.set_ylim(-0.5, n_models + 0.9)\n"
        "ax.set_yticks([])\n"
        "ax.set_xlabel('Year')\n"
        "ax.set_title('Training cutoffs vs IS / OOS corpus dates')\n"
        "ax.legend(loc='lower left', fontsize=5)\n"
        "fig.tight_layout()\n"
        "fig.savefig(FIGURES_DIR / 'method_cutoff_timeline.pdf')\n"
        "fig"
    ),
    md(
        "## 4. Year-by-year distribution\n"
        "\n"
        "Same data as the timeline, but counted into year bins so the reader can see\n"
        "how many rows landed where. Both corpora ride on what FMP's archive returned\n"
        "for the requested date windows."
    ),
    code(
        "is_years = Counter(date.fromisoformat(r['metadata']['published_at']).year for r in is_rows)\n"
        "oos_years = Counter(date.fromisoformat(r['metadata']['published_at']).year for r in oos_rows)\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(7.0, 2.5))\n"
        "all_years = sorted(set(is_years) | set(oos_years))\n"
        "x = np.arange(len(all_years))\n"
        "w = 0.4\n"
        "ax.bar(x - w/2, [is_years.get(y, 0) for y in all_years], width=w,\n"
        "       label=f'IS-memorized (n={len(is_rows)})', color='#0072B2')\n"
        "ax.bar(x + w/2, [oos_years.get(y, 0) for y in all_years], width=w,\n"
        "       label=f'OOS-control (n={len(oos_rows)})', color='#D55E00')\n"
        "ax.set_xticks(x)\n"
        "ax.set_xticklabels(all_years)\n"
        "ax.set_xlabel('Publication year')\n"
        "ax.set_ylabel('Articles')\n"
        "ax.set_title('Corpus year distribution')\n"
        "ax.legend(fontsize=6)\n"
        "fig.tight_layout()\n"
        "fig.savefig(FIGURES_DIR / 'method_year_distribution.pdf')\n"
        "fig"
    ),
    md(
        "## 5. What the rows actually look like\n"
        "\n"
        "Three random examples from each side. The classifier sees the **prompt** field\n"
        "(title + body excerpt, capped at ≈1500 chars) and learns from that model's\n"
        "log-probability signature on it. Real news, real dates, real URLs."
    ),
    code(
        "import random\n"
        "from IPython.display import Markdown, display\n"
        "rng = random.Random(7)\n"
        "\n"
        "def _row_summary(rows: list[dict], n: int = 3) -> str:\n"
        "    sample = rng.sample(rows, k=min(n, len(rows)))\n"
        "    lines = []\n"
        "    for r in sample:\n"
        "        head = r['prompt'].split('\\n')[0][:130]\n"
        "        date_s = r['metadata']['published_at']\n"
        "        url = r['metadata'].get('url', '')\n"
        "        lines.append(f'- **{date_s}** — {head}…  \\n  {url}')\n"
        "    return '\\n'.join(lines)\n"
        "\n"
        "display(Markdown('### IS-memorized samples (label = 1, earlier-date bucket)'))\n"
        "display(Markdown(_row_summary(is_rows)))\n"
        "display(Markdown('### OOS-control samples (label = 0, later-date control bucket)'))\n"
        "display(Markdown(_row_summary(oos_rows)))"
    ),
    md(
        "## 6. Length distribution\n"
        "\n"
        "If the IS rows were systematically longer or shorter than the OOS rows, the\n"
        "classifier would learn `length -> label` instead of the intended feature signal,\n"
        "and the whole method becomes a length detector. This plot is the sanity check:\n"
        "the two distributions should overlap heavily."
    ),
    code(
        "is_lens  = [len(r['prompt']) for r in is_rows]\n"
        "oos_lens = [len(r['prompt']) for r in oos_rows]\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(7.0, 2.5))\n"
        "bins = np.linspace(0, max(is_lens + oos_lens) + 100, 30)\n"
        "ax.hist(is_lens, bins=bins, alpha=0.55, color='#0072B2',\n"
        "        label=f'IS (median={int(np.median(is_lens))})')\n"
        "ax.hist(oos_lens, bins=bins, alpha=0.55, color='#D55E00',\n"
        "        label=f'OOS (median={int(np.median(oos_lens))})')\n"
        "ax.set_xlabel('Prompt length (chars)')\n"
        "ax.set_ylabel('Articles')\n"
        "ax.set_title('Prompt-length distribution: IS vs OOS')\n"
        "ax.legend(fontsize=6)\n"
        "fig.tight_layout()\n"
        "fig.savefig(FIGURES_DIR / 'method_length_distribution.pdf')\n"
        "fig"
    ),
    md(
        "## 7. What the MCS classifier does with this\n"
        "\n"
        "For each model in the shortlist, the harness sends every IS row and every OOS\n"
        "row through that model and records the per-token log-probabilities. From those\n"
        "log-probabilities it computes five MIA features (Loss, Min-K%, Min-K%++, zlib\n"
        "ratio, ref-delta) and standardises them against the model's own control-corpus\n"
        "baseline.\n"
        "\n"
        "Then it trains a per-model logistic regression on `(features, label)` pairs\n"
        "where label=1 for IS-memorized rows and label=0 for OOS-control rows. The\n"
        "trained classifier becomes that model's `p(memorized | features)` score model,\n"
        "and the harness applies it to every row in the actual eval set during scoring.\n"
        "\n"
        "**The accuracy of that classifier on a held-out IS/OOS split** (the MCS-AUC\n"
        "metric in the run report) is what tells us whether the method is even working\n"
        "for that model. If IS and OOS look indistinguishable to the classifier (AUC\n"
        "near 0.5), the model gets flagged `weak-calibration` and dropped from the\n"
        "ranking. That's the spec's `min_auc=0.6` gate.\n"
        "\n"
        "## Saved figures\n"
        "\n"
        "All three diagrams are written as single-column-width vector PDFs to\n"
        "`notebooks/figures/method_*.pdf`, ready to drop into a two-column manuscript."
    ),
    code(
        "for p in sorted(FIGURES_DIR.glob('method_*.pdf')):\n"
        "    print(p, p.stat().st_size, 'bytes')"
    ),
]

NB.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

OUT_PATH = Path("notebooks/method_overview.ipynb")
nbf.write(NB, OUT_PATH)
print(f"wrote {OUT_PATH}")
