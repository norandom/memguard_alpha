"""One-shot generator for notebooks/visualize_run.ipynb.

The notebook is a stripped-down companion to qualification.ipynb: it loads a
finished run directory's artifacts (records.jsonl, summary.csv, top3.md) and
renders the paper-ready figures via src.harness.plots. No equations, no LaTeX
cells — just visualisation of a real run for paper inclusion.
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
        "# Visualize Run\n"
        "\n"
        "Loads a completed harness run directory and renders the paper-ready "
        "figures for that run. Companion to `qualification.ipynb` (which holds "
        "the methodology + LaTeX equations); this notebook is for **plots only**."
    ),
    md("## 1. Configure"),
    code(
        "from pathlib import Path\n"
        "import csv\n"
        "import json\n"
        "import re\n"
        "\n"
        "# Edit RUN_DIR to point at any completed run.\n"
        "RUN_DIR = Path('runs/20260427_213853')\n"
        "\n"
        "RECORDS_PATH = RUN_DIR / 'records.jsonl'\n"
        "SUMMARY_PATH = RUN_DIR / 'summary.csv'\n"
        "TOP3_PATH    = RUN_DIR / 'top3.md'\n"
        "FIGURES_DIR  = Path('notebooks/figures'); FIGURES_DIR.mkdir(parents=True, exist_ok=True)\n"
        "\n"
        "for p in [RECORDS_PATH, SUMMARY_PATH, TOP3_PATH]:\n"
        "    assert p.exists(), f'missing artifact: {p}'\n"
        "print(f'Loading run: {RUN_DIR}')"
    ),
    md(
        "## 2. Apply paper style\n"
        "\n"
        "Single-column width, colorblind-safe palette, B&W-distinguishable markers."
    ),
    code(
        "from src.harness import configure_paper_style\n"
        "configure_paper_style()"
    ),
    md(
        "## 3. Reconstruct dataclasses from on-disk artifacts\n"
        "\n"
        "The harness writes primitive types to JSONL / CSV; the plot helpers want frozen\n"
        "dataclasses. The cell below rebuilds `Record`, `CIBound`, `ModelEvalResult`, and\n"
        "`CompositeScore` instances from the run files."
    ),
    code(
        "from src.harness import Record, CIBound, ModelEvalResult, CompositeScore\n"
        "from src.mia import MiaFeatures\n"
        "\n"
        "def _load_records(path: Path) -> dict[str, list[Record]]:\n"
        "    by_model: dict[str, list[Record]] = {}\n"
        "    with path.open() as f:\n"
        "        for line in f:\n"
        "            row = json.loads(line)\n"
        "            feats = MiaFeatures(**row['features_raw']) if row.get('features_raw') else None\n"
        "            rec = Record(\n"
        "                model=row['model'],\n"
        "                prompt_hash=row['prompt_hash'],\n"
        "                parse_ok=row['parse_ok'],\n"
        "                predicted_direction=row['predicted_direction'],\n"
        "                raw_confidence=row['raw_confidence'],\n"
        "                penalized_confidence=row['penalized_confidence'],\n"
        "                target_direction=row['target_direction'],\n"
        "                features_raw=feats,\n"
        "                features_standardised=row.get('features_standardised') or {},\n"
        "                p_memorized=row.get('p_memorized'),\n"
        "                fail_reason=row.get('fail_reason'),\n"
        "            )\n"
        "            by_model.setdefault(rec.model, []).append(rec)\n"
        "    return by_model\n"
        "\n"
        "def _f(s: str) -> float:\n"
        "    return float(s) if s not in ('', None) else 0.0\n"
        "\n"
        "def _load_summary(path: Path):\n"
        "    results: list[ModelEvalResult] = []\n"
        "    majority: CIBound | None = None\n"
        "    raw_rows: list[dict] = []\n"
        "    with path.open() as f:\n"
        "        for row in csv.DictReader(f):\n"
        "            raw_rows.append(row)\n"
        "    return raw_rows\n"
        "\n"
        "records_by_model = _load_records(RECORDS_PATH)\n"
        "summary_rows = _load_summary(SUMMARY_PATH)\n"
        "\n"
        "results: list[ModelEvalResult] = []\n"
        "scores: list[CompositeScore] = []\n"
        "majority: CIBound | None = None\n"
        "\n"
        "for row in summary_rows:\n"
        "    name = row['model']\n"
        "    if name == '__majority_baseline__':\n"
        "        majority = CIBound(_f(row['raw_acc_point']), _f(row['raw_acc_lo']), _f(row['raw_acc_hi']))\n"
        "        continue\n"
        "    raw_acc = CIBound(_f(row['raw_acc_point']), _f(row['raw_acc_lo']), _f(row['raw_acc_hi']))\n"
        "    mg_acc  = CIBound(_f(row['memguard_acc_point']), _f(row['memguard_acc_lo']), _f(row['memguard_acc_hi']))\n"
        "    auc     = CIBound(_f(row['mcs_auc_point']), _f(row['mcs_auc_lo']), _f(row['mcs_auc_hi']))\n"
        "    parse   = _f(row['parse_success_rate'])\n"
        "    pf      = int(row['parse_failures']) if row['parse_failures'] not in ('', None) else 0\n"
        "    warns   = [w for w in (row['warnings'] or '').split(',') if w.strip()]\n"
        "    recs    = records_by_model.get(name, [])\n"
        "    results.append(ModelEvalResult(\n"
        "        model=name, raw_accuracy=raw_acc, memguard_accuracy=mg_acc,\n"
        "        mcs_auc=auc, parse_success_rate=parse, parse_failures=pf,\n"
        "        warnings=warns, records=recs,\n"
        "    ))\n"
        "    scores.append(CompositeScore(\n"
        "        model=name,\n"
        "        score=_f(row['score']),\n"
        "        components={'memguard_acc_lo': mg_acc.lo, 'mcs_auc_point': auc.point, 'parse_success_rate': parse},\n"
        "        survives_gates=(row['survives_gates'].lower() == 'true'),\n"
        "        warnings=warns,\n"
        "    ))\n"
        "\n"
        "assert majority is not None, '__majority_baseline__ row missing from summary.csv'\n"
        "print(f'Loaded {len(results)} models with {sum(len(r.records) for r in results)} records total.')\n"
        "print(f'Majority baseline: point={majority.point:.3f} CI=[{majority.lo:.3f}, {majority.hi:.3f}]')"
    ),
    md("## 4. Top-3 narrative"),
    code(
        "from IPython.display import Markdown, display\n"
        "display(Markdown(TOP3_PATH.read_text()))"
    ),
    md("## 5. Figure: Accuracy with bootstrap 95% CI vs majority baseline"),
    code(
        "from src.harness import plot_accuracy_with_ci\n"
        "fig = plot_accuracy_with_ci(results, majority)\n"
        "fig.savefig(FIGURES_DIR / 'accuracy_ci.pdf')\n"
        "fig"
    ),
    md("## 6. Figure: MCS-AUC with bootstrap 95% CI"),
    code(
        "from src.harness import plot_mcs_auc_with_ci\n"
        "fig = plot_mcs_auc_with_ci(results)\n"
        "fig.savefig(FIGURES_DIR / 'mcs_auc_ci.pdf')\n"
        "fig"
    ),
    md("## 7. Figure: Composite ranking"),
    code(
        "from src.harness import plot_composite_ranking\n"
        "fig = plot_composite_ranking(scores)\n"
        "fig.savefig(FIGURES_DIR / 'composite_ranking.pdf')\n"
        "fig"
    ),
    md(
        "## 8. Saved figures\n"
        "\n"
        "All three are written to `notebooks/figures/*.pdf` at single-column width "
        "for direct inclusion in a two-column manuscript."
    ),
    code(
        "for p in sorted(FIGURES_DIR.glob('*.pdf')):\n"
        "    print(p, p.stat().st_size, 'bytes')"
    ),
]

NB.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

OUT_PATH = Path("notebooks/visualize_run.ipynb")
nbf.write(NB, OUT_PATH)
print(f"wrote {OUT_PATH}")
