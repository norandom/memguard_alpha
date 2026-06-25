"""Generate the API-reference pages and literate nav at mkdocs build time.

Run by ``mkdocs-gen-files``. Walks the ``recall_guard`` package and writes one
``reference/<module>.md`` stub per public module (each is just a ``::: dotted.path``
mkdocstrings directive), plus ``reference/SUMMARY.md`` consumed by
``mkdocs-literate-nav``. Pages are generated from docstrings + type hints, so the
reference always reflects the code (Req 8.1, 8.3); private modules (leading
underscore) are skipped.
"""

from __future__ import annotations

from pathlib import Path

import mkdocs_gen_files

PACKAGE = "recall_guard"
_ROOT = Path(__file__).resolve().parent.parent
nav = mkdocs_gen_files.Nav()

for path in sorted((_ROOT / PACKAGE).rglob("*.py")):
    module_path = path.relative_to(_ROOT).with_suffix("")
    doc_path = path.relative_to(_ROOT).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)
    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1].startswith("_"):
        continue

    if not parts:
        continue

    nav[parts] = doc_path.as_posix()
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        identifier = ".".join(parts)
        fd.write(f"# `{identifier}`\n\n::: {identifier}\n")
    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(_ROOT))

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
