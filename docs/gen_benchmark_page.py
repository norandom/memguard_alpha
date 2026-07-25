"""Generate the benchmark page from the top-level README.benchmark.md file.

Run by ``mkdocs-gen-files`` so the docs nav can expose the benchmark without
copying the content into a second maintained Markdown file.
"""

from __future__ import annotations

from pathlib import Path

import mkdocs_gen_files

_ROOT = Path(__file__).resolve().parent.parent
source = _ROOT / "README.benchmark.md"
content = source.read_text(encoding="utf-8")

target = Path("benchmark.md")
with mkdocs_gen_files.open(target, "w") as fd:
    fd.write(content)

mkdocs_gen_files.set_edit_path(target, source.relative_to(_ROOT))
