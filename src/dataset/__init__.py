"""FMP-backed calibration corpus builder.

Public API per design: ``ArticleRecord``, ``build_calibration``, ``update_oos``.

The re-exports use a lazy ``__getattr__`` shim so that ``python -m
src.dataset.fmp_corpora`` does not warn about double-import on package
initialisation.
"""

from __future__ import annotations

from typing import Any

__all__ = ["ArticleRecord", "build_calibration", "update_oos"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from src.dataset import fmp_corpora

        return getattr(fmp_corpora, name)
    raise AttributeError(f"module 'src.dataset' has no attribute {name!r}")
