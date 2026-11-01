"""Inspect installed software without loading models or GPU libraries.

The image content ID/digest comes from Docker's image inspection, outside the
container. Pair this inventory with that identity in the execution plan.
"""

from __future__ import annotations

import hashlib
import json
import platform
from importlib.metadata import distributions
from pathlib import Path


def environment_inventory() -> dict[str, object]:
    """Return installed distributions and the exact checkout lock identity."""
    lock_path = Path(__file__).resolve().parents[2] / "uv.lock"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        "dependencies": dict(sorted(
            (distribution.metadata["Name"], distribution.version)
            for distribution in distributions()
        )),
    }


if __name__ == "__main__":
    print(json.dumps(environment_inventory(), indent=2, sort_keys=True))
