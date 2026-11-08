"""Suite-wide determinism: the same tests pass natively and inside the shared container."""

import pytest


@pytest.fixture(autouse=True)
def _not_in_container_unless_a_test_says_so(monkeypatch):
    # Host preflight refuses local memory inside a container (C20). Tests that
    # exercise that path pass in_container=True or set the variable to 1 themselves.
    monkeypatch.setenv("RECALL_GUARD_IN_CONTAINER", "0")
