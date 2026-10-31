"""Pure execution-metadata checks shared by experiment publication boundaries.

Inspect structured keys, preserving raw task text verbatim. Only the three
schema-defined label/answer slots may contain arbitrary task JSON.
"""

from collections.abc import Mapping

_CREDENTIAL_FIELDS = frozenset({
    "apikey", "authorization", "accesstoken", "password", "credential", "credentials", "credentialref",
    "xapikey", "bearertoken", "secret", "clientsecret", "headers",
})


def check_execution_metadata(value):
    """Reject credential/header keys without inspecting or echoing values."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key.lower().replace("_", "").replace("-", "") in _CREDENTIAL_FIELDS:
                raise ValueError("Execution metadata must exclude credential and authorization fields")
            check_execution_metadata(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            check_execution_metadata(child)


def check_run_metadata(snapshot):
    """Check a run or journal snapshot, exempting only known task payloads.

    Additional journal fields (including write-ahead intent) remain metadata.
    A nested ``answer`` outside the exact draw/outcome slot is not exempt.
    """
    checkpoint = snapshot["checkpoint"]
    metadata = dict(snapshot) | {
        "checkpoint": dict(checkpoint) | {
            "items": [{key: value for key, value in item.items() if key != "label"}
                      for item in checkpoint["items"]],
            "draws": [dict(draw) | {
                "metadata": {key: value for key, value in draw["metadata"].items() if key != "answer"},
            } for draw in checkpoint["draws"]],
        },
        "arms": [dict(arm) | {
            "outcome": {key: value for key, value in arm["outcome"].items() if key != "answer"},
        } for arm in snapshot["arms"]],
    }
    check_execution_metadata(metadata)
