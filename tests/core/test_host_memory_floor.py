"""Host memory floor in the profile's host requirements (task 6.4, Requirement 11.1).

A full-distribution host refuses new requests with 503 ``memory_low`` while free physical
memory is below ``memory_floor_bytes``. The floor defaults to one quarter of
``min_free_ram_bytes`` and the llama.cpp profile records it explicitly. Offline, no HTTP.
"""
import json

import pytest

from recall_guard.core.profiles import (
    PROFILE_DIRECTORY,
    HostResourceRequirement,
    ProfileError,
    ProviderProfile,
    load_profile,
)

GIB = 1024 ** 3
LLAMACPP = "llamacpp-gemma3-4b"


def test_llamacpp_profile_records_the_floor_at_one_quarter_of_the_load_minimum():
    data = json.loads((PROFILE_DIRECTORY / f"{LLAMACPP}.json").read_text(encoding="utf-8"))
    recorded = data["host_requirements"]["memory_floor_bytes"]
    requirements = load_profile(LLAMACPP).host_requirements
    assert recorded == requirements.memory_floor_bytes == requirements.min_free_ram_bytes // 4 == 1_610_612_736
    assert requirements.memory_floor() == recorded
    assert "memory_low" in load_profile(LLAMACPP).notes


@pytest.mark.parametrize(("minimum", "floor", "expected"), [
    (8 * GIB, None, 2 * GIB),          # Default: one quarter of the load minimum.
    (8 * GIB, 3 * GIB, 3 * GIB),       # An explicit floor wins.
    (None, None, None),                # No load minimum, no floor.
    (None, GIB, GIB),
])
def test_memory_floor_defaults_to_one_quarter_of_the_load_minimum(minimum, floor, expected):
    requirements = HostResourceRequirement(min_free_ram_bytes=minimum, memory_floor_bytes=floor)
    assert requirements.memory_floor() == expected


@pytest.mark.parametrize("floor", [0, -1, True, 1.5, "1GiB"])
def test_memory_floor_must_be_a_positive_integer(floor):
    with pytest.raises(ProfileError, match="memory_floor_bytes"):
        HostResourceRequirement(min_free_ram_bytes=8 * GIB, memory_floor_bytes=floor)


def test_memory_floor_above_the_load_minimum_is_refused():
    with pytest.raises(ProfileError, match="memory_floor_bytes"):
        HostResourceRequirement(min_free_ram_bytes=4 * GIB, memory_floor_bytes=4 * GIB + 1)


def test_floor_is_serialised_only_when_set_so_other_profiles_export_unchanged():
    assert "memory_floor_bytes" not in HostResourceRequirement(min_free_ram_bytes=8 * GIB).to_dict()
    assert "memory_floor_bytes" not in load_profile("ollama-gemma3-4b").to_dict()["host_requirements"]
    profile = load_profile(LLAMACPP)
    exported = profile.to_dict()
    assert exported["host_requirements"]["memory_floor_bytes"] == 1_610_612_736
    assert ProviderProfile.from_dict(exported) == profile


def test_a_profile_file_with_a_bad_floor_is_refused_at_load(tmp_path):
    data = json.loads((PROFILE_DIRECTORY / f"{LLAMACPP}.json").read_text(encoding="utf-8"))
    data["host_requirements"]["memory_floor_bytes"] = 0
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ProfileError, match="memory_floor_bytes"):
        load_profile(path)
