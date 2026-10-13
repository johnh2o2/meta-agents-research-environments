# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Tests for the lightweight scenario loader in gaia2-core."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gaia2_core.loader import ScenarioLoader


def _write_scenario(tmp_path: Path, apps: list[dict]) -> Path:
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(json.dumps({"apps": apps, "events": []}))
    return scenario_path


def test_extract_user_details_from_internal_contacts(tmp_path: Path) -> None:
    scenario_path = _write_scenario(
        tmp_path,
        [
            {
                "name": "InternalContacts",
                "class_name": "InternalContacts",
                "app_state": {
                    "contacts": {
                        "user": {
                            "first_name": "Astrid",
                            "last_name": "Lundqvist",
                            "address": "Ostermalmsgatan 12",
                            "is_user": True,
                        }
                    }
                },
            }
        ],
    )

    loader = ScenarioLoader(str(scenario_path))
    user_details = loader._extract_user_details()

    assert user_details is not None
    assert user_details.first_name == "Astrid"
    assert user_details.last_name == "Lundqvist"
    assert user_details.address == "Ostermalmsgatan 12"


# ---------------------------------------------------------------------------
# Media refs on the initial USER turn
# ---------------------------------------------------------------------------

_SHA = "f7bfd6bc685b4f733ac5443dbec091dacfebb7fc4daf440e618a661a7e5e908b"
_REF = {"asset_id": "a_f7bfd6bc685b4f733ac5443dbec091da", "mime_type": "audio/wav", "sha256": _SHA}


def _user_event(event_id: str, content: str, deps: list[str], media=None) -> dict:
    ev = {
        "class_name": "Event",
        "event_type": "USER",
        "event_id": event_id,
        "dependencies": deps,
        "action": {
            "app": "AgentUserInterface",
            "function": "send_message_to_agent",
            "operation_type": "READ",
            "args": [
                {"name": "content", "value": content, "value_type": "str"},
                {"name": "attachments", "value": "[]", "value_type": "list"},
            ],
        },
    }
    if media is not None:
        ev["media"] = media
    return ev


def _write_events(tmp_path: Path, events: list[dict]) -> str:
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(json.dumps({"apps": [], "events": events}))
    return str(scenario_path)


def test_text_only_scenario_has_no_task_media(tmp_path: Path) -> None:
    loader = ScenarioLoader(_write_events(tmp_path, [_user_event("u0", "Find the file.", [])]))
    assert loader.task == "Find the file."
    assert loader.task_media == []


def test_initial_turn_media_is_parsed(tmp_path: Path) -> None:
    events = [
        _user_event("u1", "Follow-up.", ["u0"]),
        _user_event("u0", "[User request provided as audio]", [], media=[_REF]),
    ]
    loader = ScenarioLoader(_write_events(tmp_path, events))
    assert loader.task == "[User request provided as audio]"
    assert loader.task_media == [_REF]
    # media never leaks into the action args the agent-side code reads
    assert set(loader.events[1].action.args) == {"content", "attachments"}


def test_media_on_later_user_turn_is_not_the_task_media(tmp_path: Path) -> None:
    events = [
        _user_event("u0", "Hi", []),
        _user_event("u1", "Later", ["u0"], media=[_REF]),
    ]
    assert ScenarioLoader(_write_events(tmp_path, events)).task_media == []


@pytest.mark.parametrize(
    ("media", "match"),
    [
        ({"asset_id": "x"}, "must be a list"),
        (["a_x"], "exactly the keys"),
        ([{**_REF, "path": "assets/x.wav"}], "exactly the keys"),
        ([{**_REF, "asset_id": "../etc/passwd"}], "asset_id"),
        ([{**_REF, "mime_type": "audio/mpeg"}], "mime_type"),
        ([{**_REF, "sha256": _SHA.upper()}], "sha256"),
    ],
)
def test_malformed_media_raises(tmp_path: Path, media, match: str) -> None:
    path = _write_events(tmp_path, [_user_event("u0", "x", [], media=media)])
    with pytest.raises(ValueError, match=match):
        ScenarioLoader(path)
