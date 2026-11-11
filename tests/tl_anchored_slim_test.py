# Copyright 2026- blackjax-devs.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TL-anchored slim: in `slim` mode, every agent is trimmed to the lead's
wall-clock floor (derived from inbound-message timestamps in its journal), so a
restart can't desync the team (the busy coordinator getting a shorter recent
window than its peers). Idle agents keep only a last-turn breadcrumb.

A "window" is what sagent resumes from: the records of the journal's last
context window — a ``ContextClear``, bookkeeping (the model in force, each
turn's ``TokenUsage``), and the conversation.
"""

import dataclasses

from datetime import UTC, datetime

import pytest

from sagent.runtime.types import AgentSendQueuedMessage
from sagent.session.types import (
    AgentToAgentMessage,
    AssistantMessage,
    ContextClear,
    TokenUsage,
    ToolCall,
    UncategorizedRecord,
    UncategorizedToolResult,
    UserMessage,
)

import agent_team.serve as serve


@pytest.fixture(autouse=True)
def _slim_mode(monkeypatch):
    monkeypatch.setattr(serve, "_RESUME_MODE", "slim")
    monkeypatch.setattr(serve, "_RESUME_KEEP_RECORDS", 4)
    monkeypatch.setattr(serve, "_RESUME_LEAD_ROLE", "tl")


def _iso(ts):
    return datetime.fromtimestamp(ts, tz=UTC).isoformat(timespec="seconds")


def _turn(ts, *, peer=False):
    """An inbound (user-role) message at wall-clock ``ts`` + an assistant reply."""
    inbound = (
        AgentToAgentMessage(sender="peer", content=f"in@{ts}", timestamp=_iso(ts))
        if peer
        else UserMessage(content=f"in@{ts}", timestamp=_iso(ts))
    )
    return [inbound, AssistantMessage(content=f"reply@{ts}")]


def _tape(*tss, peer=False):
    out = []
    for ts in tss:
        out += _turn(ts, peer=peer)
    return out


def _floor(records):
    return serve._inbound_floor_ts(records)


def test_anchors_peers_to_lead_floor():
    # lead is busy: 4 turns; keep=4 records => last 2 turns => floor = 1500.
    lead = _tape(500, 1000, 1500, 2000)
    active = _tape(800, 1600, 2100)  # spans before+after the floor
    idle = _tape(600, 900)  # entirely before the floor
    plan = serve._plan_slim_resume(dict(tl=lead, swe=active, statistician=idle))

    lead_slim, _ = plan["tl"]
    lead_floor = _floor(lead_slim)
    assert lead_floor == 1500.0, lead_floor
    assert len(lead_slim) == 4  # snapped to the last 2 turns

    # active peer: anchored — its window starts at the first inbound >= floor.
    a_slim, a_note = plan["swe"]
    assert _floor(a_slim) >= lead_floor, (_floor(a_slim), lead_floor)
    assert _floor(a_slim) == 1600.0  # u@1600, not the older u@800
    assert "anchored" in a_note

    # idle peer: kept only its most recent turn (breadcrumb), not its full window.
    i_slim, _ = plan["statistician"]
    assert len(i_slim) == 2, i_slim  # one turn: u@900 + reply
    assert _floor(i_slim) == 900.0  # its last turn, not u@600 too

    # the active agents share one horizon: no active agent older than the lead.
    active_floors = [_floor(lead_slim), _floor(a_slim)]
    assert max(active_floors) - min(active_floors) <= 200  # within the window


def test_agent_send_is_a_boundary():
    # peer-message (AgentToAgentMessage) inbound also anchors correctly.
    lead = _tape(1000, 2000)
    peer = _tape(500, 1500, 2500, peer=True)
    plan = serve._plan_slim_resume(dict(tl=lead, swe=peer))
    lead_floor = _floor(plan["tl"][0])
    assert _floor(plan["swe"][0]) >= lead_floor


def test_no_lead_falls_back_to_per_agent(monkeypatch):
    monkeypatch.setattr(serve, "_RESUME_LEAD_ROLE", "absent")
    big = _tape(*(100 * i for i in range(1, 9)))  # 16 records
    plan = serve._plan_slim_resume(dict(swe=big))
    slim, note = plan["swe"]
    assert len(slim) == 4 and "per-agent" in note  # _RESUME_KEEP_RECORDS slim


def test_full_resumes_whole_window(monkeypatch):
    monkeypatch.setattr(serve, "_RESUME_MODE", "full")
    big = _tape(*(100 * i for i in range(1, 9)))
    plan = serve._plan_slim_resume(dict(tl=big))
    slim, note = plan["tl"]
    assert slim is big and note == ""  # untouched


@pytest.mark.parametrize(
    ("raw", "mode", "warns"),
    [
        ("slim", "slim", False),
        ("FULL ", "full", False),
        # sagent 0.2 resumes no claude session file: the old default selects
        # what was its fallback, and says so.
        ("materialize", "slim", True),
        ("bogus", "slim", True),
    ],
)
def test_resume_mode_resolution(raw, mode, warns):
    resolved, warning = serve._resolve_resume_mode(raw)
    assert resolved == mode
    assert bool(warning) is warns


def test_inbound_floor_and_message_fields():
    # guard: the timestamp field anchoring reads still exists.
    assert "timestamp" in {f.name for f in dataclasses.fields(UserMessage)}
    assert "timestamp" in {f.name for f in dataclasses.fields(AgentToAgentMessage)}
    recs = _turn(777.0)
    assert serve._inbound_floor_ts(recs) == 777.0


def test_held_peer_message_is_dated_by_the_reply_it_drew():
    """A peer's message is pushed as ``AgentSendQueuedMessage``, which sagent
    journals without a timestamp; the ``TokenUsage`` after it dates the turn."""
    # guard: the queued message really has nowhere to carry a timestamp.
    assert "timestamp" not in {
        f.name for f in dataclasses.fields(AgentSendQueuedMessage)
    }
    window = [
        AgentToAgentMessage(sender="swe", content="old"),
        AssistantMessage(content="a1"),
        TokenUsage(timestamp=_iso(1000)),
        AgentToAgentMessage(sender="swe", content="new"),
        AssistantMessage(content="a2"),
        TokenUsage(timestamp=_iso(2000)),
    ]
    assert serve._inbound_floor_ts(window) == 1000.0
    anchored = serve._slim_tape_to_floor(window, 1500.0)
    assert [r.content for r in anchored if serve._is_inbound(r)] == ["new"]


def test_slim_keeps_bookkeeping_and_drops_the_compaction_summary():
    """What a resume needs whole survives a slim; what lay before the cut does not."""
    model = UncategorizedRecord(kind="model", payload={"standing": "model"})
    head = ContextClear(
        system_prompt="You are TL.",
        summary="everything before",
        history=tuple(_tape(100)),
        extra={"spend_usd": 1.5},
    )
    usage = [TokenUsage(info={"spend_usd": 0.25}) for _ in range(4)]
    window = [head, model]
    for ts, spent in zip((200, 300, 400, 500), usage, strict=True):
        window += [*_turn(ts), spent]

    slim, dropped = serve._slim_resume_tape(window, 4)

    assert dropped == 6  # the kept history's turn + the turns at 200 and 300
    new_head = slim[0]
    assert isinstance(new_head, ContextClear)
    assert new_head.summary is None and new_head.history == ()
    assert new_head.system_prompt == "You are TL."
    assert new_head.extra["spend_usd"] == 1.5  # spend before the window
    assert model in slim
    assert [r for r in slim if isinstance(r, TokenUsage)] == usage  # every turn's spend
    assert [r.content for r in slim if serve._is_inbound(r)] == ["in@400", "in@500"]


def test_slim_never_starts_on_a_dangling_tool_result():
    """The cut snaps forward to an inbound message, so no result loses its call."""
    window = [
        UserMessage(content="u1", timestamp=_iso(100)),
        ToolCall(call_id="c1", name="Read"),
        UncategorizedToolResult(call_id="c1", content="file"),
        AssistantMessage(content="a1"),
        UserMessage(content="u2", timestamp=_iso(200)),
        AssistantMessage(content="a2"),
    ]
    slim, _ = serve._slim_resume_tape(window, 4)  # raw tail would start at the result
    assert [type(r).__name__ for r in slim] == ["UserMessage", "AssistantMessage"]
    assert slim[0].content == "u2"


def test_small_window_is_left_untouched():
    window = [ContextClear(), *_tape(100, 200)]
    slim, dropped = serve._slim_resume_tape(window, 4)
    assert slim is window and dropped == 0
    assert serve._slim_resume_tape(window, 0) == (window, 0)  # keep<=0 disables
