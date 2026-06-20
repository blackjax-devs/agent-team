"""TL-anchored slim: in `slim` mode, every agent is trimmed to the lead's
wall-clock floor (derived from inbound-message timestamps on the tape), so a
restart can't desync the team (the busy coordinator getting a shorter recent
window than its peers). Idle agents keep only a last-turn breadcrumb.
"""

import dataclasses

import pytest

from sagent.types.runtime import AgentSendMessage, AssistantMessage, UserMessage
from sagent.types.tape import ReferrableTapeEvent, TapeRef

import agent_team.serve as serve


@pytest.fixture(autouse=True)
def _slim_mode(monkeypatch):
    monkeypatch.setattr(serve, "_RESUME_MODE", "slim")
    monkeypatch.setattr(serve, "_RESUME_KEEP_RECORDS", 4)
    monkeypatch.setattr(serve, "_RESUME_LEAD_ROLE", "tl")


_ORD = [0]


def _rec(event):
    _ORD[0] += 1
    return ReferrableTapeEvent(
        ref=TapeRef(session_id="s", ordinal=_ORD[0]), event=event
    )


def _turn(ts, *, peer=False):
    """An inbound (user-role) message at wall-clock ``ts`` + an assistant reply."""
    inbound = (
        AgentSendMessage(source="peer", text=f"in@{ts}", timestamp=ts)
        if peer
        else UserMessage(text=f"in@{ts}", timestamp=ts)
    )
    return [_rec(inbound), _rec(AssistantMessage(text=f"reply@{ts}"))]


def _tape(*tss, peer=False):
    out = []
    for ts in tss:
        out += _turn(ts, peer=peer)
    return out


def _loaded(**tapes):
    # _plan_slim_resume only reads loaded[label][1] (the tape).
    return {label: (None, tape, None) for label, tape in tapes.items()}


def _floor(records):
    return serve._inbound_floor_ts(records)


def test_anchors_peers_to_lead_floor():
    # lead is busy: 4 turns; keep=4 records => last 2 turns => floor = 1500.
    lead = _tape(500, 1000, 1500, 2000)
    active = _tape(800, 1600, 2100)  # spans before+after the floor
    idle = _tape(600, 900)  # entirely before the floor
    plan = serve._plan_slim_resume(_loaded(tl=lead, swe=active, statistician=idle))

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
    # peer-message (AgentSendMessage) inbound also anchors correctly.
    lead = _tape(1000, 2000)
    peer = _tape(500, 1500, 2500, peer=True)
    plan = serve._plan_slim_resume(_loaded(tl=lead, swe=peer))
    lead_floor = _floor(plan["tl"][0])
    assert _floor(plan["swe"][0]) >= lead_floor


def test_no_lead_falls_back_to_per_agent(monkeypatch):
    monkeypatch.setattr(serve, "_RESUME_LEAD_ROLE", "absent")
    big = _tape(*(100 * i for i in range(1, 9)))  # 16 records
    plan = serve._plan_slim_resume(_loaded(swe=big))
    slim, note = plan["swe"]
    assert len(slim) == 4 and "per-agent" in note  # _RESUME_KEEP_RECORDS slim


@pytest.mark.parametrize("mode", ["full", "materialize"])
def test_full_and_materialize_resume_whole_tape(monkeypatch, mode):
    monkeypatch.setattr(serve, "_RESUME_MODE", mode)
    big = _tape(*(100 * i for i in range(1, 9)))
    plan = serve._plan_slim_resume(_loaded(tl=big))
    slim, note = plan["tl"]
    assert slim is big and note == ""  # untouched


def test_inbound_floor_and_message_fields():
    # guard: the timestamp field materialize/anchoring read still exists.
    assert "timestamp" in {f.name for f in dataclasses.fields(UserMessage)}
    assert "timestamp" in {f.name for f in dataclasses.fields(AgentSendMessage)}
    recs = _turn(777.0)
    assert serve._inbound_floor_ts(recs) == 777.0
