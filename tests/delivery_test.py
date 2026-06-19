"""Tests for the audit-log helpers in :mod:`mcp_sagent.delivery`.

The HTTP-bridged ``route_send`` / ``schedule_defer`` are exercised by
the live smoke test (see README § "Verified end-to-end"); unit-testing
them would require a fake HTTP server fixture and would mostly be
testing ``urllib.request`` rather than logic. The audit-log primitives
below ARE process-local and DO have unit-test surface.
"""

from __future__ import annotations

import json

from agent_team.mcp_sagent import delivery

import pytest


@pytest.fixture
def isolated_log(tmp_path, monkeypatch):
    log_path = tmp_path / "main.jsonl"
    monkeypatch.setattr(delivery, "MAIN_JSONL_PATH", log_path)
    return log_path


def test_append_record_writes_chat_compatible_line(isolated_log):
    delivery.append_record(from_role="tl", to=["swe"], body="hello")
    rec = json.loads(isolated_log.read_text().strip())
    assert rec["from"] == "tl"
    assert rec["to"] == ["swe"]
    assert rec["body"] == "hello"
    assert rec["ts"].endswith("Z")


def test_append_user_message_uses_user_as_from(isolated_log):
    delivery.append_user_message(to_role="tl", body="prompt from operator")
    rec = json.loads(isolated_log.read_text().strip())
    assert rec["from"] == "user"
    assert rec["to"] == ["tl"]


def test_multiple_records_append_in_order(isolated_log):
    delivery.append_record(from_role="tl", to=["swe"], body="one")
    delivery.append_record(from_role="swe", to=["tl"], body="two")
    delivery.append_record(from_role="tl", to=["user"], body="three")
    lines = isolated_log.read_text().splitlines()
    assert len(lines) == 3
    recs = [json.loads(l) for l in lines]
    assert [r["body"] for r in recs] == ["one", "two", "three"]


def test_iso8601_z_format():
    s = delivery.iso8601_z()
    # YYYY-MM-DDTHH:MM:SS.mmmZ — 24 chars.
    assert len(s) == 24
    assert s[4] == "-" and s[10] == "T" and s[19] == "."
    assert s.endswith("Z")


def test_cancel_all_deferred_returns_zero_in_http_bridge_mode():
    # The in-process scheduler has been replaced by the HTTP bridge;
    # cancel_all_deferred is now a stable-API no-op.
    assert delivery.cancel_all_deferred() == 0


# --- route_send urgent flag --------------------------------------------------
# route_send is HTTP-bridged, but the urgent branch carries real logic (it must
# pass urgent ON the same /api/post call — not a second /api/interrupt — and
# interpret was_in_flight). Mock the single HTTP boundary (_http_post) so we
# test that logic, not urllib.


def _fake_post(monkeypatch, response: dict):
    calls = []

    def fake(path, payload):
        calls.append((path, payload))
        return True, json.dumps(response), 200

    monkeypatch.setattr(delivery, "_http_post", fake)
    return calls


def test_route_send_default_queues_and_passes_urgent_false(monkeypatch):
    calls = _fake_post(monkeypatch, {"ok": True, "was_in_flight": None})
    ok, status = delivery.route_send(from_role="tl", to="swe", content="hi")
    assert ok
    # Single /api/post call carrying urgent=False; no second endpoint.
    assert [p for p, _ in calls] == ["/api/post"]
    assert calls[0][1] == {"from": "tl", "to": "swe", "body": "hi", "urgent": False}
    assert "Delivered to swe" in status and "halted" not in status


def test_route_send_urgent_is_single_post_and_reports_halt(monkeypatch):
    calls = _fake_post(monkeypatch, {"ok": True, "was_in_flight": True})
    ok, status = delivery.route_send(
        from_role="statistician", to="swe", content="STOP — wrong option", urgent=True
    )
    assert ok
    # urgent is one POST to /api/post (server composes the halt), NOT a
    # separate /api/interrupt call — that was the UI bug we are not repeating.
    assert [p for p, _ in calls] == ["/api/post"]
    assert calls[0][1]["urgent"] is True
    assert "halted its in-flight turn" in status


def test_route_send_urgent_recipient_idle_reports_queued(monkeypatch):
    _fake_post(monkeypatch, {"ok": True, "was_in_flight": False})
    ok, status = delivery.route_send(from_role="tl", to="swe", content="x", urgent=True)
    assert ok
    assert "was idle" in status
