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
"""The team, end to end, with everything real but the model.

A one-role roster is booted through ``serve._build_all_agents`` exactly as
``agent-team serve`` boots it: sagent's ``Agent`` on a ``JsonlJournal``, the
plugin's peer-messaging MCP server as a real subprocess over stdio, and the
HTTP API on a real loopback port (the MCP server POSTs to it). Only the model
is sagent's scripted one, so no ``claude`` binary, credentials or network are
needed.
"""

import asyncio
import dataclasses
import json
import shutil
import socket
import urllib.error
import urllib.request

from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

import pytest
import uvicorn

from sagent.model.scripted import ScriptedProvider, call, reply
from sagent.runtime.types import AgentIdle
from sagent.session.types import AgentToAgentMessage, ToolResult, UserMessage

from agent_team import serve
from agent_team.mcp_sagent import delivery
from agent_team.roles import common
from agent_team.runtime import trace_writer
from agent_team.team_profile import load_profile


# A session log exactly as sagent 0.1.20 wrote it (two turns: a user message
# and a peer's), which a deployment upgrading across the 0.2 port still holds.
_LEGACY_LOG = Path(__file__).parent / "data" / "sagent_0_1_20_session.jsonl"

_SEND = "mcp__sagent_chat__sagent_send"
_DEFER = "mcp__sagent_chat__sagent_defer"


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture
def team_env(tmp_path, monkeypatch):
    """Point every data path at ``tmp_path`` and the MCP server at a free port."""
    sessions = tmp_path / "sessions"
    main = tmp_path / "main.jsonl"
    port = _free_port()
    monkeypatch.setenv("SAGENT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SAGENT_HTTP_PORT", str(port))
    monkeypatch.setattr(delivery, "DATA_DIR", tmp_path)
    monkeypatch.setattr(delivery, "SESSIONS_DIR", sessions)
    monkeypatch.setattr(delivery, "MAIN_JSONL_PATH", main)
    monkeypatch.setattr(trace_writer, "_SESSIONS_DIR", sessions)
    monkeypatch.setattr(serve, "_MAIN_JSONL", main)
    monkeypatch.setattr(serve, "_SUPPRESS_FLAG", sessions / "_suppress_audit")
    profile = dataclasses.replace(
        load_profile(), roster=["tl"], sandbox_root=tmp_path / "sandbox"
    )
    return dataclasses.make_dataclass(
        "TeamEnv", ["profile", "port", "main", "sessions"]
    )(profile, port, main, sessions)


@asynccontextmanager
async def _team(env, monkeypatch, replies):
    """Boot the roster on a scripted model and serve its HTTP API."""
    provider = ScriptedProvider(replies=replies)
    monkeypatch.setattr(common, "build_provider", lambda: provider)
    async with AsyncExitStack() as stack:
        agents = await serve._build_all_agents(env.profile, stack)
        server = uvicorn.Server(
            uvicorn.Config(
                serve._build_http_app(agents),
                host="127.0.0.1",
                port=env.port,
                log_level="warning",
                access_log=False,
            )
        )
        http = asyncio.create_task(server.serve())
        while not server.started:
            await asyncio.sleep(0.01)
        try:
            yield agents, provider
        finally:
            server.should_exit = True
            await http


def _idle(agent) -> asyncio.Event:
    """An event set the next time ``agent`` has nothing left to do."""
    done = asyncio.Event()

    def watch(event):
        if isinstance(event, AgentIdle):
            done.set()
            agent.runtime.observers.remove(watch)

    agent.runtime.observers.append(watch)
    return done


async def _http(env, method, path, payload=None):
    def request():
        req = urllib.request.Request(
            f"http://127.0.0.1:{env.port}{path}",
            data=None if payload is None else json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    return await asyncio.to_thread(request)


def _audit(env):
    if not env.main.exists():
        return []
    return [
        (r["from"], r["to"], r["body"])
        for r in map(json.loads, env.main.read_text().splitlines())
    ]


def _texts(request):
    return [
        r.content
        for r in request.records
        if isinstance(r, (UserMessage, AgentToAgentMessage))
    ]


async def test_operator_message_is_answered_through_the_peer_messaging_server(
    team_env, monkeypatch
):
    replies = [
        reply(calls=[call(_SEND, "c1", to="user", content="pong")]),
        reply("sent"),
    ]
    async with _team(team_env, monkeypatch, replies) as (agents, provider):
        tl = agents["tl"]
        # The model is offered the role's tools and the three chat tools.
        assert {"Read", "Bash", _SEND, _DEFER} <= set(tl.tools)
        assert "Edit" not in tl.tools  # TL is read-only by scope

        idle = _idle(tl)
        status, body = await _http(
            team_env, "POST", "/api/post", {"to": "tl", "body": "ping"}
        )
        assert (status, body["ok"]) == (200, True)
        await asyncio.wait_for(idle.wait(), 30)

        # user → tl went to the audit log; tl's tool call went out of process
        # to the MCP server, which POSTed it back in as tl → user.
        assert _audit(team_env) == [
            ("user", ["tl"], "ping"),
            ("tl", ["user"], "pong"),
        ]
        assert _texts(provider.requests[0]) == ["ping"]
        results = [r for r in provider.requests[1].records if isinstance(r, ToolResult)]
        assert [(r.call_id, bool(r.is_error), r.text()) for r in results] == [
            ("c1", False, "Delivered to user.")
        ]

        # The trace the web UI reads: the fields it renders are there.
        trace = [
            json.loads(line)
            for line in (team_env.sessions / "tl.trace.jsonl").read_text().splitlines()
        ]
        inbound = next(e for e in trace if e["_event"] == "UserMessage")
        assert inbound["text"] == "ping"
        replied = [e for e in trace if e["_event"] == "ModelResponseComplete"]
        assert replied[0]["message"]["tool_calls"] == [
            {"name": _SEND, "args": {"to": "user", "content": "pong"}}
        ]
        assert replied[1]["message"]["text"] == "sent"

        status, body = await _http(team_env, "GET", "/api/agents")
        (row,) = body["agents"]
        assert (status, row["role"], row["alive"]) == (200, "tl", True)
        assert row["model_id"] == "scripted"
        assert row["last_result"]["ok"] is True
        assert row["pending"] == 0
        status, body = await _http(team_env, "GET", "/api/roles")
        assert body == {"roles": ["user", "tl", "system"]}


async def test_send_to_unknown_peer_is_an_error_the_model_sees(team_env, monkeypatch):
    replies = [
        reply(calls=[call(_SEND, "c1", to="nobody", content="hi")]),
        reply("gave up"),
    ]
    async with _team(team_env, monkeypatch, replies) as (agents, provider):
        idle = _idle(agents["tl"])
        await _http(team_env, "POST", "/api/post", {"to": "tl", "body": "go"})
        await asyncio.wait_for(idle.wait(), 30)
        (result,) = [
            r for r in provider.requests[1].records if isinstance(r, ToolResult)
        ]
        assert result.is_error is True
        assert "unknown target 'nobody'" in result.text()
        assert _audit(team_env) == [("user", ["tl"], "go")]


async def test_peer_message_is_delivered_as_from_that_peer(team_env, monkeypatch):
    async with _team(team_env, monkeypatch, [reply("ack")]) as (agents, provider):
        idle = _idle(agents["tl"])
        status, body = await _http(
            team_env,
            "POST",
            "/api/post",
            {"from": "swe", "to": "tl", "body": "PR is up", "urgent": True},
        )
        assert status == 200
        # urgent is accepted but never preempts.
        assert body["urgent_halt_disabled"] is True and body["was_in_flight"] is None
        await asyncio.wait_for(idle.wait(), 30)
        (message,) = [
            r
            for r in provider.requests[0].records
            if isinstance(r, AgentToAgentMessage)
        ]
        assert (message.sender, message.content) == ("swe", "PR is up")
        assert _audit(team_env) == [("swe", ["tl"], "PR is up")]


async def test_post_rejects_what_it_cannot_route(team_env, monkeypatch):
    async with _team(team_env, monkeypatch, []) as (_agents, provider):
        for payload, code in [
            ({"to": "nobody", "body": "x"}, 404),
            ({"to": "user", "body": "x"}, 400),  # user → user
            ({"to": "tl"}, 400),  # no body
        ]:
            status, _ = await _http(team_env, "POST", "/api/post", payload)
            assert status == code, payload
        # A message addressed to the operator has no agent to wake: audit only.
        status, body = await _http(
            team_env, "POST", "/api/post", {"from": "tl", "to": "user", "body": "fyi"}
        )
        assert (status, body["ok"]) == (200, True)
        assert _audit(team_env) == [("tl", ["user"], "fyi")]
        assert provider.requests == []


async def test_self_defer_wakes_the_agent_after_the_delay(team_env, monkeypatch):
    replies = [
        reply(calls=[call(_DEFER, "c1", delay_s=1, body="check CI")]),
        reply("waiting"),
        reply("checked"),
    ]
    async with _team(team_env, monkeypatch, replies) as (agents, provider):
        tl = agents["tl"]
        idle = _idle(tl)
        await _http(team_env, "POST", "/api/post", {"to": "tl", "body": "watch CI"})
        await asyncio.wait_for(idle.wait(), 30)
        assert len(provider.requests) == 2

        woke = _idle(tl)
        await asyncio.wait_for(woke.wait(), 30)
        wake = [
            r
            for r in provider.requests[2].records
            if isinstance(r, AgentToAgentMessage)
        ]
        assert [(m.sender, m.content) for m in wake] == [
            ("tl", "[defer +1s] check CI")
        ]
        assert ("tl", ["tl"], "[defer +1s scheduled] check CI") in _audit(team_env)

        for payload, code in [
            ({"to": "tl", "body": "x", "delay_s": 0}, 400),
            ({"to": "user", "body": "x", "delay_s": 5}, 404),
        ]:
            status, _ = await _http(team_env, "POST", "/api/defer", payload)
            assert status == code, payload


async def test_warmup_is_silent_and_reports_ready(team_env, monkeypatch):
    replies = [
        reply(calls=[call("mcp__sagent_chat__sagent_self", "c1", status="ready")]),
        reply("ok"),
    ]
    async with _team(team_env, monkeypatch, replies) as (agents, provider):
        assert await serve._warmup_agents(agents, timeout_s=30) == {"tl": True}
        assert "BOOTSTRAP PROBE" in _texts(provider.requests[0])[0]
        (result,) = [
            r for r in provider.requests[1].records if isinstance(r, ToolResult)
        ]
        assert result.text() == "status acknowledged: 'ready'"
        assert _audit(team_env) == []  # nothing reached the audit log
        assert not serve._SUPPRESS_FLAG.exists()  # sentinel lifted again


async def test_restart_resumes_the_session_and_slims_it(team_env, monkeypatch):
    monkeypatch.setattr(serve, "_RESUME_MODE", "slim")
    monkeypatch.setattr(serve, "_RESUME_KEEP_RECORDS", 2)
    journal = team_env.sessions / "tl.sagent" / "session.jsonl"

    async with _team(
        team_env, monkeypatch, [reply("one"), reply("two"), reply("three")]
    ) as (agents, _provider):
        for body in ("first", "second", "third"):
            idle = _idle(agents["tl"])
            await _http(team_env, "POST", "/api/post", {"to": "tl", "body": body})
            await asyncio.wait_for(idle.wait(), 30)
    written = journal.read_bytes()
    assert written.count(b"\n") > 6

    # A restart: the agent resumes only its recent tail, and the file is as it was.
    async with _team(team_env, monkeypatch, [reply("four")]) as (agents, provider):
        assert journal.read_bytes().startswith(written)
        idle = _idle(agents["tl"])
        await _http(team_env, "POST", "/api/post", {"to": "tl", "body": "fourth"})
        await asyncio.wait_for(idle.wait(), 30)
        assert _texts(provider.requests[0]) == ["third", "fourth"]

    # A full resume still finds everything: slimming never rewrote the journal.
    monkeypatch.setattr(serve, "_RESUME_MODE", "full")
    async with _team(team_env, monkeypatch, [reply("five")]) as (agents, provider):
        idle = _idle(agents["tl"])
        await _http(team_env, "POST", "/api/post", {"to": "tl", "body": "fifth"})
        await asyncio.wait_for(idle.wait(), 30)
        assert _texts(provider.requests[0]) == [
            "first",
            "second",
            "third",
            "fourth",
            "fifth",
        ]


async def test_a_pre_0_2_session_log_is_upgraded_and_resumed(team_env, monkeypatch):
    monkeypatch.setattr(serve, "_RESUME_MODE", "full")
    journal = team_env.sessions / "tl.sagent" / "session.jsonl"
    journal.parent.mkdir(parents=True)
    shutil.copy(_LEGACY_LOG, journal)

    async with _team(team_env, monkeypatch, [reply("caught up")]) as (agents, provider):
        idle = _idle(agents["tl"])
        await _http(team_env, "POST", "/api/post", {"to": "tl", "body": "status?"})
        await asyncio.wait_for(idle.wait(), 30)
        # The old conversation is what the model is told, peer attribution intact.
        seen = [
            (type(r).__name__, getattr(r, "sender", None), r.content)
            for r in provider.requests[0].records
            if isinstance(r, (UserMessage, AgentToAgentMessage))
        ]
        assert seen == [
            ("UserMessage", None, "first question"),
            ("AgentToAgentMessage", "swe", "PR 7 is up"),
            ("UserMessage", None, "status?"),
        ]
        replies = [
            r.content
            for r in provider.requests[0].records
            if type(r).__name__ == "AssistantMessage"
        ]
        assert replies == ["reply 1", "reply 2"]
    # sagent keeps the original log, whole, beside the journal it rewrote.
    kept = journal.with_name("session.legacy.jsonl")
    assert kept.read_bytes() == _LEGACY_LOG.read_bytes()


async def test_an_unusable_pre_0_2_log_is_moved_aside_not_fatal(team_env, monkeypatch):
    journal = team_env.sessions / "tl.sagent" / "session.jsonl"
    journal.parent.mkdir(parents=True)
    # Only the old log's meta line: nothing was said, so nothing to resume.
    journal.write_text(_LEGACY_LOG.read_text().splitlines()[0] + "\n")

    async with _team(team_env, monkeypatch, [reply("hello")]) as (agents, provider):
        idle = _idle(agents["tl"])
        await _http(team_env, "POST", "/api/post", {"to": "tl", "body": "hi"})
        await asyncio.wait_for(idle.wait(), 30)
        assert _texts(provider.requests[0]) == ["hi"]  # a fresh start
    (aside,) = journal.parent.glob("session.unreadable-*.jsonl")
    assert aside.read_text().count("\n") == 1
    assert b"py/object" in journal.read_bytes()  # today's journal now


async def test_restart_endpoint_clears_and_reanchors(team_env, monkeypatch):
    replies = [reply("working on the fix"), reply("asking TL")]
    async with _team(team_env, monkeypatch, replies) as (agents, provider):
        tl = agents["tl"]
        idle = _idle(tl)
        await _http(team_env, "POST", "/api/post", {"to": "tl", "body": "fix bug 7"})
        await asyncio.wait_for(idle.wait(), 30)

        idle = _idle(tl)
        status, body = await _http(
            team_env, "POST", "/api/restart", {"role": "tl", "mode": "reanchor"}
        )
        assert (status, body["ok"]) == (200, True), body
        assert "history cleared" in body["output"]
        await asyncio.wait_for(idle.wait(), 30)

        # The cleared agent sees only the re-anchor prompt, which carries the
        # breadcrumb of what it was doing.
        (prompt,) = _texts(provider.requests[1])
        assert prompt.startswith("RESTART NOTICE.")
        assert "[inbound] fix bug 7" in prompt
        assert "[tl] working on the fix" in prompt

        status, body = await _http(
            team_env, "POST", "/api/restart", {"role": "tl", "mode": "bogus"}
        )
        assert status == 400
        status, body = await _http(team_env, "POST", "/api/restart", {"role": "x"})
        assert status == 404


async def test_interrupt_on_an_idle_agent_halts_nothing(team_env, monkeypatch):
    async with _team(team_env, monkeypatch, []) as (_agents, _provider):
        status, body = await _http(team_env, "POST", "/api/interrupt", {"role": "tl"})
        assert (status, body["ok"], body["was_in_flight"]) == (200, True, False)
        status, _ = await _http(team_env, "POST", "/api/interrupt", {"role": "x"})
        assert status == 404


async def test_boot_fails_loudly_when_peer_messaging_cannot_connect(
    team_env, monkeypatch
):
    """An agent that could message no one must not come up looking healthy."""
    monkeypatch.setattr(
        common,
        "_sagent_mcp_server_entry",
        lambda role: {"command": "false", "args": [], "env": {}},
    )
    monkeypatch.setattr(common, "build_provider", ScriptedProvider)
    async with AsyncExitStack() as stack:
        with pytest.raises(RuntimeError, match="peer-messaging MCP server for 'tl'"):
            await serve._build_all_agents(team_env.profile, stack)


async def test_unknown_roster_role_is_rejected_before_anything_starts(
    team_env, monkeypatch
):
    profile = dataclasses.replace(team_env.profile, roster=["tl", "astronaut"])
    async with AsyncExitStack() as stack:
        with pytest.raises(ValueError, match="unknown role 'astronaut'"):
            await serve._build_all_agents(profile, stack)
