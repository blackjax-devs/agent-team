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
"""Consumer-side guard for what ``build_agent`` hands sagent.

``tests/sagent_seams_test.py`` checks *sagent's* side (does it still accept
what we pass). This checks *our* side: that ``build_agent`` still PASSES the
load-bearing pieces — the peer-messaging tools, the journal a restart resumes
from, the compactor, the role's own prompt. The model is sagent's scripted
one, so it needs no credentials, no ``claude`` binary, and no network.
"""

import pytest

from sagent.agent.compactor import SummaryCompactor, summary_tokens
from sagent.model.scripted import ScriptedProvider, reply
from sagent.session.jsonl import JsonlJournal
from sagent.session.types import UncategorizedToolResult

from agent_team.roles import common
from agent_team import serve


_CHAT_TOOLS = (
    "mcp__sagent_chat__sagent_send",
    "mcp__sagent_chat__sagent_defer",
    "mcp__sagent_chat__sagent_self",
)


class _Tool:
    """A stand-in tool: only its name matters to ``build_agent``."""

    description = "stand-in"
    schema = {"type": "object", "properties": {}}
    display = None

    def __init__(self, name):
        self.name = name

    def summary(self, arguments):
        return self.name

    def serialize_key(self, context, arguments):
        return None

    async def call(self, context, arguments):
        return UncategorizedToolResult(call_id="", content="ok")


def _build(provider, **kwargs):
    return common.build_agent(
        role_name="tl",
        tools=[_Tool("Read")],
        model_id="claude-opus-4-8",
        system="You are TL.",
        peers=["tl", "swe"],
        provider=provider,
        **kwargs,
    )


def test_build_agent_offers_role_tools_then_peer_messaging():
    agent = _build(
        ScriptedProvider(), chat_tools=[_Tool(name) for name in _CHAT_TOOLS]
    )
    # The peer-messaging tools are load-bearing: without them an agent can
    # reach no one.
    assert list(agent.tools) == ["Read", *_CHAT_TOOLS]
    assert agent.runtime.tool_context.label == "tl"


def test_build_agent_prompt_is_the_role_prompt_plus_reminders(tmp_path, monkeypatch):
    # An AGENTS.md of the host's must not join a role's prompt.
    (tmp_path / "AGENTS.md").write_text("HOST DIRECTIVE")
    monkeypatch.chdir(tmp_path)
    agent = _build(ScriptedProvider())
    system = "\n\n".join(agent.runtime.system)
    assert system.startswith("You are TL.")
    assert "mcp__sagent_chat__sagent_send" in system
    assert "`tl`, `swe`, `user`" in system
    assert common.HEAVY_BG_REMINDER in system
    assert "HOST DIRECTIVE" not in system


def test_build_agent_requires_a_prompt():
    with pytest.raises(ValueError, match="'system' or 'role_md_path'"):
        common.build_agent(
            role_name="tl", tools=[], model_id="x", provider=ScriptedProvider()
        )


def test_build_agent_passes_round_and_spend_caps():
    agent = _build(ScriptedProvider(), max_tool_call_rounds=7, max_budget_usd=1.25)
    assert agent.max_rounds == 7
    assert agent.max_spend_usd == 1.25


def test_build_agent_wires_the_compactor(monkeypatch):
    monkeypatch.delenv("AGENT_TEAM_COMPACT_TRIGGER", raising=False)
    agent = _build(ScriptedProvider())
    compactor = agent.session.compactor
    assert isinstance(compactor, SummaryCompactor)
    # sagent asks it before each model call, and compacts on demand through it.
    assert agent.runtime.compactor is compactor
    assert agent.runtime.compact_due == compactor.should_run


@pytest.mark.parametrize(
    ("trigger", "share"),
    [(None, 0.80), ("0.5", 0.5), ("0.95", 0.95)],
)
def test_compaction_fires_at_the_configured_share_of_the_window(
    monkeypatch, trigger, share
):
    """``AGENT_TEAM_COMPACT_TRIGGER`` (default 0.80) sets when an agent compacts."""
    if trigger is None:
        monkeypatch.delenv("AGENT_TEAM_COMPACT_TRIGGER", raising=False)
    else:
        monkeypatch.setenv("AGENT_TEAM_COMPACT_TRIGGER", trigger)
    agent = _build(ScriptedProvider())
    compactor = agent.session.compactor
    limits = agent.model.capability.limits
    window = min(limits.max_request_tokens, agent.model.settings.context_size.tokens)
    fixed = compactor._system_tokens() + compactor._prompt_tokens()
    usable = window - fixed
    # Due once the body holds ``share`` of what the window leaves it; sagent's
    # own reserve (5-10%) is the floor, which a trigger above ~0.9 runs into.
    expected = min(share, 1 / (1 + summary_tokens(usable) / usable))
    assert compactor.trigger_tokens() == pytest.approx(usable * expected, rel=0.02)


@pytest.mark.parametrize("trigger", ["0", "1", "1.5", "-0.2"])
def test_out_of_range_trigger_falls_back_to_sagents_reserve(trigger):
    assert common._summary_room(1000.0, trigger=float(trigger)) == summary_tokens(
        1000.0
    )


async def test_build_agent_resumes_from_its_journal(tmp_path):
    """The journal ``build_agent`` is given is what a restart resumes from."""
    path = tmp_path / "tl.sagent" / "session.jsonl"
    first = ScriptedProvider(replies=[reply("noted")])
    async with JsonlJournal(path) as journal, _build(first, journal=journal) as agent:
        assert await agent.send("remember 42") == "noted"

    second = ScriptedProvider(replies=[reply("42")])
    async with JsonlJournal(path) as journal, _build(second, journal=journal) as agent:
        assert await agent.send("what was it?") == "42"
    said = [getattr(r, "content", None) for r in second.requests[0].records]
    assert said[:3] == ["remember 42", "noted", "what was it?"] or (
        "remember 42" in said and "noted" in said
    )


_ROLE_TOOLS = {
    "tl": ["Read", "Grep", "Glob", "Bash", "WebSearch", "WebFetch"],
    "swe": ["Read", "Edit", "Write", "Bash", "Grep", "Glob"],
    "junior-swe": ["Read", "Edit", "Write", "Bash", "Grep", "Glob"],
    "statistician": [
        "Read",
        "Grep",
        "Glob",
        "Bash",
        "WebSearch",
        "WebFetch",
        "Write",
        "Edit",
    ],
    "tech-writer": [
        "Read",
        "Grep",
        "Glob",
        "Edit",
        "Write",
        "Bash",
        "WebSearch",
        "WebFetch",
    ],
}


@pytest.mark.parametrize("role", sorted(_ROLE_TOOLS))
def test_every_role_factory_builds_its_agent(tmp_path, role):
    import dataclasses

    from agent_team import sandboxed_tools
    from agent_team.team_profile import load_profile

    profile = dataclasses.replace(load_profile(), sandbox_root=tmp_path / "sandbox")
    chat = [_Tool(name) for name in _CHAT_TOOLS]
    agent = serve._role_builders()[role](
        profile, provider=ScriptedProvider(), chat_tools=chat
    )
    names = list(agent.tools)
    assert names[-3:] == list(_CHAT_TOOLS)
    assert names[:-3] == _ROLE_TOOLS[role]
    assert agent.runtime.tool_context.label == role
    assert "\n\n".join(agent.runtime.system).count("## Peer messaging") == 1
    if role == "statistician":
        # Its Write/Edit are the sandboxed ones, confined to the profile's root.
        for name in ("Write", "Edit"):
            tool = agent.tools[name]
            assert isinstance(
                tool, (sandboxed_tools.SandboxedWrite, sandboxed_tools.SandboxedEdit)
            )
            assert tool.sandbox_root == (tmp_path / "sandbox").resolve()


def test_session_journal_path_is_per_role_under_the_data_dir(tmp_path, monkeypatch):
    from agent_team.mcp_sagent import delivery

    monkeypatch.setattr(delivery, "SESSIONS_DIR", tmp_path / "sessions")
    path = common.journal_path_for("swe")
    assert path == tmp_path / "sessions" / "swe.sagent" / "session.jsonl"
    assert path.parent.is_dir()


def test_chat_server_is_the_plugins_mcp_server_for_the_role(monkeypatch):
    monkeypatch.setenv("SAGENT_HTTP_PORT", "9123")
    monkeypatch.setenv("SAGENT_DATA_DIR", "/data/team")
    server = common.chat_server("swe")
    # sagent names each tool ``mcp__<server>__<tool>``: the name is the prefix
    # the role prompts teach.
    assert server.name == "sagent_chat"
    assert server.transport == "stdio"
    assert server.parameters.args == ["-m", "agent_team.mcp_sagent.server"]
    assert server.parameters.env == {
        "SAGENT_ROLE": "swe",
        "SAGENT_HTTP_URL": "http://127.0.0.1:9123",
        "SAGENT_DATA_DIR": "/data/team",
    }


@pytest.mark.parametrize(("timeout", "expected"), [(None, 25.0), ("41.5", 41.5)])
def test_chat_session_passes_the_mcp_connect_timeout(monkeypatch, timeout, expected):
    import sagent.tools.mcp

    if timeout is None:
        monkeypatch.delenv("AGENT_TEAM_MCP_CONNECT_TIMEOUT_SEC", raising=False)
    else:
        monkeypatch.setenv("AGENT_TEAM_MCP_CONNECT_TIMEOUT_SEC", timeout)
    seen = {}

    def session(servers, *, connect_timeout_sec):
        seen["servers"] = [s.name for s in servers]
        seen["timeout"] = connect_timeout_sec

    monkeypatch.setattr(sagent.tools.mcp, "McpSession", session)
    common.chat_session("tl")
    assert seen == {"servers": ["sagent_chat"], "timeout": expected}


async def test_chat_session_gives_the_tools_the_prompts_teach():
    """The real server, over real stdio: the names the model is told to call."""
    async with common.chat_session("tl") as chat:
        assert chat.status() == {"sagent_chat": "connected, 3 tools"}
        assert sorted(tool.name for tool in chat.tools()) == sorted(_CHAT_TOOLS)


async def test_startup_reaches_agent_build(monkeypatch):
    class ReachedBuild(Exception):
        pass

    def stop(profile, stack):
        raise ReachedBuild

    monkeypatch.setattr(serve, "_build_all_agents", stop)
    # _amain sets this variable before it constructs agents; restore it after
    # the smoke so later tests don't inherit a changed server port.
    monkeypatch.setenv("SAGENT_HTTP_PORT", "8767")
    with pytest.raises(ReachedBuild):
        await serve._amain("127.0.0.1", 9876)
