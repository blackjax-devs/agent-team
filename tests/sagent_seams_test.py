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
"""The sagent surface agent-team is built on — the drift guard.

``roles/common.py`` and ``serve.py`` pass these names to sagent. A sagent that
renames or drops one fails here with the seam named, instead of somewhere
inside a boot. CI runs this against the locked sagent; ``nightly-sagent-head``
runs it against sagent ``main`` to warn before a release lands.

Signatures only: no credentials, no ``claude`` binary, no network.
"""

import dataclasses
import inspect

import pytest


def _parameters(obj) -> set[str]:
    return set(inspect.signature(obj).parameters)


def _fields(cls) -> set[str]:
    return {f.name for f in dataclasses.fields(cls)}


def _missing(have: set[str], need: set[str]) -> list[str]:
    return sorted(need - have)


def test_agent_accepts_what_build_agent_passes():
    from sagent.agent.agent import Agent

    need = {
        "model",
        "system_prompt",
        "tools",
        "journal",
        "compactor",
        "tool_context",
        "max_rounds",
        "max_spend_usd",
        "agents_md",
    }
    assert not _missing(_parameters(Agent), need)
    # what serve.py drives a live agent through
    for name in ("aopen", "aclose", "running", "model"):
        assert hasattr(Agent, name), name


def test_provider_serves_models_by_name():
    """``build_provider()`` is ``AnthropicCLI()``; a role's model is ``provider[id]``."""
    from sagent.providers.anthropic.cli import AnthropicCLI

    provider = AnthropicCLI()  # no I/O until a model streams
    for model_id in (
        "claude-opus-4-8",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
    ):
        model = provider[model_id]
        assert model.capability.wire_model_id == model_id
        assert callable(model.stream)


def test_mcp_session_mounts_a_stdio_server_as_tools():
    """Peer messaging reaches the model through ``McpSession([McpServer(...)])``."""
    from mcp import StdioServerParameters
    from sagent.tools.mcp import McpServer, McpSession

    assert not _missing(_fields(McpServer), {"name", "parameters", "description"})
    assert not _missing(_parameters(McpSession), {"servers", "connect_timeout_sec"})
    for name in ("tools", "status", "aopen", "aclose"):
        assert hasattr(McpSession, name), name
    assert not _missing(
        set(StdioServerParameters.model_fields), {"command", "args", "env"}
    )


def test_compactor_takes_a_summary_room():
    """``AGENT_TEAM_COMPACT_TRIGGER`` is applied through ``summary_tokens``."""
    from sagent.agent.compactor import SummaryCompactor, summary_tokens

    assert not _missing(_parameters(SummaryCompactor), {"agent", "summary_tokens"})
    assert summary_tokens(100_000.0) > 0
    for name in ("run", "should_run", "trigger_tokens"):
        assert hasattr(SummaryCompactor, name), name


def test_journal_and_legacy_upgrade():
    """A role resumes from ``JsonlJournal``; a pre-0.2 log is upgraded first."""
    from sagent.agent.types import Journal
    from sagent.session import legacy_journal
    from sagent.session.jsonl import JsonlJournal

    for name in ("path", "append", "history", "flush", "__iter__", "__len__"):
        assert hasattr(Journal, name), name
        assert hasattr(JsonlJournal, name), name
    assert callable(legacy_journal.is_legacy) and callable(legacy_journal.upgrade)


def test_runtime_surface_serve_drives():
    from sagent.runtime.runtime import GatedDeque, Runtime

    for name in ("push", "halt", "context", "jobs", "publish"):
        assert hasattr(Runtime, name), name
    for name in ("push_back", "drain_nowait", "empty"):
        assert hasattr(GatedDeque, name), name


def test_messages_serve_pushes():
    from sagent.runtime import types as runtime_types
    from sagent.session.types import (
        AgentToAgentMessage,
        ContextClear,
        TokenUsage,
        UserMessage,
    )

    assert not _missing(_fields(UserMessage), {"content", "timestamp"})
    assert not _missing(
        _fields(AgentToAgentMessage), {"content", "sender", "timestamp"}
    )
    assert not _missing(
        _fields(runtime_types.AgentSendQueuedMessage), {"sender", "text"}
    )
    assert not _missing(_fields(ContextClear), {"summary", "history", "extra"})
    assert not _missing(_fields(TokenUsage), {"timestamp", "info"})
    for name in (
        "AgentIdle",
        "ModelResponseComplete",
        "ModelCallStarted",
        "UserQueuedMessage",
        "UserDeferredMessage",
        "AgentSendDeferredMessage",
    ):
        assert hasattr(runtime_types, name), name


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("sagent.tools.read", "Read"),
        ("sagent.tools.grep", "Grep"),
        ("sagent.tools.glob_tool", "Glob"),
        ("sagent.tools.bash", "Bash"),
        ("sagent.tools.edit", "Edit"),
        ("sagent.tools.write", "Write"),
        ("sagent.tools.web_search", "WebSearch"),
        ("sagent.tools.web_fetch", "WebFetch"),
    ],
)
def test_role_tools_build_without_arguments(module, name):
    import importlib

    tool = getattr(importlib.import_module(module), name)()
    assert tool.name == name
    assert callable(tool.call)


def test_tool_context_names_the_agent_and_resolves_paths():
    from sagent.tools.types import ToolContext

    context = ToolContext(label="tl")
    assert context.label == "tl"
    assert context.path("/tmp/a.py").is_absolute()
    assert hasattr(context.files, "saw")
