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
"""Shared building blocks for per-role Agent factories.

Conventions:

- All roles use sagent's ``AnthropicCLI`` provider (Claude subscription via
  the ``claude`` CLI's login; no API key consulted).
- Peer messaging happens via the plugin's MCP server (``mcp_sagent.server``)
  rather than sagent's ``AgentSend``. ``serve.py`` connects one server per
  role through ``sagent.tools.mcp.McpSession`` and hands its tools to the
  agent, so ``sagent_send`` reaches the model as
  ``mcp__sagent_chat__sagent_send`` on all three models
  (haiku/sonnet/opus) — see README § "Episode 3" for the probe result
  that motivated the design.
- Heavy background commands must be wrapped in ``systemd-run --user
  --scope --quiet --collect -- bash -c '<cmd>'`` to keep an OOM
  contained in a sibling cgroup instead of cascading through the
  pane. The wrap reminder is appended to every system prompt at
  build time so it survives context compaction (a structural fix
  for the chat/-era SWE-OOM-after-compaction failure mode).
"""

from __future__ import annotations

import os
import sys

from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sagent.agent.compactor import SummaryCompactor, summary_tokens
from sagent.tools.types import Tool, ToolContext


if TYPE_CHECKING:
    from sagent.agent.types import Journal
    from sagent.providers.types import Provider
    from sagent.tools.mcp import McpServer, McpSession


# Name of the peer-messaging MCP server: the ``<server>`` in every
# ``mcp__<server>__<tool>`` name the role prompts teach.
CHAT_SERVER_NAME = "sagent_chat"

# Optional model-id fallbacks. The authoritative role→model assignment now
# comes from the loaded profile's ``[models]`` table (``profile.models``);
# these constants remain only as convenience defaults for callers that build
# an agent outside a profile context.
MODEL_OPUS = "claude-opus-4-8"
MODEL_FABLE = "claude-fable-5"
MODEL_SONNET = "claude-sonnet-4-6"
MODEL_HAIKU = "claude-haiku-4-5"

HEAVY_BG_REMINDER = """\
Operating reminder: if a job is expected to be heavy (memory or CPU), \
run it with `systemd-run --user --scope --quiet --collect \
--unit=<literal> -- bash -c '<cmd>'` so an OOM stays in a sibling \
cgroup and your worker survives. Do not put $(...) or $VAR inside any \
argument starting with `-`; the Bash tool's permission system rejects \
runtime-determined content there."""

def _peer_messaging(peers: Sequence[str] | None = None) -> str:
    """Build the peer-messaging reminder, listing the live roster as peers.

    ``peers`` is the active roster (role labels). ``user`` is always appended
    as an addressable target. When ``peers`` is ``None`` the framework falls
    back to the historical default roster so the prompt is never empty.
    """
    if peers is None:
        peers = ["tl", "swe", "junior-swe", "statistician", "tech-writer"]
    known = list(peers)
    if "user" not in known:
        known.append("user")
    known_str = ", ".join(f"`{p}`" for p in known)
    # Plain ``str.replace`` (not ``str.format``) so the template's literal
    # ``{to, content}`` / ``{delay_s, body}`` braces survive untouched.
    return PEER_MESSAGING_TEMPLATE.replace("{{known}}", known_str)


PEER_MESSAGING_TEMPLATE = """\
## Peer messaging — REQUIRED structural tool calls

This is a multi-agent chat channel. To deliver a message to another \
agent or to the user, you MUST call **`mcp__sagent_chat__sagent_send`** \
with `{to, content}`. Known peers: {{known}}.

**Your assistant text is NOT delivered to anyone.** It is logged \
only to your own trace and is invisible to peers and the user. \
Writing a message in your text content blocks instead of calling \
the tool means the recipient never receives the message. There is \
no `@mention`-based prose parser. There is no DM-default fallback. \
The structured tool call IS the only routing.

**Common failure mode (do not do this):** writing text like \
"I'll send the summary to @user" or "Let me send a message to @swe" \
without actually calling `sagent_send`. Describing the call does \
not perform it. If you intend to message anyone, call the tool — \
do not write about it in prose.

**Answering a question is also a message (the miss that actually \
happens):** when the operator or a peer sends you something — a \
question, a `status report?`, a request — your REPLY is a message \
too, and it reaches them ONLY via \
`sagent_send(to=<whoever asked>, content=...)`. Ending the turn with \
your answer as ordinary prose feels like a normal reply, but it \
delivers nothing: the asker waits while you sit idle, "answered" only \
in your private trace. A ping in your inbox demands a `sagent_send` \
back out, addressed to the sender. This is not optional politeness — \
an unsent answer is a dropped message.

**Wrong-tool trap (do not do this):** if a built-in `SendMessage` \
tool (Claude Teams / Agent-SDK) appears in your catalog, do NOT use \
it to reach a peer. It routes to a private team registry that is \
EMPTY here, so the message is silently dropped — the recipient never \
receives it even though the call looks like it succeeded. Only \
`mcp__sagent_chat__sagent_send` actually delivers.

**Common success pattern:** call `sagent_send` first (one or more \
times if you need to message multiple peers, one call each), then \
optionally end the turn with a brief text content block describing \
what you sent so your own trace stays readable.

**Interrupting a peer — TEMPORARILY DISABLED.** `sagent_send`'s \
`urgent` flag is currently a no-op: it QUEUES exactly like a normal \
send and does NOT preempt the recipient (the halt mechanism wedges \
them; disabled pending upstream review). To redirect a peer, just \
send normally — they act on it on their next turn. Do not rely on \
mid-turn interruption.

**End-of-turn check (do this every turn):** before you end a turn, \
ask — did I produce a status, an answer, a result, or a question that \
a person or peer needs to see? If yes, I must already have called \
`sagent_send` to deliver it. A turn that ends with undelivered prose \
addressed to someone is a bug in your own behaviour, not a reply.

### Self-defer (CI waits, polling, "check back later")

To schedule a wake-up for YOURSELF, call \
**`mcp__sagent_chat__sagent_defer`** with `{delay_s, body}`. After \
scheduling, end your turn — the runtime pushes the body back into \
your inbox after the delay and you process it in a fresh turn. \
**Do NOT use `bash sleep N`** — that blocks your entire turn for N \
seconds and makes you uninterruptible. **Do NOT just write "I'll \
check back in N minutes" in text** — that does not schedule \
anything; the recipient (yourself) never receives a wake-up.

**Waiting on a background job?** Do BOTH, then end the turn: (1) \
`sagent_send` a brief "in progress / waiting on X" status now so the \
team isn't blind, and (2) `sagent_defer` to re-check when it should be \
done. Never go idle silently intending to "report once it finishes" — \
nothing will wake you to do so, and your work strands invisibly."""


def compose_system_prompt(
    body: str, *, peers: Sequence[str] | None = None
) -> str:
    """Append the standing reminders to an already-rendered role prompt body.

    ``body`` is the role's system-prompt text with the ``{{workspace}}`` token
    already filled (see :func:`team_profile.render_prompt`). ``peers`` is the
    active roster, used to build the peer-messaging known-peers list.

    Two reminders, in order, appended (not prefixed) so the role-specific
    identity + scope text leads the prompt and the reminders sit at a
    stable tail location for visual confirmation:

    1. peer-messaging — how to address peers (via MCP tool, not prose).
       Reinjected every turn so compaction doesn't drop it.
    2. ``HEAVY_BG_REMINDER`` — systemd-run wrap rule for OOM containment;
       reinjected every turn for the same reason.
    """
    peer_block = _peer_messaging(peers)
    return f"{body.strip()}\n\n---\n\n{peer_block}\n\n---\n\n{HEAVY_BG_REMINDER}"


def load_system_prompt(
    role_md_path: Path, *, peers: Sequence[str] | None = None
) -> str:
    """Back-compat: read a role-prompt markdown file then compose reminders.

    Retained for callers that still pass a path. New callers should render the
    prompt via :func:`team_profile.render_prompt` and call
    :func:`compose_system_prompt` directly.
    """
    body = role_md_path.read_text(encoding="utf-8")
    return compose_system_prompt(body, peers=peers)


def build_provider() -> Provider:
    """Construct the AnthropicCLI provider every role's model comes from.

    One provider is shared across the roster: it owns the tool server each
    model's ``claude`` reaches its tools through and the ``claude`` process
    pool; per-agent state lives on the model ``provider[model_id]`` returns.
    The caller closes it (``async with`` / ``aclose``).
    """
    from sagent.providers.anthropic.cli import AnthropicCLI

    return AnthropicCLI()


def _sagent_mcp_server_entry(role: str) -> dict[str, Any]:
    """Per-role stdio entry for the peer-messaging MCP server.

    Spawns ``python -m agent_team.mcp_sagent.server`` with three env vars:

      - ``SAGENT_ROLE``: the calling agent's label, used by the MCP
        server to attribute outgoing peer messages.
      - ``SAGENT_HTTP_URL``: where to POST ``/api/post`` and
        ``/api/defer`` — i.e. ``serve.py``'s loopback URL. The MCP
        server runs in a SEPARATE Python process from ``serve.py``,
        so it cannot reach the live agents directly; HTTP is the
        only way in.
      - ``SAGENT_DATA_DIR``: where ``main.jsonl``, the sentinel,
        the debug log, and the per-role trace files live. Inherited
        from the parent ``serve.py`` env so audit/data files
        co-locate across the process tree.
    """
    from ..mcp_sagent.config_factory import SERVER_MODULE

    port = os.environ.get("SAGENT_HTTP_PORT", "8767")
    env_out = {
        "SAGENT_ROLE": role,
        "SAGENT_HTTP_URL": f"http://127.0.0.1:{port}",
    }
    # Propagate the data dir only if explicitly set — if absent, both
    # parent and child fall back to the launch cwd, which matches anyway.
    data_dir = os.environ.get("SAGENT_DATA_DIR")
    if data_dir:
        env_out["SAGENT_DATA_DIR"] = data_dir
    return {
        "command": sys.executable,
        "args": ["-m", SERVER_MODULE],
        "env": env_out,
    }


def chat_server(role: str) -> McpServer:
    """Return the peer-messaging MCP server for ``role``, as sagent configures one.

    The name must stay :data:`CHAT_SERVER_NAME`: sagent names each tool
    ``mcp__<server>__<tool>``, and the role prompts teach
    ``mcp__sagent_chat__sagent_send`` / ``__sagent_defer`` / ``__sagent_self``.
    """
    from mcp import StdioServerParameters
    from sagent.tools.mcp import McpServer

    entry = _sagent_mcp_server_entry(role)
    return McpServer(
        name=CHAT_SERVER_NAME,
        parameters=StdioServerParameters(
            command=entry["command"],
            args=entry["args"],
            env=entry["env"],
        ),
        description=f"agent-team peer messaging for {role}",
    )


def mcp_connect_timeout_sec() -> float:
    """How long the peer-messaging MCP server may take to connect.

    Env-tunable via ``AGENT_TEAM_MCP_CONNECT_TIMEOUT_SEC`` (default 25s): a
    loaded host can take a while to import the server.
    """
    return float(os.environ.get("AGENT_TEAM_MCP_CONNECT_TIMEOUT_SEC", "25"))


def chat_session(role: str) -> McpSession:
    """Return the (unopened) MCP session holding ``role``'s peer-messaging server.

    ``async with`` it, then pass ``session.tools()`` to :func:`build_agent` as
    ``chat_tools``: that is what gives the role ``sagent_send`` / ``sagent_defer``
    / ``sagent_self``. The session owns the server subprocess.
    """
    from sagent.tools.mcp import McpSession

    return McpSession(
        [chat_server(role)],
        connect_timeout_sec=mcp_connect_timeout_sec(),
    )


def compact_trigger() -> float:
    """Share of the context window at which an agent compacts.

    Env-tunable via ``AGENT_TEAM_COMPACT_TRIGGER`` (default 0.80). We fire
    EARLIER than sagent's own trigger: a lower trigger keeps the resumed
    context small, so the first-turn full-history re-feed after a restart stays
    fast — AND it makes agents compact while HEALTHY. A wedged subprocess
    CANNOT compact: observed 2026-06-19, swe's compaction failed 'subprocess
    stdout closed' because it had already wedged before ever reaching the
    trigger, so its 1.2 MB tape never shrank.
    """
    return float(os.environ.get("AGENT_TEAM_COMPACT_TRIGGER", "0.80"))


def _summary_room(body: float, *, trigger: float) -> float:
    """Room held back for a ``body``-token context's summary.

    ``SummaryCompactor`` compacts once ``body + room(body)`` no longer fits the
    window, so holding back ``body * (1 - trigger) / trigger`` makes it due at
    ``trigger`` of the window. Never less than sagent's own reserve.
    """
    reserve = summary_tokens(body)
    if not 0.0 < trigger < 1.0:
        return reserve
    return max(reserve, body * (1.0 - trigger) / trigger)


def compactor_factory():
    """Return what builds each agent's compactor.

    Without a compactor, the agent's context grows uncapped and the provider
    hits the context-window edge for heavy multi-PR work — observed in
    production as SWE's 3.4M cache_read_input_tokens over a ~5h session, with
    increasing ``aborted_streaming`` retryable errors and eventual stream-cuts.
    sagent owns compaction; claude's auto-compact is disabled by the provider.
    """
    return partial(
        SummaryCompactor,
        summary_tokens=partial(_summary_room, trigger=compact_trigger()),
    )


def _session_dir_for(role_name: str) -> Path:
    """Per-role sagent persistence dir (holds ``session.jsonl``).

    Lives under the data dir so ``serve.py`` restarts resume each role's
    session. Co-located with the per-role traces under
    ``<SAGENT_DATA_DIR>/sessions/``.
    """
    from ..mcp_sagent import delivery

    path = delivery.SESSIONS_DIR / f"{role_name}.sagent"
    path.mkdir(parents=True, exist_ok=True)
    return path


def journal_path_for(role_name: str) -> Path:
    """Where ``role_name``'s session journal is written."""
    return _session_dir_for(role_name) / "session.jsonl"


def build_agent(
    *,
    role_name: str,
    tools: Sequence[Tool],
    model_id: str,
    system: str | None = None,
    role_md_path: Path | None = None,
    peers: Sequence[str] | None = None,
    max_tool_call_rounds: int | None = None,
    max_budget_usd: float | None = None,
    provider: Provider | None = None,
    journal: Journal | None = None,
    chat_tools: Sequence[Tool] = (),
):
    """Construct a sagent Agent for a role with shared defaults baked in.

    Args:
        role_name: The role's label: the agent's name in its team, and the
            label peers use in ``sagent_send(to=...)``.
        tools: Sequence of sagent Tool instances allowed for this role.
            Does NOT include peer messaging — pass that as ``chat_tools``.
        model_id: Claude model id (from ``profile.models[role]``).
        system: Pre-rendered role-prompt body (``{{workspace}}`` already
            filled by :func:`team_profile.render_prompt`). The standing
            reminders are appended here. Mutually exclusive with
            ``role_md_path``; one of the two must be provided.
        role_md_path: Back-compat fallback — path to a role-prompt markdown
            file read directly when ``system`` is not given.
        peers: Active roster used to build the peer-messaging known-peers
            list; ``None`` falls back to the historical default roster.
        max_tool_call_rounds: Per-turn cap; ``None`` for no cap.
        max_budget_usd: Per-agent USD cap; ``None`` disables.
        provider: The shared provider (:func:`build_provider`); ``None``
            builds one this agent's model alone uses.
        journal: Where the session is written and what a restart resumes
            from — an OPEN journal, such as ``JsonlJournal`` at
            :func:`journal_path_for`. ``None`` keeps the session in memory.
        chat_tools: The peer-messaging tools of :func:`chat_session`
            (``mcp__sagent_chat__sagent_send`` and friends), connected with
            ``SAGENT_ROLE=role_name`` so the server knows which agent it
            serves on every call.

    Returns:
        A configured Agent; ``async with`` it (or ``aopen``) to run it.

    """
    from sagent.agent.agent import Agent
    from sagent.agent.agents_md import AgentsMdConfig

    if system is not None:
        system_prompt = compose_system_prompt(system, peers=peers)
    elif role_md_path is not None:
        system_prompt = load_system_prompt(role_md_path, peers=peers)
    else:
        raise ValueError("build_agent requires either 'system' or 'role_md_path'")

    if provider is None:
        provider = build_provider()

    return Agent(
        model=provider[model_id],
        system_prompt=system_prompt,
        tools=[*tools, *chat_tools],
        journal=journal,
        compactor=compactor_factory(),
        tool_context=ToolContext(label=role_name),
        max_rounds=max_tool_call_rounds,
        max_spend_usd=max_budget_usd,
        # A role's prompt is its profile's, whole: no AGENTS.md of the host's
        # (the operator's own sagent directives, or the launch cwd's) joins it.
        agents_md=AgentsMdConfig(tiers=frozenset()),
    )
