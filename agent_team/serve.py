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
"""Live serve: bring up all 5 agents + an HTTP/web viewer on 127.0.0.1.

Launch via the installed console script (works from any cwd):

    agent-team [serve] [--port 8767] [--host 127.0.0.1]
    # equivalently: python -m agent_team.serve [--port 8767]

What this does
--------------

1. Constructs the 5 role agents via the per-role ``build()`` factories.
   Each agent's CLI subprocess is launched with ``--mcp-config`` that
   includes the plugin's MCP server (``mcp_sagent/server.py``) with
   ``SAGENT_ROLE=<label>`` in its env — so peer messaging, self-defer,
   and status-report all flow through ``mcp__sagent_chat__*`` tool calls
   instead of sagent's bridge-mounted ``AgentSend``/``AgentSelf``.
2. Marks each agent ``_persistent=True`` and registers each in
   ``sagent.tools.core.agent_registry`` keyed by role label.
3. Adds ``user`` + ``system`` FakeAgents so peer messages addressed to
   them satisfy the "unknown agent" validation. The web UI is the
   user-visible surface for ``user``-targeted messages (read from
   ``main.jsonl``).
4. Starts ``serve_forever()`` for each agent as a background task.
5. Touches the ``_suppress_audit`` sentinel, fires the bootstrap probe
   (one MCP-bridge warm-up turn per agent), waits for AgentIdle, then
   removes the sentinel — so the warmup turn writes zero audit log
   records and pushes zero peer messages.
6. Starts a Starlette+uvicorn HTTP server on the configured host:port.

Shutdown via SIGTERM or Ctrl-C: the HTTP server stops first, then
each agent's runtime drains gracefully.
"""

from __future__ import annotations

from collections.abc import Sequence

import argparse
import asyncio
import json
import logging
import os
import signal
import subprocess
import sys


# Pulls in the data-dir resolution (env-configurable). All file paths
# below derive from ``delivery.DATA_DIR`` / ``delivery.SESSIONS_DIR``
# so a single ``SAGENT_DATA_DIR`` env var co-locates the audit log,
# trace files, per-role mcp.json, sentinel, and debug log — useful
# for end-of-day merges that union this plugin's main.jsonl with the
# sibling chat/ runtime's main.jsonl in one directory.
from datetime import UTC
from importlib.resources import files as _pkg_files
from pathlib import Path
from typing import Any

from .mcp_sagent import delivery


_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8767
_MAIN_JSONL = delivery.MAIN_JSONL_PATH

# Sentinel file: when present, the plugin's MCP server skips audit
# log writes AND peer inbox pushes. Set during the bootstrap warmup
# window. Path must match :data:`mcp_sagent.server._SUPPRESS_FLAG`.
_SUPPRESS_FLAG = delivery.SESSIONS_DIR / "_suppress_audit"

# Resume-slim: on every server restart, trim each agent's resumed tape to
# ~the last N history events (snapped to a turn boundary) BEFORE handing it to
# the provider. Deterministic (no model call), so a large tape can't blow the
# MCP-catalog deadline on the first-turn re-feed — the fat-tape boot wedge that
# took down TL (855 records / 1.97 MB) after a failed live compact. ON by
# default; set to 0 to disable (resume the full tape), or raise to keep more
# recent context. The on-disk JSONL is untouched — only the in-memory re-feed
# is bounded, and every restart re-slims from the full physical history.
_RESUME_KEEP_RECORDS = int(os.environ.get("AGENT_TEAM_RESUME_KEEP", "120"))

# Resume strategy (AGENT_TEAM_RESUME_MODE): how an agent's persisted tape is
# brought back on a server restart.
#   full        — resume the untrimmed tape; first turn re-feeds the whole
#                 resolved history via `claude --session-id` (the original
#                 fat-tape boot-wedge risk). Escape hatch.
#   slim        — trim the tape to its recent tail (`_RESUME_KEEP_RECORDS`)
#                 then resume; first turn re-feeds the trimmed tail.
#                 Deterministic, no model call. Also the materialize fallback.
#   materialize — write claude's session JSONL directly from the resumed
#                 context (vendored cli_session materializer) and flip the
#                 provider so turn-1 is a clean native `--resume` of that file
#                 — NO re-feed, exact mid-thread continuation, effectively
#                 instant startup. Gated by a boot drift-canary; falls back to
#                 slim if the canary finds the claude session format has drifted
#                 (or if `claude` is unavailable at boot). THE DEFAULT — proven
#                 live (TL's 1.98 MB tape resumed clean, all agents warmed up
#                 READY, 0 errors); set AGENT_TEAM_RESUME_MODE=slim to opt out.
_VALID_RESUME_MODES = ("full", "slim", "materialize")
_RESUME_MODE = os.environ.get("AGENT_TEAM_RESUME_MODE", "materialize").strip().lower()

# Lead/coordinator role for TL-anchored slim (`slim` mode only): the lead is
# slimmed by record count and every other agent is slimmed to the lead's
# wall-clock floor, so the busy coordinator doesn't end up with a SHORTER
# recent window than its peers (same record-count = less wall-clock for the
# lead) → the asymmetric-slim desync where peers report work the lead has
# forgotten. Empty/absent-from-roster disables anchoring (per-agent slim).
_RESUME_LEAD_ROLE = os.environ.get("AGENT_TEAM_RESUME_LEAD", "tl").strip()

_LOG = logging.getLogger("agent_team.serve")


# --------------------------------------------------------------------------
# Agent bring-up
# --------------------------------------------------------------------------


# Maps a profile roster label to its per-role ``build(profile)`` factory.
# The roster (and thus which subset of these is instantiated) is driven
# entirely by the profile; a single-label roster yields a team of one.
def _role_builders() -> dict:
    from .roles.junior_swe import build as build_junior
    from .roles.statistician import build as build_stat
    from .roles.swe import build as build_swe
    from .roles.tech_writer import build as build_tw
    from .roles.tl import build as build_tl

    return {
        "tl": build_tl,
        "swe": build_swe,
        "junior-swe": build_junior,
        "statistician": build_stat,
        "tech-writer": build_tw,
    }


def _build_all_agents(profile):
    """Construct + return the roster's role agents, each persistent + registered.

    Only the roles in ``profile.roster`` are instantiated (1..N). Each
    agent's ``claude --print`` subprocess is wired via ``--mcp-config`` to
    the plugin's MCP server with ``SAGENT_ROLE=<label>`` in the server's
    env — the sagent_send/sagent_defer/sagent_self tools appear in the
    catalog as ``mcp__sagent_chat__*`` and are the only path for peer
    messaging. A single-role roster boots fine: there are simply no peers
    to address (only ``user``).
    """
    from .runtime import trace_writer

    from sagent.testing import FakeAgent
    from sagent.tools.core import agent_registry

    builders = _role_builders()
    agents = {}
    for label in profile.roster:
        builder = builders.get(label)
        if builder is None:
            raise ValueError(
                f"roster references unknown role {label!r}; "
                f"known roles: {sorted(builders)}"
            )
        agent = builder(profile)
        agent._persistent = True
        agent_registry[label] = agent
        trace_writer.install_on(agent, label)
        agents[label] = agent

    # ``user`` is a mailbox-without-listener: ``sagent_send(to='user', …)``
    # lands here. The FakeAgent's inbox accumulates undelivered notes
    # harmlessly; the web UI shows them via the audit log.
    agent_registry["user"] = FakeAgent()
    # ``system`` is reserved for diagnostic probes / future eventing.
    agent_registry["system"] = FakeAgent()

    return agents


async def _serve_agents_forever(agents):
    """Spawn one ``serve_forever`` task per agent; return the gather handle."""
    tasks = [
        asyncio.create_task(agent.serve_forever(), name=f"serve-{label}")
        for label, agent in agents.items()
    ]
    return tasks


def _slim_resume_tape(tape, keep):
    """Trim a resumed tape to ~the last ``keep`` history events.

    Drops ``ContextSplice`` compaction barriers (they mask refs by ordinal;
    after a tail slice they'd reference dropped records) and keeps the last
    ``keep`` ``ReferrableTapeEvent``s, snapped FORWARD to the first inbound
    user-role message (``UserMessage`` or ``AgentSendMessage`` — both render as
    user-role and are clean turn boundaries) so the slice starts cleanly — no
    dangling ``tool_result`` whose ``tool_call`` was cut. ``replay_tape``
    tolerates the non-contiguous tail (next ordinal = ``max(ordinal)+1``).

    Deterministic, no model call — this is the whole point: a fat tape can't
    wedge the MCP-catalog handshake on resume the way a live ``compact()`` can
    (and did, on TL). Returns ``(slimmed, dropped)``; ``keep<=0`` disables.
    """
    from sagent.types.runtime import AgentSendMessage, UserMessage
    from sagent.types.tape import ContextSplice

    # Small tape (or disabled): leave it completely untouched — no risk worth
    # taking, and the re-feed is already cheap.
    if keep <= 0 or len(tape) <= keep:
        return tape, 0
    events = [r for r in tape if not isinstance(r, ContextSplice)]
    tail = events[-keep:] if len(events) > keep else events
    for i, r in enumerate(tail):
        if isinstance(getattr(r, "event", None), (UserMessage, AgentSendMessage)):
            tail = tail[i:]
            break
    return tail, len(tape) - len(tail)


def _inbound_floor_ts(records):
    """Earliest wall-clock timestamp among inbound (user-role) messages.

    Inbound ``UserMessage`` / ``AgentSendMessage`` tape events each carry a
    ``.timestamp`` (epoch float) — these are the cross-agent sync points. The
    min over a slimmed lead tape is the lead's recent-window floor: the anchor
    every other agent is trimmed to. Returns ``None`` when there is no
    timestamped inbound message (caller falls back to per-agent slim).
    """
    from sagent.types.runtime import AgentSendMessage, UserMessage

    floors = []
    for r in records:
        ev = getattr(r, "event", None)
        if isinstance(ev, (UserMessage, AgentSendMessage)):
            t = getattr(ev, "timestamp", None)
            if isinstance(t, (int, float)) and t > 0:
                floors.append(t)
    return min(floors) if floors else None


def _slim_tape_to_floor(tape, floor_ts):
    """Keep tape records from the first inbound message at/after ``floor_ts``.

    Drops ``ContextSplice`` barriers and everything older than the shared floor,
    starting the slice at a clean user-role turn boundary. When the agent has
    been **idle since before the floor** (no inbound at/after it), keep ONLY its
    most recent inbound turn — a breadcrumb, not the full stale window that
    drives the churn (an idle agent resuming an abandoned task). Returns
    ``None`` only when there is no inbound message at all (degenerate; the
    caller keeps the per-agent record slim).
    """
    from sagent.types.runtime import AgentSendMessage, UserMessage
    from sagent.types.tape import ContextSplice

    events = [r for r in tape if not isinstance(r, ContextSplice)]
    inbound = [
        i
        for i, r in enumerate(events)
        if isinstance(getattr(r, "event", None), (UserMessage, AgentSendMessage))
    ]
    if not inbound:
        return None
    for i in inbound:
        t = getattr(events[i].event, "timestamp", None)
        if isinstance(t, (int, float)) and t >= floor_ts:
            return events[i:]
    # idle since before the floor → keep just the most recent turn.
    return events[inbound[-1] :]


def _plan_slim_resume(loaded):
    """Compute each agent's slimmed tape for a resume.

    ``loaded`` maps ``label -> (meta, tape, tool_state)``. Returns
    ``label -> (slimmed_tape, note)``.

    - ``full`` / ``materialize``: resume the whole tape (materialize writes the
      session file directly, so there's no re-feed to bound).
    - ``slim``: **TL-anchored**. Slim the lead by record count to fix its recent
      window, take that window's wall-clock floor, and slim every other agent to
      that same floor — one shared temporal horizon, so no agent retains work
      the lead has forgotten (the desync). Agents idle since before the floor,
      or any case with no derivable anchor, fall back to a per-agent record slim
      (never a regression vs the pre-anchor behaviour).
    """
    if _RESUME_MODE != "slim":
        return {label: (tape, "") for label, (_m, tape, _t) in loaded.items()}

    keep = _RESUME_KEEP_RECORDS
    plan: dict = {}
    floor = None
    lead = _RESUME_LEAD_ROLE
    if lead and lead in loaded:
        lead_slim, _ = _slim_resume_tape(loaded[lead][1], keep)
        floor = _inbound_floor_ts(lead_slim)
        plan[lead] = (
            lead_slim,
            f" lead-anchor floor={floor:.0f}" if floor else " lead (no anchor)",
        )

    for label, (_m, tape, _t) in loaded.items():
        if label in plan:
            continue
        if floor is not None:
            anchored = _slim_tape_to_floor(tape, floor)
            if anchored is not None:
                plan[label] = (anchored, " anchored-to-lead")
                continue
        slimmed, _ = _slim_resume_tape(tape, keep)
        plan[label] = (slimmed, " per-agent")
    return plan


def _probe_claude_version() -> str:
    """Return the live ``claude --version`` first token (e.g. ``2.1.183``).

    The materialized session JSONL stamps this into every entry's ``version``;
    it should match the CLI that will ``--resume`` the file. Raises on failure
    so the caller bails rather than writing a sentinel version.
    """
    out = subprocess.run(
        ["claude", "--version"], capture_output=True, text=True, timeout=10
    ).stdout.strip()
    return out.split()[0]


def _probe_git_branch(cwd) -> str:
    """Best-effort current git branch for ``cwd`` (``HEAD`` on any failure)."""
    try:
        b = subprocess.run(
            ["git", "-C", str(cwd), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        return b or "HEAD"
    except Exception:
        return "HEAD"


def _materialize_on_resume(label: str, agent) -> bool:
    """Write claude's session JSONL from the resumed context + flip the seam.

    After ``agent.resume(...)`` the runtime holds the rehydrated (resolved)
    context. We render it to a CLI-shaped session file at the exact path the
    provider will read, then set BOTH provider seams so turn-1 is a clean
    native ``--resume`` instead of a ``--session-id`` full re-feed:

    - ``_session_initialized = True`` → the spawn uses ``--resume <uuid>``
      (read at ``anthropic_cli.py:1290``).
    - ``_last_sent_index = len(messages)`` → turn-1's stdin delta
      (``request.messages[_last_sent_index:]``, line 877) is ~empty, so the
      N history messages already in the file are NOT re-fed (no double-apply,
      no wedge). Setting only ``_session_initialized`` would re-feed everything.

    Best-effort: any problem logs a warning, leaves the seams untouched, and
    returns False — the agent is already correctly resumed, so turn-1 simply
    falls back to the ``--session-id`` rebuild (the slim/full path). Never
    raises into the boot loop. Returns True iff the seam was flipped.
    """
    from sagent.types.model import ModelRequest

    from agent_team.cli_session import materialize_session

    model = getattr(agent, "model", None)
    session_id = getattr(model, "_session_id", None)
    if model is None or session_id is None:
        return False  # stateless / non-CLI provider — nothing to materialize
    if (
        getattr(model, "_session_jsonl_path", None) is None
        or getattr(model, "_claude_home", None) is None
    ):
        return False

    try:
        messages = list(agent.runtime.context().messages)
        # The materializer renders request.messages only (system/tools ride on
        # the CLI argv, not the session file).
        request = ModelRequest(messages=messages)
        try:
            cwd = Path.cwd().resolve()
        except OSError:
            cwd = Path.cwd()
        home = model._claude_home()
        cli_version = _probe_claude_version()
        path, _entries = materialize_session(
            request,
            session_id=session_id,
            cwd=cwd,
            git_branch=_probe_git_branch(cwd),
            cli_version=cli_version,
            home=home,
        )
    except Exception as exc:
        _LOG.warning(
            "materialize %s: failed (%s: %s); turn-1 will --session-id rebuild",
            label,
            type(exc).__name__,
            exc,
        )
        return False

    # The path we wrote MUST equal what the provider will --resume; if a
    # symlinked cwd or HOME made them diverge, do NOT flip the seam.
    expected = model._session_jsonl_path()
    if expected is None or Path(expected) != Path(path):
        _LOG.warning(
            "materialize %s: path mismatch (wrote %s, provider expects %s); "
            "NOT flipping seam — turn-1 will rebuild",
            label,
            path,
            expected,
        )
        return False

    # Flip both seams: turn-1 → `--resume` with a ~empty stdin delta.
    model._session_initialized = True
    model._last_sent_index = len(messages)
    _LOG.info(
        "materialize %s: wrote %d-msg session at %s (cli %s); turn-1 will --resume",
        label,
        len(messages),
        path,
        cli_version,
    )
    return True


def _resume_agents_from_session_dir(
    agents, *, materialize_enabled: bool = False
) -> None:
    """Rehydrate each agent's tape from its sagent ``session.jsonl``.

    Standard sagent restart-resume: each role's ``build_agent`` wired a
    per-role ``session_dir`` (``Agent(session_dir=...)``), so a prior
    ``serve.py`` run persisted its tape to ``<session_dir>/session.jsonl``.
    On boot we ``load_session`` + ``Agent.resume`` each one. The resumed
    tape then drives the provider: on the agent's first turn after resume,
    the AnthropicCLI model re-feeds that history to ``claude --session-id``,
    which rebuilds the on-disk CLI session; every later turn ``--resume``s and
    feeds only deltas. No claude-session-file parsing -- sagent's own tape is
    the single source of truth.

    Mode (``AGENT_TEAM_RESUME_MODE``): ``slim`` trims each tape to its recent
    tail so the first-turn re-feed stays small — the deterministic answer to the
    fat-tape boot wedge — and **anchors the trim across agents on the lead**
    (``_plan_slim_resume``) so a restart can't desync the team. ``full`` resumes
    the untrimmed tape. ``materialize`` resumes the untrimmed tape AND (when the
    boot drift-canary passed, ``materialize_enabled``) writes claude's session
    file from the resolved context + flips the provider so turn-1 is a clean
    native ``--resume`` (no re-feed at all). The on-disk sagent JSONL is
    untouched in every mode.

    Three phases so ``slim`` can anchor across agents: load every tape, plan the
    (cross-agent) slim, then resume + materialize. Best-effort and never crashes
    boot: a per-agent failure logs a warning and that agent starts fresh.
    """
    from sagent.agent.session_io import load_session

    # Phase 1 — load every agent's persisted tape.
    loaded: dict = {}  # label -> (meta, tape, tool_state)
    for label, agent in agents.items():
        if agent.session_dir is None:
            continue
        try:
            res = load_session(agent.session_dir, {})
        except Exception as exc:
            _LOG.warning(
                "resume %s: load failed (%s: %s); agent starts fresh",
                label,
                type(exc).__name__,
                exc,
            )
            continue
        if res is None:
            _LOG.info("resume %s: no prior session.jsonl — fresh start", label)
            continue
        loaded[label] = res

    # Phase 2 — plan the slim (TL-anchored in `slim`; whole tape otherwise).
    plan = _plan_slim_resume(loaded)

    # Phase 3 — resume each agent, then (materialize) write its session file.
    for label, (meta, tape, tool_state) in loaded.items():
        agent = agents[label]
        try:
            slimmed, note = plan[label]
            agent.resume(meta, slimmed, tool_state)
            if len(slimmed) != len(tape):
                _LOG.info(
                    "resume %s [%s]: %d -> %d tape records%s",
                    label,
                    _RESUME_MODE,
                    len(tape),
                    len(slimmed),
                    note,
                )
            else:
                _LOG.info(
                    "resume %s [%s]: %d tape records%s",
                    label,
                    _RESUME_MODE,
                    len(tape),
                    note,
                )
            # materialize mode: write the CLI session file from the resolved
            # context + flip the seam so turn-1 is a clean native --resume.
            if _RESUME_MODE == "materialize" and materialize_enabled:
                _materialize_on_resume(label, agent)
        except Exception as exc:
            _LOG.warning(
                "resume %s: resume failed (%s: %s); agent starts fresh",
                label,
                type(exc).__name__,
                exc,
            )


# --------------------------------------------------------------------------
# Startup warmup — fire one MCP-tool-call-using turn per agent before
# accepting user traffic, so the first real user message doesn't hit the
# AnthropicCLI's "first-turn fumbles MCP tool calls" pattern observed in
# spike sigint_probe3 and race_repro_v2.
# --------------------------------------------------------------------------


_WARMUP_PROMPT = (
    "BOOTSTRAP PROBE. You MUST call the tool "
    "`mcp__sagent_chat__sagent_self` with arguments "
    '`{"status": "ready"}` right now, as your FIRST and ONLY '
    "action in this turn. Do not respond with text in place of the "
    "tool call — describing what you would do does not count and "
    "the bootstrap will be considered failed. After the tool returns "
    "its result, end the turn with the single word `ok` as a "
    "non-empty text content block.\n\n"
    "This is a one-time startup probe, not a real request. Future "
    "messages from peers and users are real work and must be "
    "handled normally — never reuse this bootstrap pattern as a "
    "reply template (e.g. do not reply to a real message with "
    "`sagent_send(content='ready')` or `'hello, ready'`)."
)


async def _warmup_agents(agents, *, timeout_s: float = 90.0) -> dict[str, bool]:
    """Send a warmup directive to each agent in parallel; wait for AgentIdle.

    Returns a ``{label: success}`` map. ``success=True`` means the agent
    reached AgentIdle within ``timeout_s`` after the warmup push;
    ``False`` means it timed out (we proceed anyway — a slow warmup
    shouldn't block the whole server, and the audit log will surface
    any agent that never speaks).

    The MCP server's audit-log writes AND peer inbox pushes are
    suppressed by touching the ``_SUPPRESS_FLAG`` sentinel before the
    warmup push and removing it after every agent has reached
    AgentIdle (or timed out). The MCP server reads this file's
    existence on every ``CallToolRequest``, so toggling propagates
    across all per-agent MCP server subprocesses without needing to
    restart any of them. The bootstrap probe uses ``sagent_self``
    (a status-report tool that does not touch any peer inbox) so the
    warmup turn is silent to peers by construction.

    The trailing "acknowledge with 'ok'" instruction is deliberate —
    Anthropic's API rejects assistant messages with empty text
    content blocks (``400 messages: text content blocks must be
    non-empty``), so a single-word ack on the warmup turn guarantees
    the history shape is valid for subsequent turns.
    """
    from sagent.types.runtime import AgentIdle, UserMessage

    try:
        idle_events: dict[str, asyncio.Event] = {}
        observers: list[Any] = []
        for label, agent in agents.items():
            evt = asyncio.Event()
            idle_events[label] = evt

            def _watcher(ev, _evt=evt):
                if isinstance(ev, AgentIdle):
                    _evt.set()

            agent.runtime.observers.append(_watcher)
            observers.append((agent, _watcher))

        # Touch the suppression sentinel BEFORE pushing the prompt so
        # the very first MCP CallToolRequest fired by the bootstrap
        # turn sees an empty audit log + a silent inbox.
        _SUPPRESS_FLAG.parent.mkdir(parents=True, exist_ok=True)
        _SUPPRESS_FLAG.touch()

        for label, agent in agents.items():
            agent.runtime.inbox.push_back(UserMessage(text=_WARMUP_PROMPT))

        async def _wait_one(label: str) -> tuple[str, bool]:
            try:
                await asyncio.wait_for(idle_events[label].wait(), timeout_s)
                return label, True
            except TimeoutError:
                return label, False

        results = await asyncio.gather(*[_wait_one(l) for l in agents])
        for agent, watcher in observers:
            try:
                agent.runtime.observers.remove(watcher)
            except ValueError:
                pass
        # DO NOT call Agent.clear() here. It works in principle (preempt
        # + wipe history + reset per-tool recall) but in the
        # AnthropicCLI path, the history shrink trips
        # ``_should_respawn`` in the provider, which kills the warm
        # CLI subprocess and respawns a fresh one — undoing exactly
        # the MCP-bridge warmup we just paid for. Leave the warmup
        # turn in history; the new prompt's explicit "do NOT reuse
        # this pattern" framing + the use of ``sagent_self`` (silent
        # to peers) means even a parrot-after-respawn doesn't leak.
        return dict(results)
    finally:
        # Re-enable audit logging + peer deliveries for real traffic.
        _SUPPRESS_FLAG.unlink(missing_ok=True)


# --------------------------------------------------------------------------
# HTTP routes
# --------------------------------------------------------------------------


# Web UI ships INSIDE the package (``agent_team/web/*.html``); resolve via
# importlib.resources so it works when installed into a venv from any cwd
# (NOT ``__file__`` arithmetic that breaks under a relocated package). The
# returned objects are ``Traversable``s; both ``.is_file()`` and
# ``.read_text()`` work the same as on ``pathlib.Path``.
_WEB_DIR = _pkg_files("agent_team") / "web"
_DEBUG_HTML_PATH = _WEB_DIR / "debug.html"
_INDEX_HTML_PATH = _WEB_DIR / "index.html"


def _event_search_text(ev: dict[str, Any]) -> tuple[str, str]:
    """Return ``(kind, flattened-searchable-text)`` for one trace event.

    Mirrors ``chat/chat:_event_search_text`` but uses sagent's event
    field names. Trace events come from ``trace_writer.TraceWriter``
    which serializes RuntimeEvent dataclasses by ``_event`` + their
    fields (e.g. ``text``, ``tool_calls``, ``message``, ``source``).
    """
    kind = ev.get("_event") or "?"
    parts: list[str] = []
    # Top-level text on AssistantMessage / AgentSendMessage /
    # UserMessage / ModelResponsePartial events.
    txt = ev.get("text")
    if isinstance(txt, str) and txt:
        parts.append(txt)
    # Nested message payload (ModelResponseComplete).
    msg = ev.get("message")
    if isinstance(msg, dict):
        mt = msg.get("text")
        if isinstance(mt, str) and mt:
            parts.append(mt)
        tcs = msg.get("tool_calls")
        if isinstance(tcs, list):
            for tc in tcs:
                if isinstance(tc, dict):
                    name = str(tc.get("name", ""))
                    args = tc.get("args") or tc.get("arguments") or {}
                    try:
                        args_str = json.dumps(args, ensure_ascii=False)
                    except (TypeError, ValueError):
                        args_str = str(args)
                    parts.append(f"{name} {args_str}")
    # ToolLabel events carry an inline label.
    if "label" in ev and isinstance(ev["label"], str):
        parts.append(ev["label"])
    # Channel records embedded in sagent events (rare but possible).
    src = ev.get("source")
    if isinstance(src, str) and src:
        parts.append(src)
    return kind, "  ".join(parts)


def _snippet(text: str, q: str, width: int = 180) -> str:
    """One-line window of ``text`` centred on the first match of ``q``."""
    low = text.lower()
    i = low.find(q.lower())
    if i < 0:
        return text[:width].replace("\n", " ")
    start = max(0, i - 50)
    end = min(len(text), i + len(q) + (width - 50))
    s = text[start:end].replace("\n", " ")
    return ("…" if start > 0 else "") + s + ("…" if end < len(text) else "")


def _iter_trace_files():
    """Yield ``(role, path)`` for every ``<role>.trace.jsonl`` under sessions/."""
    sessions = delivery.SESSIONS_DIR
    if not sessions.exists():
        return
    for p in sorted(sessions.glob("*.trace.jsonl")):
        yield p.name[: -len(".trace.jsonl")], p


def _read_jsonl(path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                out.append(rec)
    return out


def _search_all(q: str, scope: str, limit: int) -> tuple[list[dict[str, Any]], bool]:
    """Grep ``main.jsonl`` and/or every ``sessions/*.trace.jsonl`` for ``q``."""
    results: list[dict[str, Any]] = []
    if not q:
        return results, False
    ql = q.lower()
    if scope in ("messages", "all"):
        for i, m in enumerate(_read_jsonl(_MAIN_JSONL)):
            body = m.get("body", "") or ""
            if ql in body.lower():
                results.append(
                    {
                        "source": "message",
                        "idx": i,
                        "ts": m.get("ts", ""),
                        "from": m.get("from", ""),
                        "to": m.get("to", []),
                        "snippet": _snippet(body, q),
                    }
                )
                if len(results) >= limit:
                    return results, True
    if scope in ("traces", "all"):
        for role, p in _iter_trace_files():
            for i, ev in enumerate(_read_jsonl(p)):
                kind, text = _event_search_text(ev)
                if ql in text.lower():
                    results.append(
                        {
                            "source": "trace",
                            "role": role,
                            "idx": i,
                            "ts": ev.get("_ts", ""),
                            "kind": kind,
                            "snippet": _snippet(text, q),
                        }
                    )
                    if len(results) >= limit:
                        return results, True
    return results, False


def _diagnose_agent(label: str, agent) -> dict[str, Any]:
    """Compute status + diagnosis for one agent, mirroring chat/'s _agents_status fields.

    Sagent's single-process model maps the chat/ status set (dead, hung,
    stuck, working, idle) like this:
      - dead   : never (if HTTP is up, the agent's asyncio task is alive)
      - working: model_call is not None
      - hung   : model_call active AND last assistant turn >90s ago (or never)
      - stuck  : inbox has queued messages AND no model_call AND no recent activity
      - idle   : model_call None and inbox empty
    """
    from datetime import datetime

    in_flight_call = agent.runtime.model_call is not None
    inbox_pending = 0
    try:
        inbox_pending = agent.runtime.inbox._queue.qsize()
    except AttributeError:
        pass

    # Compute last_assistant_ts + age_sec from history tail.
    last_ts_iso: str | None = None
    age_sec: int | None = None
    for m in reversed(agent.history):
        if type(m).__name__ == "AssistantMessage":
            ts = getattr(m, "timestamp", None)
            if isinstance(ts, (int, float)):
                last_ts_iso = (
                    datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.")
                    + f"{int((ts % 1) * 1000):03d}Z"
                )
                age_sec = int(datetime.now(UTC).timestamp() - ts)
            break

    # Recent trace events (last 6) — sourced from the trace file rather
    # than agent.runtime.events to match chat/'s "tail the trace" check.
    try:
        from .runtime import trace_writer

        tp = trace_writer.trace_path_for(label)
        events = _read_jsonl(tp)
    except Exception:
        events = []
    recent = []
    for ev in events[-6:]:
        kind, text = _event_search_text(ev)
        recent.append(
            {
                "ts": ev.get("_ts", ""),
                "kind": kind,
                "summary": text[:200],
            }
        )

    # Inflight detection: was there a ModelCallStarted with no matching
    # ModelIdle/ModelResponseComplete/ModelResponseError in the recent tail?
    in_turn = False
    started_idx = -1
    ended_idx = -1
    for i, ev in enumerate(events[-100:]):
        k = ev.get("_event") or "?"
        if k == "ModelCallStarted":
            started_idx = i
        elif k in ("ModelIdle", "ModelResponseComplete", "ModelResponseError"):
            ended_idx = i
    in_turn = started_idx > ended_idx

    inflight: str | None = None
    if in_turn:
        # The most recent ToolLabel published since the turn began is our
        # best proxy for "what tool is currently running".
        for ev in reversed(events[-100:]):
            if (ev.get("_event") or "?") == "ToolLabel":
                inflight = str(ev.get("text") or ev.get("label") or "")
                break
        if inflight is None:
            inflight = "model thinking"

    # Last completed turn outcome.
    last_result: dict[str, Any] | None = None
    for ev in reversed(events):
        k = ev.get("_event") or "?"
        if k == "ModelResponseComplete":
            last_result = {"ok": True, "ts": ev.get("_ts", "")}
            break
        if k == "ModelResponseError":
            last_result = {"ok": False, "ts": ev.get("_ts", "")}
            break

    # Status verdict.
    if in_flight_call and (age_sec is None or age_sec > 90):
        status = "hung"
    elif in_flight_call:
        status = "working"
    elif inbox_pending > 0 and (age_sec is None or age_sec > 120):
        status = "stuck"
    elif age_sec is not None and age_sec < 60:
        status = "working"
    else:
        status = "idle"

    # One-line diagnosis.
    if status == "hung":
        tail = f" — in-flight {inflight}" if inflight else ""
        diagnosis = (
            f"Model call in flight for {age_sec or '?'}s with no progress{tail}. "
            f"May be a slow CLI subprocess or a stuck tool; consider restart."
        )
    elif status == "stuck":
        diagnosis = (
            f"Inbox has {inbox_pending} message(s) queued and no recent activity "
            f"({age_sec or '?'}s since last assistant turn). Likely a runtime drain stall."
        )
    elif status == "working":
        if in_flight_call:
            diagnosis = f"Model call active ({age_sec or '?'}s since last reply)."
        else:
            diagnosis = f"Recently active ({age_sec}s since last assistant turn)."
    else:
        diagnosis = "Last turn complete, inbox empty — waiting for work."

    model_id = ""
    try:
        model_id = agent.model.model_id
    except AttributeError:
        pass

    # Pending preview from the inbox.
    # asyncio.Queue doesn't expose its underlying deque publicly; we peek
    # at the private ``_queue`` (collections.deque) for the preview.
    # Best-effort and read-only — never mutated.
    pending_preview: list[dict[str, Any]] = []
    try:
        deque_items = list(agent.runtime.inbox._queue._queue)
        for item in deque_items[:4]:
            text = (getattr(item, "text", "") or "").replace("\n", " ")[:140]
            src = getattr(item, "source", "") or type(item).__name__
            pending_preview.append(
                {
                    "from": src,
                    "ts": "",
                    "snippet": text,
                }
            )
    except (AttributeError, TypeError):
        pass

    return {
        "role": label,
        "alive": True,
        "pid": None,  # not meaningful in single-process model
        "status": status,
        "diagnosis": diagnosis,
        "in_turn": in_turn,
        "inflight": inflight,
        "blocked_on": None,
        "wchan": None,
        "last_result": last_result,
        "last_ts": last_ts_iso,
        "trace_mtime": None,
        "age_sec": age_sec,
        "pending": inbox_pending,
        "pending_preview": pending_preview,
        "recent": recent,
        "model_id": model_id,
        "total_cost_usd": float(agent.total_cost_usd),
    }


def _render_recent_turns(
    agent, label: str, *, n_turns: int = 2, max_chars: int = 1800
) -> str:
    """Render the agent's last ``n_turns`` turns as a plain-text breadcrumb.

    A *turn* starts at a ``UserMessage`` (an inbound operator/peer message or
    the prior bootstrap) and runs through the assistant's reply. The
    ``clear tape`` restart hands this to the freshly-cleared agent as
    orientation — WITHOUT reconstructing tape records, which would risk an
    invalid post-barrier tape (dangling tool_result / broken ordinal refs).
    Tool results and other low-signal entries are dropped. Best-effort: any
    failure returns ``""`` so the restart never blocks on it.
    """
    try:
        from sagent.types.runtime import AssistantMessage, UserMessage

        messages = list(agent.runtime.context().messages)
    except Exception:
        return ""
    user_idxs = [i for i, m in enumerate(messages) if isinstance(m, UserMessage)]
    if not user_idxs:
        return ""
    start = user_idxs[-n_turns] if len(user_idxs) >= n_turns else user_idxs[0]
    lines: list[str] = []
    for m in messages[start:]:
        if isinstance(m, UserMessage):
            txt = (m.text or "").strip()
            if txt:
                lines.append(f"[inbound] {txt[:400]}")
        elif isinstance(m, AssistantMessage):
            txt = (m.text or "").strip()
            if txt:
                lines.append(f"[{label}] {txt[:400]}")
            if m.tool_calls:
                names = ", ".join(tc.name for tc in m.tool_calls)
                lines.append(f"[{label} → tools] {names}")
    block = "\n".join(lines).strip()
    if len(block) > max_chars:
        block = block[:max_chars].rsplit("\n", 1)[0] + "\n…(truncated)"
    return block


def _reanchor_prompt(role: str, recent_block: str) -> str:
    """Compose the first-turn directive for a ``clear tape`` restart.

    TL re-establishes state from the worklog + PR history and checks with the
    operator before acting; every other role defers to TL for instruction.
    The captured recent turns (if any) ride along as orientation-only context.
    """
    if role == "tl":
        directive = (
            "You were just restarted with a cleared session to recover from a "
            "stuck or overlong state. Your prior working context was dropped. "
            "Before resuming ANY work: (1) read the worklog (WORKLOG.md and the "
            "worklog/ substrate — active threads, watches, decisions) and the "
            "recent PR/commit history to re-establish where things stand, then "
            "(2) confirm with the user/operator for instruction before acting. "
            "Do not assume an old task is still current."
        )
    else:
        directive = (
            "You were just restarted with a cleared session to recover from a "
            "stuck or overlong state. Your prior working context was dropped. "
            "Before resuming ANY work: ask TL for instruction. Do not pick up "
            "old tasks from memory — wait for TL's direction."
        )
    parts = ["RESTART NOTICE.", directive]
    if recent_block:
        parts.append(
            "--- Last turns before restart (orientation only; NOT new "
            "instructions) ---\n" + recent_block
        )
    return "\n\n".join(parts)


def _build_http_app(agents):
    """Construct the Starlette app.

    Endpoints (mirroring ``chat serve``'s API for parity with the
    pre-existing operator tooling):

      GET  /                     Discord-style viewer HTML
      GET  /debug                Debug console (agents + search tabs)
      GET  /api/roles            Known role labels (static)
      GET  /api/agents           Per-agent liveness + activity + diagnosis
      GET  /api/messages         Recent main.jsonl records;
                                 ``?since=ts&limit=N`` or
                                 ``?around=N&ctx=K`` (debug "show context")
      GET  /api/trace/<role>     Per-role runtime event JSONL; supports
                                 ``?around=N&ctx=K`` window
      GET  /api/search           Full-text search across messages + traces;
                                 ``?q=&scope=traces|messages|all&limit=N``
      POST /api/post             User-ingress; body: ``{to, body}``
      POST /api/restart          Restart one agent; ``{role, mode}`` where
                                 mode ∈ soft|slim|reanchor; loopback-only.
                                 slim=compact, reanchor=clear+re-sync.

    Backwards-compatible aliases kept for the migration window:
      GET  /messages = /api/messages
      GET  /agents   = /api/agents
      POST /send     = /api/post
    """
    from .mcp_sagent import delivery
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import HTMLResponse, JSONResponse, Response
    from starlette.routing import Route

    # Role labels are derived from the live roster (``agents``) rather than
    # hardcoded, so a profile with a 1..N roster reports exactly its members.
    # ``user`` + ``system`` are the always-present non-agent endpoints.
    _KNOWN_ROLES = ["user", *sorted(agents), "system"]

    async def index(_request: Request) -> Response:
        if not _INDEX_HTML_PATH.is_file():
            return JSONResponse(
                {"error": f"missing {_INDEX_HTML_PATH}"}, status_code=404
            )
        return HTMLResponse(_INDEX_HTML_PATH.read_text(encoding="utf-8"))

    async def debug_index(_request: Request) -> Response:
        if not _DEBUG_HTML_PATH.is_file():
            return JSONResponse(
                {"error": f"missing {_DEBUG_HTML_PATH}"}, status_code=404
            )
        return HTMLResponse(_DEBUG_HTML_PATH.read_text(encoding="utf-8"))

    async def list_roles(_request: Request) -> Response:
        return JSONResponse({"roles": _KNOWN_ROLES})

    async def search(request: Request) -> Response:
        qp = request.query_params
        q = (qp.get("q") or "").strip()
        scope = qp.get("scope") or "traces"
        if scope not in ("traces", "messages", "all"):
            scope = "traces"
        try:
            limit = max(1, min(1000, int(qp.get("limit", "300"))))
        except ValueError:
            limit = 300
        results, truncated = _search_all(q, scope, limit)
        return JSONResponse(
            {
                "q": q,
                "scope": scope,
                "results": results,
                "truncated": truncated,
            }
        )

    async def restart(request: Request) -> Response:
        # One agent, three intensities (``mode``); single-process model, so we
        # can't kill+respawn one subprocess cleanly. From gentlest to bluntest:
        #
        #   slim     -- ``Agent.compact()``: summarize old turns → keep a real
        #               summary + the recent working thread. Shrinks the
        #               effective re-feed (the splice/override persists, so a
        #               later boot resumes slim too) WITHOUT wiping context or
        #               interrupting the task. This is the normal async
        #               ``/compact`` path (NOT ``compact_now``, the synchronous
        #               overflow-recovery path that crashed tech-writer); it
        #               fails safe — the tape is untouched on CompactFailed.
        #               For a long-but-healthy agent (e.g. TL) whose tape is
        #               growing but whose coordination memory must survive.
        #   soft     -- ``Agent.clear()`` only, inbox preserved. Full wipe.
        #   reanchor -- capture a last-turns breadcrumb, drop the backlog,
        #               ``clear()``, then push a role-specific "re-sync before
        #               acting" directive. ``clear()`` writes a durable
        #               ``context_clear`` barrier so the re-feed is tiny — this
        #               kills the catalog-timeout wedge a full-tape re-feed
        #               triggers. For recovering a stuck/confused agent.
        from sagent.types.runtime import UserMessage

        client_host = (request.client.host if request.client else "") or ""
        if client_host not in ("127.0.0.1", "::1", "localhost", ""):
            return JSONResponse(
                {"error": "restart disabled: not a loopback client"},
                status_code=403,
            )
        try:
            payload = await request.json()
        except Exception as exc:
            return JSONResponse({"error": f"bad json: {exc}"}, status_code=400)
        role = str(payload.get("role") or "").strip()
        # ``mode`` is canonical; the legacy booleans
        # (``clear_tape``/``reanchor``/``skip_backlog``) still map to reanchor.
        mode = str(payload.get("mode") or "").strip().lower()
        if not mode:
            mode = (
                "reanchor"
                if (
                    payload.get("clear_tape")
                    or payload.get("reanchor")
                    or payload.get("skip_backlog")
                )
                else "soft"
            )
        if mode not in ("soft", "slim", "reanchor"):
            return JSONResponse(
                {"error": f"unknown mode {mode!r}; use soft|slim|reanchor"},
                status_code=400,
            )
        if role not in agents:
            return JSONResponse(
                {"error": f"unknown role {role!r}", "known": sorted(agents)},
                status_code=404,
            )
        agent = agents[role]
        notes: list[str] = []

        # --- slim: compaction only, no wipe, no re-anchor --------------------
        if mode == "slim":
            try:
                before = len(agent.runtime.context().messages)
            except Exception:
                before = None
            # Guards: live ``compact()`` runs a model call that re-feeds the
            # resolved context. On a large context that re-feed blows the
            # MCP-catalog deadline and crashes the compaction subprocess (the
            # TL fat-tape wedge). And on a still-settling agent (model call in
            # flight after a restart) the same re-feed is unstable. Refuse and
            # point at the safe paths rather than crash the agent.
            slim_max = int(os.environ.get("AGENT_TEAM_SLIM_MAX_MESSAGES", "250"))
            if before is not None and before > slim_max:
                return JSONResponse(
                    {
                        "ok": False,
                        "role": role,
                        "mode": mode,
                        "output": (
                            f"refused: context is {before} messages (> {slim_max}). A "
                            f"live compact would re-feed the whole tape and risk a "
                            f"catalog-timeout crash → wedge. Use a server restart "
                            f"(resume-slim trims to the last "
                            f"AGENT_TEAM_RESUME_KEEP={_RESUME_KEEP_RECORDS}) or "
                            f"mode=reanchor. Override via AGENT_TEAM_SLIM_MAX_MESSAGES."
                        ),
                    }
                )
            if getattr(agent.runtime, "model_call", None) is not None:
                return JSONResponse(
                    {
                        "ok": False,
                        "role": role,
                        "mode": mode,
                        "output": (
                            "refused: agent is mid-turn / still settling (model call in "
                            "flight). Let it reach idle before slimming, or use "
                            "mode=reanchor to preempt + reset."
                        ),
                    }
                )
            try:
                # ``compact()`` resolves on CompactComplete OR CompactFailed;
                # the timeout guards a wedged compaction subprocess. A large
                # tape's summary call can take a while, hence the generous cap.
                await asyncio.wait_for(agent.compact(), timeout=180.0)
                after = (
                    len(agent.runtime.context().messages)
                    if before is not None
                    else None
                )
                compact_err = getattr(agent, "last_compact_error", None)
                if compact_err:
                    notes.append(f"compaction failed: {compact_err}; tape unchanged")
                    ok = False
                elif before is not None and after is not None and after < before:
                    notes.append(f"compacted: context {before} → {after} messages")
                    ok = True
                else:
                    notes.append(
                        f"compaction ran; context {before}→{after} messages "
                        "(little to compact — already slim?)"
                    )
                    ok = True
            except (asyncio.TimeoutError, TimeoutError):
                notes.append(
                    "compact() timed out after 180s; tape unchanged, agent "
                    "unharmed — retry, or use reanchor if wedged."
                )
                ok = False
            except Exception as exc:
                notes.append(f"compact() failed: {type(exc).__name__}: {exc}")
                ok = False
            return JSONResponse(
                {"ok": ok, "role": role, "mode": mode, "output": "\n".join(notes)}
            )

        # --- reanchor: capture breadcrumb + drain backlog, then clear --------
        recent_block = ""
        if mode == "reanchor":
            # Capture the breadcrumb BEFORE clear() wipes the live context.
            recent_block = _render_recent_turns(agent, role)
            # Drain the inbox (drop stale backlog). GatedDeque wraps
            # asyncio.Queue (sagent runtime.py:552); its underlying
            # ``_queue`` is a ``collections.deque`` we drop directly.
            try:
                q = agent.runtime.inbox._queue
                drained = 0
                while not q.empty():
                    try:
                        q.get_nowait()
                        drained += 1
                    except Exception:
                        break
                notes.append(f"drained {drained} pending inbox item(s)")
            except AttributeError:
                notes.append("inbox drain unsupported on this sagent version")

        # --- soft + reanchor: clear() (timeout-guarded) ----------------------
        # ``Agent.clear()`` preempts the in-flight model_call + wipes
        # history + resets the per-tool recall cache. The HotSpare's
        # ``_should_respawn`` check then triggers a fresh ``claude --print``
        # subprocess on the next model call (history.length < last_sent_index).
        # Bounded by a timeout: a hard-wedged agent can block clear() on
        # ``await proc.close()`` for a SIGINT-resistant subprocess (observed
        # on tech-writer); report "needs L3" rather than hang the request.
        try:
            await asyncio.wait_for(agent.clear(), timeout=20.0)
            notes.append("history cleared; CLI subprocess will respawn on next turn")
            ok = True
        except (asyncio.TimeoutError, TimeoutError):
            notes.append(
                "clear() timed out after 20s — agent likely hard-wedged on a "
                "SIGINT-resistant subprocess; escalate to L3 (archive "
                "session.jsonl + full channel restart)."
            )
            ok = False
        except Exception as exc:
            notes.append(f"clear() failed: {type(exc).__name__}: {exc}")
            ok = False
        # Re-anchor: push the role-specific first turn (+ orientation block).
        if ok and mode == "reanchor":
            try:
                agent.runtime.inbox.push_back(
                    UserMessage(text=_reanchor_prompt(role, recent_block))
                )
                notes.append(
                    "pushed re-anchor prompt (clear tape: last turns + "
                    "re-sync directive)"
                )
            except Exception as exc:
                notes.append(f"re-anchor push failed: {type(exc).__name__}: {exc}")
        return JSONResponse(
            {
                "ok": ok,
                "role": role,
                "mode": mode,
                "output": "\n".join(notes),
            }
        )

    async def interrupt(request: Request) -> Response:
        """Soft interrupt: Halt one agent's in-flight model turn WITHOUT
        wiping history (unlike ``/api/restart`` → ``Agent.clear()``).

        Pushes a ``Halt`` via the public ``Agent.halt()``: the runtime
        cancels the in-flight ``model_call``; the AnthropicCLI provider's
        ``asyncio.CancelledError`` handler SIGINTs the mid-turn
        ``claude --print`` subprocess so the opaque internal tool loop
        stops; the runtime then drains any operator/peer message buffered
        during the turn (``_mid_stream_queue``) and the agent acts on it.

        This restores the graceful "redirect a mid-flight agent"
        capability that the removed per-message ``urgent`` flag used to
        gate (default stays non-interrupting; this is the explicit opt-in
        escape hatch). It is ENTIRELY host-side — the Halt machinery and
        the cancellable CLI stream already exist in sagent; only the
        trigger was removed upstream, and this re-adds it without any
        sagent change.

        Typical redirect: POST the new instruction to ``/api/post`` (it
        buffers while the turn runs), then POST here to abort the current
        turn so the buffered message is acted on immediately. Loopback-only.
        """
        client_host = (request.client.host if request.client else "") or ""
        if client_host not in ("127.0.0.1", "::1", "localhost", ""):
            return JSONResponse(
                {"error": "interrupt is loopback-only"}, status_code=403
            )
        try:
            payload = await request.json()
        except Exception as exc:
            return JSONResponse({"error": f"bad json: {exc}"}, status_code=400)
        role = str(payload.get("role") or "").strip()
        if role not in agents:
            return JSONResponse(
                {"error": f"unknown role {role!r}", "known": sorted(agents)},
                status_code=404,
            )
        agent = agents[role]
        in_flight = agent.runtime.model_call is not None
        try:
            agent.halt()
            return JSONResponse(
                {
                    "ok": True,
                    "role": role,
                    "was_in_flight": in_flight,
                    "output": (
                        "halted in-flight turn (CLI subprocess SIGINT'd); "
                        "history preserved; buffered messages drain next"
                        if in_flight
                        else "agent idle; nothing in flight to halt"
                    ),
                }
            )
        except Exception as exc:
            return JSONResponse(
                {"ok": False, "role": role, "error": f"{type(exc).__name__}: {exc}"},
                status_code=500,
            )

    def _read_all_records() -> list[dict[str, Any]]:
        if not _MAIN_JSONL.exists():
            return []
        out: list[dict[str, Any]] = []
        with open(_MAIN_JSONL, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec, dict):
                    out.append(rec)
        return out

    async def list_messages(request: Request) -> Response:
        qp = request.query_params
        records = _read_all_records()

        # ``around=N&ctx=K`` returns a window of K records before and after
        # index N, plus offset/total/hit. Used by the debug page's
        # "show context" tab — identical contract to chat serve.
        if "around" in qp:
            total = len(records)
            try:
                around = int(qp["around"])
            except ValueError:
                around = total - 1
            try:
                ctx = max(0, min(50, int(qp.get("ctx", "5"))))
            except ValueError:
                ctx = 5
            start = max(0, around - ctx)
            end = min(total, around + ctx + 1)
            return JSONResponse(
                {
                    "records": records[start:end],
                    "offset": start,
                    "total": total,
                    "hit": around,
                }
            )

        since = qp.get("since")
        try:
            limit = max(1, min(2000, int(qp.get("limit", "200"))))
        except ValueError:
            limit = 200
        if since:
            records = [r for r in records if r.get("ts", "") > since]
        if len(records) > limit:
            records = records[-limit:]
        return JSONResponse({"records": records})

    async def list_agents(_request: Request) -> Response:
        # Per-agent liveness, activity, diagnosis, recent trace, inflight
        # tool, and pending-queue preview. Returns the **list** shape that
        # debug.html's render loop iterates over (chat-serve parity).
        from datetime import datetime

        items = [_diagnose_agent(label, agent) for label, agent in agents.items()]
        items.sort(key=lambda d: d["role"])
        now = datetime.now(UTC)
        now_iso = now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"
        return JSONResponse({"agents": items, "now": now_iso})

    async def get_trace(request: Request) -> Response:
        """Per-role runtime-event trace.

        Mirrors chat serve's ``GET /api/trace/<role>``:

          GET /api/trace/<role>              → last 500 events
          GET /api/trace/<role>?around=N&ctx=K → window of K events on each side
                                                  of index N + offset/total/hit
        """
        from .runtime import trace_writer

        role = request.path_params["role"]
        if role not in agents:
            return JSONResponse(
                {"error": f"unknown role {role!r}", "known": sorted(agents)},
                status_code=404,
            )
        path = trace_writer.trace_path_for(role)
        if not path.exists():
            return JSONResponse({"events": [], "total": 0})

        events: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        # ``total`` is ALWAYS the on-disk count (post-fix 2026-06-03).
        # The previous shape returned ``total = len(events)`` AFTER
        # tail-truncation to the limit, so once a trace file grew past
        # ``limit`` events, the reported total froze at ``limit`` and
        # the frontend's "did anything new arrive?" check
        # (``total > traceTotal``) never fired -- panel stuck.
        total_on_disk = len(events)
        qp = request.query_params
        if "around" in qp:
            try:
                around = int(qp["around"])
            except ValueError:
                around = total_on_disk - 1
            try:
                ctx = max(0, min(200, int(qp.get("ctx", "14"))))
            except ValueError:
                ctx = 14
            start = max(0, around - ctx)
            end = min(total_on_disk, around + ctx + 1)
            return JSONResponse(
                {
                    "events": events[start:end],
                    "offset": start,
                    "total": total_on_disk,
                    "hit": around,
                }
            )

        # Default: tail of the last N events. Cap raised from 500
        # → 2000 so a busy multi-hour TL session fits comfortably.
        try:
            limit = max(1, min(20000, int(qp.get("limit", "2000"))))
        except ValueError:
            limit = 2000
        sliced = events[-limit:] if total_on_disk > limit else events
        return JSONResponse(
            {
                "events": sliced,
                "total": total_on_disk,
                "returned": len(sliced),
            }
        )

    async def post(request: Request) -> Response:
        """Operator ingress + cross-process peer routing.

        Body shape: ``{to, body, from?, urgent?}``.

          - ``from`` absent or ``"user"``: operator ingress. Writes a
            ``user → [to]`` audit record and pushes ``UserMessage``
            into the target's inbox (existing web-UI behavior).
          - ``from`` is a known agent label: peer routing called by
            the plugin's MCP server. Writes a ``<from> → [to]`` audit
            record and pushes ``AgentSendMessage(source=<from>)`` into
            the target's inbox. This is how the out-of-process MCP
            server (subprocess of ``claude --print``) reaches the
            in-process ``agent_registry`` that ``serve.py`` owns.
            Respects the ``_SUPPRESS_FLAG`` sentinel for warmup-window
            silence.
        """
        from sagent.tools.core import agent_registry
        from sagent.types.runtime import AgentSendMessage, UserMessage

        payload = await request.json()
        to = str(payload.get("to", "")).strip()
        body = str(payload.get("body", ""))
        from_role = str(payload.get("from", "user")).strip() or "user"
        urgent = bool(payload.get("urgent", False))
        if not to or not body:
            return JSONResponse(
                {"error": "both 'to' and 'body' are required"}, status_code=400
            )
        target = agent_registry.get(to)
        if target is None:
            return JSONResponse(
                {"error": f"unknown target {to!r}; active: {sorted(agents)}"},
                status_code=404,
            )
        # User-as-recipient is a "mailbox without listener" — audit
        # log is the visible surface; no inbox push.
        suppress = _SUPPRESS_FLAG.exists()
        if from_role == "user":
            if to == "user":
                return JSONResponse(
                    {"error": "user→user is meaningless; pick an agent"},
                    status_code=400,
                )
            if not suppress:
                delivery.append_user_message(to_role=to, body=body)
            target.runtime.inbox.push_back(
                UserMessage(text=body),
            )
        else:
            # Peer routing path called by the MCP server.
            if not suppress:
                delivery.append_record(from_role=from_role, to=[to], body=body)
            if to != "user":
                target.runtime.inbox.push_back(
                    AgentSendMessage(
                        source=from_role,
                        text=body,
                    ),
                )
        # ``urgent`` — DISABLED 2026-06-19 (queues only; does NOT preempt).
        # Intended behaviour was to ``Agent.halt()`` the recipient's in-flight
        # turn so it acts on this message now. But halt() SIGINTs the CLI
        # subprocess mid-MCP-session and leaves the streamable-http bridge wedged:
        # every later turn fails "MCP bridge catalog not fetched", recoverable only
        # by a full restart (observed: swe 10:22, tech-writer 13:44 UTC). The CLI
        # session-persistent cancel path SIGINTs the subprocess but doesn't reset
        # the bridge (unlike the stateless path) — but whether that's an upstream
        # bug or intended is unresolved: the maintainer notes API-mode interrupts
        # deliberately DON'T halt (messages stack), so no-preempt may be by design.
        # Either way, calling halt() here wedges us, so we do NOT: the message is
        # still delivered (queued above), it just won't preempt. Re-enable the halt
        # call only once upstream confirms+fixes the CLI bridge-reset. See worklog
        # 2026-06-19 large-tape-mcp-connect-wedge lesson + the halt-wedge decision.
        if urgent and to != "user":
            _LOG.warning(
                "urgent=true to %r: halt is DISABLED (wedges the MCP bridge, "
                "pending upstream fix) — message queued, not preempted",
                to,
            )
        return JSONResponse(
            {
                "ok": True,
                "to": to,
                "from": from_role,
                "urgent": urgent,
                "was_in_flight": None,  # halt disabled; nothing is ever preempted
                "urgent_halt_disabled": bool(urgent),
            }
        )

    async def defer(request: Request) -> Response:
        """Operator-side + MCP-server-side wake-up scheduling.

        Body: ``{to, body, delay_s, from?}``. Defaults ``from="operator"``.

        Schedules an ``asyncio.call_later`` that pushes
        ``AgentSendMessage(source=<from>, text='[defer +Ns] <body>')``
        into the target's inbox after ``delay_s`` seconds. Audit log
        receives one record at schedule time with body
        ``[defer +Ns scheduled] <body>``. Sentinel-respecting same as
        ``/api/post``.
        """
        from sagent.tools.core import agent_registry
        from sagent.types.runtime import AgentSendMessage

        client_host = (request.client.host if request.client else "") or ""
        if client_host not in ("127.0.0.1", "::1", "localhost", ""):
            return JSONResponse(
                {"error": "defer disabled: not a loopback client"},
                status_code=403,
            )
        try:
            payload = await request.json()
        except Exception as exc:
            return JSONResponse({"error": f"bad json: {exc}"}, status_code=400)
        to = str(payload.get("to", "")).strip()
        body = str(payload.get("body", "")).strip()
        from_role = str(payload.get("from", "operator")).strip() or "operator"
        try:
            delay_s = int(payload.get("delay_s", 0))
        except (TypeError, ValueError):
            return JSONResponse({"error": "delay_s must be an int"}, status_code=400)
        if not to or not body:
            return JSONResponse(
                {"error": "both 'to' and 'body' are required"}, status_code=400
            )
        if delay_s < 1 or delay_s > 86400:
            return JSONResponse(
                {"error": "delay_s must be in [1, 86400]"}, status_code=400
            )
        target = agent_registry.get(to)
        if target is None or to == "user":
            return JSONResponse(
                {"error": f"unknown target {to!r}; active: {sorted(agents)}"},
                status_code=404,
            )

        marked = f"[defer +{delay_s}s] {body}"

        def _fire() -> None:
            tgt = agent_registry.get(to)
            if tgt is None:
                _LOG.warning("defer: target @%s gone from registry; dropping", to)
                return
            tgt.runtime.inbox.push_back(
                AgentSendMessage(source=from_role, text=marked),
            )

        asyncio.get_running_loop().call_later(delay_s, _fire)
        if not _SUPPRESS_FLAG.exists():
            delivery.append_record(
                from_role=from_role,
                to=[to],
                body=f"[defer +{delay_s}s scheduled] {body}",
            )
        return JSONResponse(
            {"ok": True, "to": to, "from": from_role, "delay_s": delay_s}
        )

    return Starlette(
        debug=False,
        routes=[
            Route("/", index, methods=["GET"]),
            Route("/debug", debug_index, methods=["GET"]),
            Route("/debug.html", debug_index, methods=["GET"]),
            # Canonical /api/* paths (chat-serve parity).
            Route("/api/roles", list_roles, methods=["GET"]),
            Route("/api/agents", list_agents, methods=["GET"]),
            Route("/api/messages", list_messages, methods=["GET"]),
            Route("/api/trace/{role}", get_trace, methods=["GET"]),
            Route("/api/search", search, methods=["GET"]),
            Route("/api/post", post, methods=["POST"]),
            Route("/api/defer", defer, methods=["POST"]),
            Route("/api/restart", restart, methods=["POST"]),
            Route("/api/interrupt", interrupt, methods=["POST"]),
            # Backwards-compat aliases for the inline viewer's current
            # poll URLs; can be removed after the viewer is updated.
            Route("/messages", list_messages, methods=["GET"]),
            Route("/agents", list_agents, methods=["GET"]),
            Route("/send", post, methods=["POST"]),
        ],
    )


# --------------------------------------------------------------------------
# Main entry
# --------------------------------------------------------------------------


async def _amain(host: str, port: int) -> int:
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Expose the port to the per-role MCP server entries built inside
    # ``_build_all_agents``: each role embeds ``SAGENT_HTTP_URL=http://
    # 127.0.0.1:<port>`` into the spawned MCP server's env, and that
    # URL is where ``mcp_sagent.delivery`` POSTs ``/api/post`` /
    # ``/api/defer``. The MCP server runs in a separate Python
    # process; this env var is the only handshake.
    os.environ["SAGENT_HTTP_PORT"] = str(port)

    from .team_profile import load_profile

    profile = load_profile()
    _LOG.info(
        "loaded profile %r (namespace=%s) roster=%s",
        profile.name,
        profile.session_id_namespace,
        profile.roster,
    )

    # Big resumed tapes slow the first-turn full-history re-feed to the CLI,
    # which can push the MCP-catalog connect past sagent's default 8s ceiling
    # (anthropic_cli._MCP_CONNECT_TIMEOUT_SEC) and raise SubprocessTransportError
    # on a cold turn -> respawn loop (observed 2026-06-19, swe). Raise that
    # ceiling host-side (module constant, read at call time) so a slow cold
    # start tolerates the load instead of erroring. Env-tunable; the real fix is
    # smaller tapes (earlier compaction, see roles/common.py).
    import sagent.providers.anthropic_cli as _acli

    _mcp_to = float(os.environ.get("AGENT_TEAM_MCP_CONNECT_TIMEOUT_SEC", "25"))
    if _mcp_to != _acli._MCP_CONNECT_TIMEOUT_SEC:
        _LOG.info(
            "MCP-connect ceiling: %.0fs (sagent default %.1fs)",
            _mcp_to,
            _acli._MCP_CONNECT_TIMEOUT_SEC,
        )
        _acli._MCP_CONNECT_TIMEOUT_SEC = _mcp_to

    agents = _build_all_agents(profile)
    _LOG.info("brought up %d agents: %s", len(agents), sorted(agents))

    # Restart-resume: rehydrate each agent's tape from its sagent
    # ``session.jsonl`` BEFORE it starts serving (and before warmup).
    # See AGENT_TEAM_RESUME_MODE (full|slim|materialize).
    if _RESUME_MODE not in _VALID_RESUME_MODES:
        _LOG.warning(
            "AGENT_TEAM_RESUME_MODE=%r invalid; using 'slim' (valid: %s)",
            _RESUME_MODE,
            "|".join(_VALID_RESUME_MODES),
        )
    # materialize mode is gated by a one-shot boot drift-canary: spawn a
    # throwaway claude, materialize a session, structurally diff vs claude's
    # own output. On drift (claude changed its session format) we DISABLE
    # materialize for this boot and fall back to slim, logging the findings.
    _materialize_enabled = False
    if _RESUME_MODE == "materialize":
        try:
            from agent_team.cli_session import (
                arun_canary_against_live_cli,
                is_safe_to_enable,
            )

            _LOG.info("resume-mode=materialize: running boot drift-canary…")
            _canary = await arun_canary_against_live_cli()
            if is_safe_to_enable(_canary.findings):
                _materialize_enabled = True
                _LOG.info(
                    "drift-canary PASSED (cli %s) — materialize-on-resume enabled",
                    _probe_claude_version(),
                )
            else:
                _LOG.warning(
                    "drift-canary FOUND DRIFT (%d findings); FALLING BACK TO "
                    "SLIM for this boot:",
                    len(_canary.findings),
                )
                for _f in _canary.findings:
                    _LOG.warning(
                        "  drift %s: %s",
                        getattr(_f, "location", "?"),
                        getattr(_f, "detail", "?"),
                    )
        except Exception as exc:
            _LOG.warning(
                "drift-canary errored (%s: %s); falling back to slim for this boot",
                type(exc).__name__,
                exc,
            )
    _resume_agents_from_session_dir(agents, materialize_enabled=_materialize_enabled)

    serve_tasks = await _serve_agents_forever(agents)

    # Startup warmup — empirically required for the AnthropicCLI
    # provider. Without it, the very first model_call after the
    # ``claude --print`` subprocess spawns hangs indefinitely
    # (observed 2026-06-01: >5 min no reply), the same first-turn
    # MCP-tool-fumble pattern we documented in the Phase 0 spike.
    # The warmup is a "hello → AgentSend(to=user, content='ready')"
    # round-trip per agent, structured so the model is conditioned on
    # the SAME ``AgentSend(to=user)`` template we want it to use for
    # real replies. The cost is one ``hello, ready`` record per agent
    # in ``main.jsonl`` at startup — visible noise but small and
    # easily filtered.
    _LOG.info("warming up agents (≤90s)…")
    warmup_results = await _warmup_agents(agents)
    ok = [l for l, v in warmup_results.items() if v]
    timed_out = [l for l, v in warmup_results.items() if not v]
    _LOG.info("warmup done: ready=%s timed_out=%s", sorted(ok), sorted(timed_out))

    app = _build_http_app(agents)
    config = uvicorn.Config(
        app, host=host, port=port, log_level="warning", access_log=False
    )
    server = uvicorn.Server(config)
    _LOG.info("starting HTTP server on http://%s:%d", host, port)
    http_task = asyncio.create_task(server.serve(), name="http")

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    await stop.wait()
    _LOG.info("shutdown signal received; stopping HTTP server")
    server.should_exit = True
    await http_task
    _LOG.info("stopping agents")
    for agent in agents.values():
        agent.shutdown()
    await asyncio.gather(*serve_tasks, return_exceptions=True)
    _LOG.info("clean exit")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="agent-team",
        description=__doc__.split("\n\n")[0],
    )
    # Optional ``serve`` verb so both ``agent-team`` and ``agent-team serve``
    # work (the latter is the documented launch form; there is only one command
    # today, so the verb is accepted but optional). ``--help`` exits 0 via
    # argparse BEFORE any agent/server boot.
    ap.add_argument(
        "command",
        nargs="?",
        default="serve",
        choices=["serve"],
        help="command to run (default: serve)",
    )
    ap.add_argument("--host", default=_DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=_DEFAULT_PORT)
    args = ap.parse_args(argv)
    if args.host not in ("127.0.0.1", "::1", "localhost"):
        print(
            f"refusing to bind public host {args.host!r}; only 127.0.0.1/::1/localhost allowed",
            file=sys.stderr,
        )
        return 2
    try:
        return asyncio.run(_amain(args.host, args.port))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
