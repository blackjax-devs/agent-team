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

1. Connects one peer-messaging MCP server (``mcp_sagent/server.py``) per
   role, with ``SAGENT_ROLE=<label>`` in its env, through sagent's
   ``McpSession`` — so peer messaging, self-defer, and status-report all flow
   through ``mcp__sagent_chat__*`` tool calls instead of sagent's own
   ``AgentSend``/``AgentSelf``.
2. Constructs the roster's role agents via the per-role ``build()``
   factories, each on its session journal, and opens them: an agent with a
   prior session resumes it (see ``AGENT_TEAM_RESUME_MODE``).
3. Keeps the agents by role label; ``user`` + ``system`` are addressable
   mailboxes without an agent. The web UI is the user-visible surface for
   ``user``-targeted messages (read from ``main.jsonl``).
4. Touches the ``_suppress_audit`` sentinel, fires the bootstrap probe
   (one warm-up turn per agent), waits for AgentIdle, then removes the
   sentinel — so the warmup turn writes zero audit log records and pushes
   zero peer messages.
5. Starts a Starlette+uvicorn HTTP server on the configured host:port.

Shutdown via SIGTERM or Ctrl-C: the HTTP server stops first, then
each agent closes, writing its session.
"""

from __future__ import annotations

from collections.abc import Sequence

import argparse
import asyncio
import dataclasses
import json
import logging
import os
import signal
import sys
import time


# Pulls in the data-dir resolution (env-configurable). All file paths
# below derive from ``delivery.DATA_DIR`` / ``delivery.SESSIONS_DIR``
# so a single ``SAGENT_DATA_DIR`` env var co-locates the audit log,
# trace files, per-role mcp.json, sentinel, and debug log — useful
# for end-of-day merges that union this plugin's main.jsonl with the
# sibling chat/ runtime's main.jsonl in one directory.
from contextlib import AsyncExitStack
from datetime import UTC, datetime
from importlib.resources import files as _pkg_files
from typing import Any

from .mcp_sagent import delivery


_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8767
_MAIN_JSONL = delivery.MAIN_JSONL_PATH

# Sentinel file: when present, the plugin's MCP server skips audit
# log writes AND peer inbox pushes. Set during the bootstrap warmup
# window. Path must match :data:`mcp_sagent.server._SUPPRESS_FLAG`.
_SUPPRESS_FLAG = delivery.SESSIONS_DIR / "_suppress_audit"

# Resume-slim: on every server restart, trim each agent's resumed context to
# ~the last N conversation records (snapped to a turn boundary) BEFORE the agent
# reads it. Deterministic (no model call), so a large session can't stall the
# first turn after a restart — the fat-tape boot wedge that took down TL
# (855 records / 1.97 MB) after a failed live compact. ON by default; set to 0
# to disable (resume the full window), or raise to keep more recent context.
# The on-disk JSONL is untouched — only what the agent resumes with is bounded,
# and every restart re-slims from the full physical history.
_RESUME_KEEP_RECORDS = int(os.environ.get("AGENT_TEAM_RESUME_KEEP", "120"))

# Resume strategy (AGENT_TEAM_RESUME_MODE): how an agent's persisted session is
# brought back on a server restart.
#   slim — trim the journal's last window to its recent tail
#          (`_RESUME_KEEP_RECORDS`) then resume; the first turn re-feeds the
#          trimmed tail. Deterministic, no model call. THE DEFAULT.
#   full — resume the untrimmed window; the first turn re-feeds all of it.
#          Escape hatch.
# `materialize` (the default before the port to sagent 0.2) is gone: it wrote
# claude's own session file and flipped the provider to a native `--resume`.
# sagent 0.2 runs `claude --no-session-persistence` and never reads a claude
# session, so there is nothing to materialize; the value now selects `slim`,
# which was its fallback.
_VALID_RESUME_MODES = ("full", "slim")


def _resolve_resume_mode(raw: str) -> tuple[str, str]:
    """Return the resume mode ``raw`` selects, and a warning (``""`` for none)."""
    mode = raw.strip().lower()
    if mode in _VALID_RESUME_MODES:
        return mode, ""
    if mode == "materialize":
        return "slim", (
            "AGENT_TEAM_RESUME_MODE=materialize is not available on sagent>=0.2 "
            "(claude session files are no longer resumed); using 'slim'"
        )
    return "slim", (
        f"AGENT_TEAM_RESUME_MODE={raw!r} invalid; using 'slim' "
        f"(valid: {'|'.join(_VALID_RESUME_MODES)})"
    )


_RESUME_MODE, _RESUME_MODE_WARNING = _resolve_resume_mode(
    os.environ.get("AGENT_TEAM_RESUME_MODE", "slim")
)

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


# Labels that are addressable but are not agents. ``user`` is a
# mailbox-without-listener: ``sagent_send(to='user', …)`` lands in the audit
# log, which the web UI shows. ``system`` is reserved for diagnostic probes /
# future eventing.
_MAILBOXES = ("user", "system")


def _now_iso() -> str:
    """The time now, local with its UTC offset — how sagent stamps its records."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _is_inbound(record) -> bool:
    """Whether ``record`` is a user-role message: a clean turn boundary."""
    from sagent.session.types import AgentToAgentMessage, UserMessage

    return isinstance(record, (UserMessage, AgentToAgentMessage))


def _is_conversation(record) -> bool:
    """Whether ``record`` is part of what the model sees (so a slim may drop it).

    A window also holds bookkeeping the resume needs whole — the model and
    system prompt in force (``UncategorizedRecord``) and what each turn cost
    (``TokenUsage``) — which a slim always keeps.
    """
    from sagent.session.types import ContextClear, TokenUsage, UncategorizedRecord

    return not isinstance(record, (ContextClear, TokenUsage, UncategorizedRecord))


def _epoch(record) -> float | None:
    """``record.timestamp`` (ISO 8601) as seconds since the epoch, if it has one."""
    stamp = getattr(record, "timestamp", None)
    if not isinstance(stamp, str):
        return None
    try:
        return datetime.fromisoformat(stamp).timestamp()
    except ValueError:
        return None


def _inbound_times(body) -> dict[int, float]:
    """Wall-clock time of each inbound message in ``body``, by index.

    A message sagent held back until the agent was free is journaled without a
    timestamp of its own; it is dated by the first stamped record after it (the
    ``TokenUsage`` of the reply it drew, seconds later).
    """
    times: dict[int, float] = {}
    later: float | None = None
    for i in range(len(body) - 1, -1, -1):
        own = _epoch(body[i])
        if own is not None:
            later = own
        if _is_inbound(body[i]) and later is not None:
            times[i] = later
    return times


def _split_window(window):
    """Split a journal window into its opening ``ContextClear`` and what follows.

    The records a compaction kept (``ContextClear.history``) are part of the
    conversation, so they lead the body. Returns ``(head, body)``; ``head`` is
    ``None`` for a window that does not open with a ``ContextClear``.
    """
    from sagent.session.types import ContextClear

    records = list(window)
    if records and isinstance(records[0], ContextClear):
        return records[0], [*records[0].history, *records[1:]]
    return None, records


def _cut_window(head, body, cut):
    """Return the window that keeps ``body``'s conversation from index ``cut`` on.

    The opening ``ContextClear`` is kept without the summary and history it
    carried (both lie before the cut), and so is every bookkeeping record.
    """
    kept = [r for i, r in enumerate(body) if i >= cut or not _is_conversation(r)]
    if head is None:
        return kept
    return [dataclasses.replace(head, summary=None, history=()), *kept]


def _slim_resume_tape(window, keep):
    """Trim a journal window to ~its last ``keep`` conversation records.

    Keeps the last ``keep`` records the model sees, snapped FORWARD to the first
    inbound user-role message (``UserMessage`` or ``AgentToAgentMessage`` — both
    render as user-role and are clean turn boundaries) so the slice starts
    cleanly — no dangling tool result whose call was cut. A compaction's summary
    lies before any such cut and is dropped with what it summarized.

    Deterministic, no model call — this is the whole point: a fat context can't
    stall the first turn after a restart the way a live compaction can (and did,
    on TL). Returns ``(slimmed, dropped)``; ``keep<=0`` disables.
    """
    head, body = _split_window(window)
    spoken = [i for i, r in enumerate(body) if _is_conversation(r)]
    # Small window (or disabled): leave it completely untouched — no risk worth
    # taking, and the re-feed is already cheap.
    if keep <= 0 or len(spoken) <= keep:
        return window, 0
    cut = spoken[-keep]
    for i in spoken[-keep:]:
        if _is_inbound(body[i]):
            cut = i
            break
    return _cut_window(head, body, cut), sum(1 for i in spoken if i < cut)


def _inbound_floor_ts(records):
    """Earliest wall-clock time among inbound (user-role) messages.

    Inbound messages are the cross-agent sync points. The min over a slimmed
    lead window is the lead's recent-window floor: the anchor every other agent
    is trimmed to. Returns ``None`` when no inbound message can be dated
    (caller falls back to per-agent slim).
    """
    _head, body = _split_window(records)
    times = _inbound_times(body)
    return min(times.values()) if times else None


def _slim_tape_to_floor(window, floor_ts):
    """Keep a window's conversation from the first inbound message at/after ``floor_ts``.

    Drops everything older than the shared floor, starting the slice at a clean
    user-role turn boundary. When the agent has been **idle since before the
    floor** (no inbound at/after it), keep ONLY its most recent inbound turn — a
    breadcrumb, not the full stale window that drives the churn (an idle agent
    resuming an abandoned task). Returns ``None`` only when there is no inbound
    message at all (degenerate; the caller keeps the per-agent record slim).
    """
    head, body = _split_window(window)
    inbound = [i for i, r in enumerate(body) if _is_inbound(r)]
    if not inbound:
        return None
    times = _inbound_times(body)
    for i in inbound:
        if times.get(i, float("-inf")) >= floor_ts:
            return _cut_window(head, body, i)
    # idle since before the floor → keep just the most recent turn.
    return _cut_window(head, body, inbound[-1])


def _plan_slim_resume(loaded):
    """Compute the window each agent resumes from.

    ``loaded`` maps ``label -> window`` (the records of the journal's last
    context window). Returns ``label -> (window, note)``.

    - ``full``: resume the whole window.
    - ``slim``: **TL-anchored**. Slim the lead by record count to fix its recent
      window, take that window's wall-clock floor, and slim every other agent to
      that same floor — one shared temporal horizon, so no agent retains work
      the lead has forgotten (the desync). Agents idle since before the floor,
      or any case with no derivable anchor, fall back to a per-agent record slim
      (never a regression vs the pre-anchor behaviour).
    """
    if _RESUME_MODE != "slim":
        return {label: (window, "") for label, window in loaded.items()}

    keep = _RESUME_KEEP_RECORDS
    plan: dict = {}
    floor = None
    lead = _RESUME_LEAD_ROLE
    if lead and lead in loaded:
        lead_slim, _ = _slim_resume_tape(loaded[lead], keep)
        floor = _inbound_floor_ts(lead_slim)
        plan[lead] = (
            lead_slim,
            f" lead-anchor floor={floor:.0f}" if floor else " lead (no anchor)",
        )

    for label, window in loaded.items():
        if label in plan:
            continue
        if floor is not None:
            anchored = _slim_tape_to_floor(window, floor)
            if anchored is not None:
                plan[label] = (anchored, " anchored-to-lead")
                continue
        slimmed, _ = _slim_resume_tape(window, keep)
        plan[label] = (slimmed, " per-agent")
    return plan


class _SlimmedJournal:
    """A journal whose resume reads a trimmed window; everything else is the file's.

    sagent resumes an agent from ``iter(journal)`` — the last context window.
    This hands it the slimmed window instead, while every write, and
    ``history()``, still goes to the journal on disk: the file is untouched, so
    the next restart re-slims from the full physical history.
    """

    def __init__(self, journal, window) -> None:
        self._journal = journal
        self._window = list(window)

    @property
    def path(self):
        return self._journal.path

    def append(self, *records) -> None:
        from sagent.session.types import ContextClear

        self._journal.append(*records)
        for record in records:
            if isinstance(record, ContextClear):
                self._window = []
            self._window.append(record)

    def __iter__(self):
        return iter(tuple(self._window))

    def __len__(self) -> int:
        return len(self._journal)

    def history(self, start: int = 0):
        return self._journal.history(start)

    def flush(self) -> None:
        self._journal.flush()


def _session_journal(label: str):
    """Return ``label``'s (unopened) session journal, where a restart resumes from.

    A log an older sagent (<0.2) wrote at the same path is first rewritten in
    place as today's journal — sagent keeps the original beside it as
    ``session.legacy.jsonl``. One that cannot be rewritten is moved aside and
    the agent starts fresh; boot never fails on it.
    """
    from sagent.session import legacy_journal
    from sagent.session.jsonl import JsonlJournal

    from .roles.common import journal_path_for

    path = journal_path_for(label)
    try:
        if legacy_journal.is_legacy(path):
            if legacy_journal.upgrade(path):
                _LOG.info("resume %s: upgraded the pre-0.2 session log", label)
            else:
                raise ValueError("it holds nothing to resume")
    except Exception as exc:
        aside = path.with_name(f"session.unreadable-{int(time.time())}.jsonl")
        path.replace(aside)
        _LOG.warning(
            "resume %s: pre-0.2 session log not upgraded (%s: %s); moved to %s, "
            "agent starts fresh",
            label,
            type(exc).__name__,
            exc,
            aside,
        )
    return JsonlJournal(path)


def _plan_resume(journals):
    """Return the journal each agent resumes from, slimmed as the mode says.

    Standard sagent restart-resume: each role's agent is given its journal
    (``<SAGENT_DATA_DIR>/sessions/<role>.sagent/session.jsonl``) and, when it
    opens, loads the journal's last context window; its first turn then tells a
    fresh ``claude`` that conversation. sagent's journal is the single source of
    truth — no claude session file is read or written.

    Mode (``AGENT_TEAM_RESUME_MODE``): ``slim`` trims each window to its recent
    tail so that first-turn re-feed stays small, and **anchors the trim across
    agents on the lead** (``_plan_slim_resume``) so a restart can't desync the
    team. ``full`` resumes the untrimmed window. The on-disk journal is
    untouched in every mode.

    Best-effort and never crashes boot: if the plan cannot be computed, every
    agent resumes its whole window.
    """
    loaded = {label: list(journal) for label, journal in journals.items()}
    try:
        plan = _plan_slim_resume(loaded)
    except Exception as exc:
        _LOG.warning(
            "resume: slim plan failed (%s: %s); resuming every agent in full",
            type(exc).__name__,
            exc,
        )
        plan = {label: (window, "") for label, window in loaded.items()}

    resumed = {}
    for label, journal in journals.items():
        window, note = plan[label]
        full = loaded[label]
        if not full:
            _LOG.info("resume %s: no prior session.jsonl — fresh start", label)
            resumed[label] = journal
        elif len(window) != len(full):
            _LOG.info(
                "resume %s [%s]: %d -> %d journal records%s",
                label,
                _RESUME_MODE,
                len(full),
                len(window),
                note,
            )
            resumed[label] = _SlimmedJournal(journal, window)
        else:
            _LOG.info(
                "resume %s [%s]: %d journal records%s",
                label,
                _RESUME_MODE,
                len(full),
                note,
            )
            resumed[label] = journal
    return resumed


async def _build_all_agents(profile, stack):
    """Construct, resume and open the roster's role agents; return them by label.

    Only the roles in ``profile.roster`` are instantiated (1..N). Each agent
    gets the tools of its own peer-messaging MCP server, started with
    ``SAGENT_ROLE=<label>`` in the server's env — sagent_send/sagent_defer/
    sagent_self reach the model as ``mcp__sagent_chat__*`` and are the only
    path for peer messaging. A single-role roster boots fine: there are simply
    no peers to address (only ``user``).

    Everything opened here — the shared provider, each journal, each MCP
    session, each agent — is entered on ``stack``, which the caller closes on
    shutdown (agents first, so each writes its session before its journal
    closes).
    """
    from .roles.common import build_provider, chat_session
    from .runtime import trace_writer

    builders = _role_builders()
    for label in profile.roster:
        if label not in builders:
            raise ValueError(
                f"roster references unknown role {label!r}; "
                f"known roles: {sorted(builders)}"
            )

    provider = await stack.enter_async_context(build_provider())
    journals = {}
    for label in profile.roster:
        journals[label] = await stack.enter_async_context(_session_journal(label))
    resumed = _plan_resume(journals)

    agents = {}
    for label in profile.roster:
        chat = await stack.enter_async_context(chat_session(label))
        chat_tools = chat.tools()
        if not chat_tools:
            raise RuntimeError(
                f"peer-messaging MCP server for {label!r} did not connect "
                f"({chat.status()}); the agent could not message anyone"
            )
        agent = builders[label](
            profile,
            provider=provider,
            journal=resumed[label],
            chat_tools=chat_tools,
        )
        trace_writer.install_on(agent, label)
        # Opening resumes the journal's window and starts the agent's loop.
        await stack.enter_async_context(agent)
        agents[label] = agent
    return agents


# --------------------------------------------------------------------------
# Startup warmup — fire one MCP-tool-call-using turn per agent before
# accepting user traffic, so the first real user message doesn't pay the
# ``claude`` cold start or hit the "first-turn fumbles MCP tool calls"
# pattern observed in spike sigint_probe3 and race_repro_v2.
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
    from sagent.runtime.types import AgentIdle
    from sagent.session.types import UserMessage

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
            agent.runtime.push(
                UserMessage(content=_WARMUP_PROMPT, timestamp=_now_iso())
            )

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
        # DO NOT clear the session here. A cleared history no longer
        # matches what the warm ``claude`` subprocess has seen, so the
        # provider starts a fresh one — undoing exactly the warmup we
        # just paid for. Leave the warmup turn in history; the prompt's
        # explicit "do NOT reuse this pattern" framing + the use of
        # ``sagent_self`` (silent to peers) means even a
        # parrot-after-respawn doesn't leak.
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
      - working: a model call or a tool call it waits on is in flight
      - hung   : in flight AND last assistant turn >90s ago (or never)
      - stuck  : inbox has queued messages AND nothing in flight AND no recent activity
      - idle   : nothing in flight and inbox empty
    """
    # The tool calls the model is waiting on. sagent runs them itself (the
    # ``claude`` subprocess only holds the call), so no model call is in flight
    # while they run.
    tools_running = [
        job.call.name for job in agent.runtime.jobs.values() if job.state == "running"
    ]
    in_flight_call = agent.runtime.model_call is not None or bool(tools_running)
    inbox_pending = 0
    try:
        inbox_pending = agent.runtime.inbox._queue.qsize()
    except AttributeError:
        pass

    # Recent trace events (last 6) — sourced from the trace file rather
    # than agent.runtime.events to match chat/'s "tail the trace" check.
    try:
        from .runtime import trace_writer

        tp = trace_writer.trace_path_for(label)
        events = _read_jsonl(tp)
    except Exception:
        events = []

    # last_assistant_ts + age_sec: when the model last finished a reply, as the
    # trace recorded it (sagent's own records carry no wall-clock for a reply).
    last_ts_iso: str | None = None
    age_sec: int | None = None
    for ev in reversed(events):
        if (ev.get("_event") or "?") == "ModelResponseComplete":
            last_ts_iso = ev.get("_ts") or None
            try:
                spoke = datetime.fromisoformat(last_ts_iso.replace("Z", "+00:00"))
                age_sec = int(datetime.now(UTC).timestamp() - spoke.timestamp())
            except (AttributeError, ValueError):
                last_ts_iso = None
            break

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
    in_turn = started_idx > ended_idx or bool(tools_running)

    inflight: str | None = None
    if in_turn:
        inflight = ", ".join(tools_running) or "model thinking"

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
        model_id = agent.model.capability.model_id
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
            body = getattr(item, "content", None) or getattr(item, "text", "")
            text = (body or "").replace("\n", " ")[:140]
            src = getattr(item, "sender", "") or type(item).__name__
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
        "total_cost_usd": float(agent.spend_usd),
    }


def _render_recent_turns(
    agent, label: str, *, n_turns: int = 2, max_chars: int = 1800
) -> str:
    """Render the agent's last ``n_turns`` turns as a plain-text breadcrumb.

    A *turn* starts at an inbound message (an operator's ``UserMessage``, a
    peer's ``AgentToAgentMessage``, or the prior bootstrap) and runs through
    the assistant's reply. The
    ``clear tape`` restart hands this to the freshly-cleared agent as
    orientation — WITHOUT reconstructing tape records, which would risk an
    invalid post-barrier tape (dangling tool_result / broken ordinal refs).
    Tool results and other low-signal entries are dropped. Best-effort: any
    failure returns ``""`` so the restart never blocks on it.
    """
    try:
        from sagent.model.types import Response
        from sagent.session.types import AssistantMessage, ToolCall

        messages = list(agent.runtime.context().messages)
    except Exception:
        return ""
    user_idxs = [i for i, m in enumerate(messages) if _is_inbound(m)]
    if not user_idxs:
        return ""
    start = user_idxs[-n_turns] if len(user_idxs) >= n_turns else user_idxs[0]
    lines: list[str] = []
    for m in messages[start:]:
        if _is_inbound(m):
            txt = (m.content or "").strip()
            if txt:
                lines.append(f"[inbound] {txt[:400]}")
        elif isinstance(m, Response):
            txt = "".join(
                r.content or "" for r in m.records if isinstance(r, AssistantMessage)
            ).strip()
            if txt:
                lines.append(f"[{label}] {txt[:400]}")
            names = ", ".join(r.name for r in m.records if isinstance(r, ToolCall))
            if names:
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
        #   slim     -- the session's compactor: summarize old turns → keep a
        #               real summary + the recent working thread. Shrinks the
        #               effective re-feed (the new window is journaled, so a
        #               later boot resumes slim too) WITHOUT wiping context or
        #               interrupting the task. This is sagent's normal
        #               ``/compact`` path, run at the runtime's next safe
        #               point; it fails safe — the session is untouched on
        #               CompactFailed. For a long-but-healthy agent (e.g. TL)
        #               whose session is growing but whose coordination memory
        #               must survive.
        #   soft     -- ``session.clear()`` only, inbox preserved. Full wipe.
        #   reanchor -- capture a last-turns breadcrumb, drop the backlog,
        #               ``clear()``, then push a role-specific "re-sync before
        #               acting" directive. ``clear()`` starts a new journaled
        #               window so the re-feed is tiny — this kills the wedge a
        #               full-session re-feed triggers. For recovering a
        #               stuck/confused agent.
        from sagent.runtime import types as runtime_types
        from sagent.session.types import AgentToAgentMessage, UserMessage

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
                # ``run()`` returns on CompactComplete and raises what a
                # CompactFailed carried; the timeout guards a wedged compaction
                # subprocess. A large session's summary call can take a while,
                # hence the generous cap.
                compact_err: BaseException | None = None
                try:
                    await asyncio.wait_for(
                        agent.session.compactor.run(), timeout=180.0
                    )
                except (asyncio.TimeoutError, TimeoutError):
                    raise
                except Exception as exc:
                    compact_err = exc
                after = (
                    len(agent.runtime.context().messages)
                    if before is not None
                    else None
                )
                if compact_err is not None:
                    notes.append(
                        f"compaction failed: {type(compact_err).__name__}: "
                        f"{compact_err}; tape unchanged"
                    )
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
            # Drop the stale backlog: every message still waiting in the
            # inbox. Anything else queued there (a tool result, scheduled
            # work) is the runtime's own and goes back in order.
            backlog = (
                UserMessage,
                AgentToAgentMessage,
                runtime_types.UserQueuedMessage,
                runtime_types.UserDeferredMessage,
                runtime_types.AgentSendQueuedMessage,
                runtime_types.AgentSendDeferredMessage,
            )
            drained = 0
            for item in agent.runtime.inbox.drain_nowait():
                if isinstance(item, backlog):
                    drained += 1
                else:
                    agent.runtime.inbox.push_back(item)
            notes.append(f"drained {drained} pending inbox item(s)")

        # --- soft + reanchor: clear() (timeout-guarded) ----------------------
        # ``session.clear()`` cancels the in-flight model call and every
        # tool, wipes the history and forgets which files the model has read.
        # The provider then starts a fresh ``claude --print`` subprocess on
        # the next model call (the one running no longer matches the history).
        # Bounded by a timeout: a hard-wedged agent can block clear() on a
        # SIGINT-resistant subprocess (observed on tech-writer); report
        # "needs L3" rather than hang the request.
        try:
            await asyncio.wait_for(agent.session.clear(), timeout=20.0)
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
                agent.runtime.push(
                    UserMessage(
                        content=_reanchor_prompt(role, recent_block),
                        timestamp=_now_iso(),
                    )
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
        wiping history (unlike ``/api/restart`` → ``session.clear()``).

        Pushes a ``Halt`` via the public ``runtime.halt()``: the runtime
        cancels the in-flight model call, which stops the mid-turn
        ``claude --print`` subprocess (tool calls already started keep
        running); the runtime then takes any operator message buffered
        while the reply streamed and the agent acts on it. With nothing
        buffered, the agent waits for the next message.

        This restores the graceful "redirect a mid-flight agent"
        capability that the removed per-message ``urgent`` flag used to
        gate (default stays non-interrupting; this is the explicit opt-in
        escape hatch). It is ENTIRELY host-side — the Halt machinery and
        the cancellable CLI stream already exist in sagent.

        Typical redirect: POST the new instruction to ``/api/post`` as the
        operator (it buffers while the reply streams), then POST here to
        abort the current turn so the buffered message is acted on
        immediately. A PEER's message is held until the agent is free, so
        it is not what a halt hands over: it is delivered with the next
        message that arrives. Loopback-only.
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
            agent.runtime.halt()
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
            into the target's inbox (existing web-UI behavior): the
            agent takes it up at its next model round.
          - ``from`` is a known agent label: peer routing called by
            the plugin's MCP server. Writes a ``<from> → [to]`` audit
            record and pushes ``AgentSendQueuedMessage(sender=<from>)``
            into the target's inbox, which the runtime holds until the
            target's current turn is done — a peer never preempts. This
            is how the out-of-process MCP server reaches the live agents
            that ``serve.py`` owns. Respects the ``_SUPPRESS_FLAG``
            sentinel for warmup-window silence.
        """
        from sagent.runtime.types import AgentSendQueuedMessage
        from sagent.session.types import UserMessage

        payload = await request.json()
        to = str(payload.get("to", "")).strip()
        body = str(payload.get("body", ""))
        from_role = str(payload.get("from", "user")).strip() or "user"
        urgent = bool(payload.get("urgent", False))
        if not to or not body:
            return JSONResponse(
                {"error": "both 'to' and 'body' are required"}, status_code=400
            )
        target = agents.get(to)
        if target is None and to not in _MAILBOXES:
            return JSONResponse(
                {"error": f"unknown target {to!r}; active: {sorted(agents)}"},
                status_code=404,
            )
        # User-as-recipient is a "mailbox without listener" — audit
        # log is the visible surface; no inbox push.
        suppress = _SUPPRESS_FLAG.exists()
        if from_role == "user":
            if target is None:
                return JSONResponse(
                    {"error": f"user→{to} is meaningless; pick an agent"},
                    status_code=400,
                )
            if not suppress:
                delivery.append_user_message(to_role=to, body=body)
            target.runtime.push(
                UserMessage(content=body, timestamp=_now_iso()),
            )
        else:
            # Peer routing path called by the MCP server.
            if not suppress:
                delivery.append_record(from_role=from_role, to=[to], body=body)
            if target is not None:
                target.runtime.push(
                    AgentSendQueuedMessage(sender=from_role, text=body),
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
        if urgent and target is not None:
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
        ``AgentSendQueuedMessage(sender=<from>, text='[defer +Ns] <body>')``
        into the target's inbox after ``delay_s`` seconds. Audit log
        receives one record at schedule time with body
        ``[defer +Ns scheduled] <body>``. Sentinel-respecting same as
        ``/api/post``.
        """
        from sagent.runtime.types import AgentSendQueuedMessage

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
        target = agents.get(to)
        if target is None:
            return JSONResponse(
                {"error": f"unknown target {to!r}; active: {sorted(agents)}"},
                status_code=404,
            )

        marked = f"[defer +{delay_s}s] {body}"

        def _fire() -> None:
            if not target.running:
                _LOG.warning("defer: target @%s is no longer running; dropping", to)
                return
            target.runtime.push(
                AgentSendQueuedMessage(sender=from_role, text=marked),
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

    if _RESUME_MODE_WARNING:
        _LOG.warning("%s", _RESUME_MODE_WARNING)

    # Everything the team holds open — the provider, each role's journal, MCP
    # session and agent — closes with this stack, in reverse order.
    async with AsyncExitStack() as stack:
        # Restart-resume happens as each agent opens: it loads its session
        # journal's last window (slimmed, in `slim` mode) BEFORE it serves a
        # turn (and before warmup). See AGENT_TEAM_RESUME_MODE (slim|full).
        agents = await _build_all_agents(profile, stack)
        _LOG.info("brought up %d agents: %s", len(agents), sorted(agents))

        # Startup warmup — one "call ``sagent_self``, then say ok" round-trip
        # per agent, so the ``claude`` subprocess, its tool catalog and the
        # peer-messaging server are all proven live before real traffic.
        # Without it (sagent 0.1's CLI provider), the very first model call
        # after the subprocess spawned could hang indefinitely (observed
        # 2026-06-01: >5 min no reply). The probe is silent: the sentinel
        # keeps it out of ``main.jsonl`` and ``sagent_self`` reaches no peer.
        _LOG.info("warming up agents (≤90s)…")
        warmup_results = await _warmup_agents(agents)
        ok = [l for l, v in warmup_results.items() if v]
        timed_out = [l for l, v in warmup_results.items() if not v]
        _LOG.info(
            "warmup done: ready=%s timed_out=%s", sorted(ok), sorted(timed_out)
        )

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
