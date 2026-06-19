"""Shared building blocks for per-role Agent factories.

Conventions:

- All roles use ``AnthropicCLI.from_credentials()`` (Claude subscription,
  no API key consulted).
- Peer messaging happens via the plugin's MCP server (``mcp_sagent.server``)
  rather than sagent's bridge-mounted ``AgentSend``. The MCP server's
  ``sagent_send`` tool appears in the CLI's catalog as
  ``mcp__sagent_chat__sagent_send`` and works structurally on all three
  models (haiku/sonnet/opus) — see README § "Episode 3" for the probe
  result that motivated the design.
- Heavy background commands must be wrapped in ``systemd-run --user
  --scope --quiet --collect -- bash -c '<cmd>'`` to keep an OOM
  contained in a sibling cgroup instead of cascading through the
  pane. The wrap reminder is appended to every system prompt at
  build time so it survives context compaction (a structural fix
  for the chat/-era SWE-OOM-after-compaction failure mode).
"""

from __future__ import annotations

import inspect

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sagent.compaction.summary import SummaryCompactor
from sagent.types.tools import Tool


# Optional model-id fallbacks. The authoritative role→model assignment now
# comes from the loaded profile's ``[models]`` table (``profile.models``);
# these constants remain only as convenience defaults for callers that build
# an agent outside a profile context.
MODEL_OPUS = "claude-opus-4-8"
MODEL_FABLE = "claude-fable-5"
MODEL_SONNET = "claude-sonnet-4-6"
MODEL_HAIKU = "claude-haiku-4-5"

# Fixed UUID namespace for deriving per-role session ids. Combined with the
# profile's ``session_id_namespace`` string so two deployments don't collide.
_SESSION_UUID_NAMESPACE = "9e0e2c30-3f7e-4a13-9f5b-1a3a2c4d5e6f"
# Fallback session-id namespace string when no profile namespace is threaded
# through. Generic framework default; real deployments always override this
# via ``profile.session_id_namespace`` (see ``team.toml [team]``).
_DEFAULT_SESSION_NAMESPACE = "agent-team"

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

**Interrupting a peer (`urgent=true`).** `sagent_send` takes an \
optional `urgent` flag. By default messages QUEUE — the recipient \
sees them on their next turn. With `urgent=true` the recipient's \
in-flight turn is HALTED (history preserved) and they act on your \
message immediately. Use it SPARINGLY — only when a peer is actively \
going the wrong way and must stop now; in particular, to **correct a \
previous, wrong message you sent** that they are about to act on. \
Normal coordination is `urgent=false`.

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


def build_provider():
    """Construct the shared AnthropicCLI provider.

    One provider instance is shared across all five roles (it owns
    only credentials + the bridge URL; per-agent state lives on the
    Model returned by ``provider.model(...)``).
    """
    from sagent.providers import AnthropicCLI

    return AnthropicCLI.from_credentials()


def _sagent_mcp_server_entry(role: str) -> dict[str, Any]:
    """Per-role stdio MCP entry for the CLI's ``--mcp-config``.

    Spawns ``python -m agent_team.mcp_sagent.server`` with three env vars:

      - ``SAGENT_ROLE``: the calling agent's label, used by the MCP
        server to attribute outgoing peer messages.
      - ``SAGENT_HTTP_URL``: where to POST ``/api/post`` and
        ``/api/defer`` — i.e. ``serve.py``'s loopback URL. The MCP
        server runs in a SEPARATE Python process from ``serve.py``,
        so its in-process ``agent_registry`` is empty; HTTP is the
        only way to reach the live registry.
      - ``SAGENT_DATA_DIR``: where ``main.jsonl``, the sentinel,
        the debug log, and the per-role trace files live. Inherited
        from the parent ``serve.py`` env so audit/data files
        co-locate across the three-process tree.
    """
    import os
    import sys

    from ..mcp_sagent.config_factory import SERVER_MODULE

    port = os.environ.get("SAGENT_HTTP_PORT", "8767")
    env_out = {
        "SAGENT_ROLE": role,
        "SAGENT_HTTP_URL": f"http://127.0.0.1:{port}",
    }
    # Propagate the data dir only if explicitly set — if absent, both
    # parent and child fall back to the plugin source dir, which
    # matches anyway.
    data_dir = os.environ.get("SAGENT_DATA_DIR")
    if data_dir:
        env_out["SAGENT_DATA_DIR"] = data_dir
    return {
        "command": sys.executable,
        "args": ["-m", SERVER_MODULE],
        "env": env_out,
    }


def _model_spec_for(model_id: str):
    """Build a ``ModelSpec`` that lets ``AgentSelf`` swap the model later.

    Without a spec, ``AgentSelf(model_id=...)`` rejects with
    "Agent has no model spec; cannot swap" — the runtime needs to
    know how to reconstruct the provider for the new model.
    """
    from sagent.types.model import ModelSpec

    return ModelSpec(
        provider="AnthropicCLI",
        auth="credentials",
        model_id=model_id,
    )


def _session_id_for(role_name: str, *, namespace: str | None = None) -> str:
    """Stable per-role UUIDv5 (so server restarts ``--resume`` the same
    session rather than orphaning the prior one).

    Uses a fixed UUID namespace combined with the deployment's
    ``session_id_namespace`` string (from the profile's ``[team]`` table) so
    the same role under the same deployment always derives the same
    session_id, while two deployments on the same machine/cwd don't collide.
    When ``namespace`` is ``None`` the generic framework default
    (:data:`_DEFAULT_SESSION_NAMESPACE`) is used. To start fresh,
    ``rm $HOME/.claude/projects/-<encoded-cwd>/<uuid>.jsonl`` (or use a
    different ``SAGENT_DATA_DIR``).
    """
    import uuid as _uuid

    ns_str = namespace or _DEFAULT_SESSION_NAMESPACE
    namespace_uuid = _uuid.UUID(_SESSION_UUID_NAMESPACE)
    return str(_uuid.uuid5(namespace_uuid, f"{ns_str}:{role_name}"))


def _session_dir_for(role_name: str) -> Path:
    """Per-role sagent persistence dir (holds ``session.jsonl``).

    Lives under the data dir so ``serve.py`` restarts can
    ``load_session`` + ``Agent.resume`` each role's tape. Co-located
    with the per-role traces under ``<SAGENT_DATA_DIR>/sessions/``.
    """
    from ..mcp_sagent import delivery

    path = delivery.SESSIONS_DIR / f"{role_name}.sagent"
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_agent(
    *,
    role_name: str,
    tools: Sequence[Tool],
    model_id: str,
    system: str | None = None,
    role_md_path: Path | None = None,
    session_namespace: str | None = None,
    peers: Sequence[str] | None = None,
    max_tool_call_rounds: int | None = None,
    max_budget_usd: float | None = None,
):
    """Construct a sagent Agent for a role with shared defaults baked in.

    The plugin's MCP server (``mcp_sagent/server.py``) is auto-wired
    into the agent's ``claude --print`` subprocess via
    ``provider.model(..., extra_mcp_servers={"sagent": {...}})``. The
    role label is baked into the MCP server's env, so the server
    knows which agent it serves on every CallToolRequest.

    Args:
        role_name: Label used in ``agent_registry`` and as the ``name``
            field on the Agent. Must match the role's label used in
            ``sagent_send(to=...)`` calls from peers.
        tools: Sequence of sagent Tool instances allowed for this role.
            Does NOT include AgentSend or AgentSelf — those live in
            the MCP server (``mcp__sagent_chat__sagent_send``,
            ``mcp__sagent_chat__sagent_self``).
        model_id: Claude model id (from ``profile.models[role]``).
        system: Pre-rendered role-prompt body (``{{workspace}}`` already
            filled by :func:`team_profile.render_prompt`). The standing
            reminders are appended here. Mutually exclusive with
            ``role_md_path``; one of the two must be provided.
        role_md_path: Back-compat fallback — path to a role-prompt markdown
            file read directly when ``system`` is not given.
        session_namespace: Deployment session-id namespace (from
            ``profile.session_id_namespace``); ``None`` falls back to the
            historical default.
        peers: Active roster used to build the peer-messaging known-peers
            list; ``None`` falls back to the historical default roster.
        max_tool_call_rounds: Per-turn cap; ``None`` for sagent default.
        max_budget_usd: Per-agent USD cap; ``None`` disables.

    Returns:
        A configured Agent ready to be registered + driven.

    """
    from sagent.agent import Agent

    if system is not None:
        system_prompt = compose_system_prompt(system, peers=peers)
    elif role_md_path is not None:
        system_prompt = load_system_prompt(role_md_path, peers=peers)
    else:
        raise ValueError("build_agent requires either 'system' or 'role_md_path'")

    provider = build_provider()
    # Stdout-idle timeout for the ``claude`` subprocess transport
    # (Subproc default is 60s). Bumped to 300s to accommodate
    # long-running synchronous Bash tool calls — e.g. a project's
    # ``pre-commit run`` (type-checking / linting touched files, often
    # 60-120s), ``uv run python ...`` calibration scripts (JIT
    # warmup + compilation can be 60+s), heavy test suites. Without
    # the bump, the transport reads claude's silence during the tool
    # wait as a hang, raises ``SubprocessTransportError``, sagent's
    # ``send_with_retry`` retries the model call in-place, and the
    # retried response diverges from the cached partial — emitting
    # the divergence marker AND eating the closing assistant message
    # (the peer ``sagent_send`` back to TL never lands). See worklog
    # ``v2.1-cli-session-materialize`` § 2026-06-09 for the 7 SWE
    # divergence diagnosis. 300s is well below opus/sonnet's
    # 9-10min mid-turn API duration ceiling, so we still catch real
    # hangs without false-positive timing out on legitimate work.
    subprocess_read_timeout_sec = 300.0

    # SummaryCompactor wired. Without a compactor, the agent's tape
    # grows uncapped and the provider hits the API output-cap /
    # context-window edge for heavy multi-PR work — observed in
    # production as SWE's 3.4M cache_read_input_tokens over a ~5h
    # session, with increasing ``aborted_streaming`` retryable errors
    # and eventual stream-cuts. SummaryCompactor's defaults
    # (``utilization_trigger=0.95``, ``compression=0.075``) fire
    # compaction at ~95% of the usable window after reserving 7.5%
    # for the compacted result; the resulting ``ContextSplice`` rides
    # sagent's tape and is fed to claude on the next turn. sagent owns
    # compaction; claude's auto-compact is disabled by the provider.
    compactor = SummaryCompactor()

    # NOTE: must not collide with sagent's bridge server name (``"sagent"``,
    # hardcoded at sagent/providers/lib/mcp_bridge.py:175). The bridge
    # exposes Read/Bash/Glob/Grep/etc. as ``mcp__sagent__<tool>``; the
    # plugin's MCP server exposes peer-messaging as
    # ``mcp__sagent_chat__sagent_send`` / ``__sagent_defer`` /
    # ``__sagent_self``.
    # extra_mcp_servers (peer messaging) + the read timeout are required by every
    # CLI-backed agent — passed unconditionally.
    model_kwargs: dict[str, Any] = {
        "extra_mcp_servers": {"sagent_chat": _sagent_mcp_server_entry(role_name)},
        "subprocess_read_timeout_sec": subprocess_read_timeout_sec,
    }
    # CLI session-persistence (``--session-id``/``--resume``) keeps the prompt
    # cache warm across turns, but it is an OPTIONAL optimization: sagent's tape is
    # the source of truth (the provider rebuilds claude's on-disk session from it
    # on the first turn, incl. after a restart once ``Agent.resume`` rehydrated the
    # tape from ``session.jsonl``). A transient sagent export regression once
    # dropped this kwarg from ``model()`` (fixed upstream, #197/#198), so pass it
    # only if the provider's ``model()`` still accepts it — defensive against a
    # re-drop, with no behaviour change when it's present (the normal case).
    if "session_id" in inspect.signature(provider.model).parameters:
        model_kwargs["session_id"] = _session_id_for(
            role_name, namespace=session_namespace
        )

    return Agent(
        model=provider.model(model_id, **model_kwargs),
        model_spec=_model_spec_for(model_id),
        system=system_prompt,
        tools=list(tools),
        name=role_name,
        # sagent-owned persistence: writes ``<session_dir>/session.jsonl``
        # so a ``serve.py`` restart can ``load_session`` + ``Agent.resume``
        # the tape (which then drives the CLI session rebuild). This is
        # the standard sagent resume path — no claude-session-file parsing.
        session_dir=_session_dir_for(role_name),
        max_tool_call_rounds=max_tool_call_rounds,
        max_budget_usd=max_budget_usd,
        compactor=compactor,
    )
