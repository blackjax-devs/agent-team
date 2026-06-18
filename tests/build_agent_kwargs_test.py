"""Consumer-side guard for the provider-kwargs seam.

The 2026-06 sagent export regression dropped ``extra_mcp_servers`` / ``session_id``
/ ``subprocess_read_timeout_sec`` from ``AnthropicCLI.model()``. ``nightly-sagent
-head`` checks *sagent's* side (does upstream ``main``'s ``model()`` still accept
those kwargs). This test checks *our* side: that ``build_agent`` still PASSES them
— and that the ``session_id`` conditional (defensive against a re-drop) holds. It
mocks the provider, so it needs no live sagent, no credentials, and no network.
"""
import sagent.agent

from agent_team.roles import common


class _FullSigProvider:
    """Provider whose ``model()`` has the full #177 signature; records each call."""

    def __init__(self):
        self.calls = []

    def model(self, model_id=None, max_request_tokens=None, *,
              extra_mcp_servers=None, session_id=None, subprocess_read_timeout_sec=None):
        self.calls.append(dict(
            model_id=model_id, extra_mcp_servers=extra_mcp_servers,
            session_id=session_id, subprocess_read_timeout_sec=subprocess_read_timeout_sec))
        return object()


class _NoSessionIdProvider:
    """Provider whose ``model()`` does NOT accept ``session_id`` (a session_id-less sagent)."""

    def __init__(self):
        self.calls = []

    def model(self, model_id=None, max_request_tokens=None, *,
              extra_mcp_servers=None, subprocess_read_timeout_sec=None):
        self.calls.append(dict(
            model_id=model_id, extra_mcp_servers=extra_mcp_servers,
            subprocess_read_timeout_sec=subprocess_read_timeout_sec))
        return object()


def _build(monkeypatch, tmp_path, provider):
    monkeypatch.setenv("SAGENT_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(common, "build_provider", lambda: provider)
    # Stub Agent: capture-and-return, don't construct a real one (no claude needed).
    monkeypatch.setattr(sagent.agent, "Agent", lambda **kw: kw)
    return common.build_agent(
        role_name="tl", tools=[], model_id="claude-opus-4-8",
        system="You are TL.", session_namespace="test-ns", peers=["tl", "swe"])


def test_build_agent_passes_required_model_kwargs(monkeypatch, tmp_path):
    p = _FullSigProvider()
    _build(monkeypatch, tmp_path, p)
    assert len(p.calls) == 1
    call = p.calls[0]
    # extra_mcp_servers carries the peer-messaging MCP server — load-bearing.
    assert "sagent_chat" in (call["extra_mcp_servers"] or {}), call
    assert call["subprocess_read_timeout_sec"] == 300.0, call
    # session_id passed because this provider's model() accepts it.
    assert call["session_id"] is not None, call


def test_build_agent_omits_session_id_when_provider_lacks_it(monkeypatch, tmp_path):
    # Forward-compat: a provider whose model() lacks session_id must NOT TypeError.
    p = _NoSessionIdProvider()
    _build(monkeypatch, tmp_path, p)  # must not raise
    assert len(p.calls) == 1
    call = p.calls[0]
    assert "session_id" not in call                            # conditional dropped it
    assert "sagent_chat" in (call["extra_mcp_servers"] or {})  # still required
    assert call["subprocess_read_timeout_sec"] == 300.0
