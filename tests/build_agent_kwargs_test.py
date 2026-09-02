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
"""Consumer-side guard for the provider-kwargs seam.

The 2026-06 sagent export regression dropped ``extra_mcp_servers`` / ``session_id``
/ ``subprocess_read_timeout_sec`` from ``AnthropicCLI.model()``. ``nightly-sagent
-head`` checks *sagent's* side (does upstream ``main``'s ``model()`` still accept
those kwargs). This test checks *our* side: that ``build_agent`` still PASSES them
— and that the ``session_id`` conditional (defensive against a re-drop) holds. It
mocks the provider, so it needs no live sagent, no credentials, and no network.
"""
import sagent.agent
import sagent.types.model

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


def _legacy_agent(*, model_spec=None, **kwargs):
    return {"model_spec": model_spec, **kwargs}


def _build(monkeypatch, tmp_path, provider, agent_factory=_legacy_agent):
    monkeypatch.setenv("SAGENT_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(common, "build_provider", lambda: provider)
    # Stub Agent: capture-and-return, don't construct a real one (no claude needed).
    monkeypatch.setattr(sagent.agent, "Agent", agent_factory)
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


def test_build_agent_passes_model_recipe_to_new_sagent(monkeypatch, tmp_path):
    sentinel = object()

    def new_agent(*, model_recipe=None, **kwargs):
        return {"model_recipe": model_recipe, **kwargs}

    monkeypatch.setattr(common, "_model_recipe_for", lambda model_id: sentinel)
    built = _build(monkeypatch, tmp_path, _FullSigProvider(), new_agent)
    assert built["model_recipe"] is sentinel
    assert "model_spec" not in built


def test_model_recipe_prefers_new_sagent_name(monkeypatch):
    class Recipe:
        def __init__(self, *, provider, auth, model_id):
            self.provider = provider
            self.auth = auth
            self.model_id = model_id

    monkeypatch.setattr(sagent.types.model, "ModelRecipe", Recipe, raising=False)
    recipe = common._model_recipe_for("claude-opus-4-8")
    assert isinstance(recipe, Recipe)
    assert recipe.provider == "AnthropicCLI"
    assert recipe.auth == "credentials"
    assert recipe.model_id == "claude-opus-4-8"
