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
"""Tests for the ``materialize`` resume mode.

Two halves:

1. ``test_provider_seams_present`` — a GUARD against sagent drift. The host-side
   materialize wiring (``serve._materialize_on_resume``) reaches into private
   ``_AnthropicCLIModel`` internals: the ``session_id`` kwarg → ``_session_id``,
   plus ``_session_initialized`` / ``_last_sent_index`` / ``_claude_home`` /
   ``_session_jsonl_path``. If a sagent change renames/removes any of these, our
   ``--resume`` flip silently stops working. nightly-sagent-head.yml runs this
   against sagent ``main`` so a re-drop is caught before we bump the pin (the
   same role the provider-signature smoke plays for ``model()``'s kwargs).

2. ``test_materialize_on_resume_*`` — the wiring logic: it must write the CLI
   session file AND flip BOTH seams, with a path-mismatch guard and best-effort
   bail paths. Mock provider, real ``materialize_session``, throwaway HOME.
"""

import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sagent.providers.anthropic_cli import _AnthropicCLIModel
from sagent.providers.lib.cost import ModelProfile, Pricing
from sagent.types.runtime import AssistantMessage, UserMessage

from agent_team.cli_session.materializer import session_jsonl_path
from agent_team.serve import _materialize_on_resume


@pytest.fixture(autouse=True)
def _stub_probes(monkeypatch):
    """The wiring tests exercise _materialize_on_resume's logic, not the live
    CLI — stub the `claude --version` / git probes so they run headless (CI has
    no `claude` binary). Harmless for the seam test, which never probes."""
    monkeypatch.setattr("agent_team.serve._probe_claude_version", lambda: "2.1.183")
    monkeypatch.setattr("agent_team.serve._probe_git_branch", lambda cwd: "HEAD")


# Exact private surface serve._materialize_on_resume depends on.
_REQUIRED_SEAMS = (
    "_session_id",
    "_session_initialized",
    "_last_sent_index",
    "_claude_home",
    "_session_jsonl_path",
)
_UID = "12345678-1111-2222-3333-444444444444"


def _stub_profile() -> ModelProfile:
    return ModelProfile(
        max_request_tokens=200_000,
        max_response_tokens=8_000,
        supports_thinking=False,
        pricing=Pricing(),
    )


def _real_model(*, session_id: str = _UID) -> _AnthropicCLIModel:
    """A real ``_AnthropicCLIModel`` with a mocked provider (no creds/network)."""
    provider = MagicMock(name="AnthropicCLI")
    provider.account = None
    return _AnthropicCLIModel(
        provider=provider,
        model_id="claude-opus-4-8",
        profile=_stub_profile(),
        max_request_tokens=200_000,
        session_id=session_id,
    )


def test_provider_seams_present():
    """sagent's _AnthropicCLIModel must still expose the seams materialize uses.

    Guards against upstream drift (run against sagent main in the nightly).
    """
    model = _real_model()
    missing = [s for s in _REQUIRED_SEAMS if not hasattr(model, s)]
    assert not missing, (
        f"sagent _AnthropicCLIModel dropped materialize seams: {missing}. "
        "serve._materialize_on_resume cannot flip to --resume without them."
    )
    # the session_id kwarg must land on _session_id (the file stem we --resume)
    assert model._session_id == _UID
    # the flippable seams start in their pre-resume state
    assert model._session_initialized is False
    assert model._last_sent_index == 0
    # the path helpers are callable (used to align our write with the read path)
    assert callable(model._claude_home) and callable(model._session_jsonl_path)


# ---- _materialize_on_resume wiring -------------------------------------------

_MSGS = [
    UserMessage(text="u1"),
    AssistantMessage(text="a1"),
    UserMessage(text="u2"),
    AssistantMessage(text="a2"),
]


def _fake_agent(*, home: Path, session_id=_UID, path_override=None):
    cwd = Path.cwd().resolve()
    correct = session_jsonl_path(session_id, cwd=cwd, home=home) if session_id else None
    reported = path_override if path_override is not None else correct
    model = types.SimpleNamespace(
        _session_id=session_id,
        _session_initialized=False,
        _last_sent_index=0,
        _claude_home=lambda: home,
        _session_jsonl_path=lambda: reported,
    )
    runtime = types.SimpleNamespace(
        context=lambda: types.SimpleNamespace(messages=list(_MSGS))
    )
    return types.SimpleNamespace(model=model, runtime=runtime)


def test_materialize_on_resume_flips_both_seams(tmp_path):
    agent = _fake_agent(home=tmp_path)
    assert _materialize_on_resume("tl", agent) is True
    # BOTH seams flipped — the load-bearing pair (anthropic_cli.py:877/1290).
    assert agent.model._session_initialized is True
    assert agent.model._last_sent_index == len(_MSGS)
    # the file was actually written at the provider's read path
    written = session_jsonl_path(_UID, cwd=Path.cwd().resolve(), home=tmp_path)
    assert written.exists()
    import json

    entries = [json.loads(x) for x in written.read_text().splitlines() if x.strip()]
    assert any(e.get("sessionId") == _UID for e in entries)


def test_materialize_on_resume_path_mismatch_does_not_flip(tmp_path):
    agent = _fake_agent(home=tmp_path, path_override=Path("/other/place/x.jsonl"))
    assert _materialize_on_resume("tl", agent) is False
    assert agent.model._session_initialized is False
    assert agent.model._last_sent_index == 0


def test_materialize_on_resume_no_session_id_bails(tmp_path):
    agent = _fake_agent(home=tmp_path, session_id=None)
    assert _materialize_on_resume("user", agent) is False
    assert agent.model._session_initialized is False


def test_materialize_on_resume_never_raises(tmp_path):
    broken = types.SimpleNamespace(
        model=types.SimpleNamespace(
            _session_id=_UID,
            _session_initialized=False,
            _last_sent_index=0,
            _claude_home=lambda: tmp_path,
            _session_jsonl_path=lambda: None,
        ),
        runtime=types.SimpleNamespace(
            context=lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        ),
    )
    assert _materialize_on_resume("swe", broken) is False
    assert broken.model._session_initialized is False
