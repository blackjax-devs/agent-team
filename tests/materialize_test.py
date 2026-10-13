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
"""Guard materialization's provider contract and native resume behavior.

Real providers use an entirely fake subprocess and MCP bridge: no CLI session,
credentials, or external model calls are needed. Wiring failures must preserve
provider state and reject mismatched paths before writing any session file.
"""

import importlib
import json
import types
from pathlib import Path

import pytest

from sagent.providers import AnthropicCLI
from sagent.types.model import ModelRequest, ModelResponse
from sagent.types.runtime import AssistantMessage, UserMessage

from agent_team.cli_session.materializer import materialize_session, session_jsonl_path
from agent_team.cli_session.parser import parse_jsonl_to_messages
from agent_team.serve import _materialize_on_resume


@pytest.fixture(autouse=True)
def _stub_probes(monkeypatch, tmp_path):
    """The wiring tests exercise _materialize_on_resume's logic, not the live
    CLI — stub the `claude --version` / git probes so they run headless (CI has
    no `claude` binary). Harmless for the seam test, which never probes."""
    monkeypatch.setattr("agent_team.serve._probe_claude_version", lambda: "2.1.183")
    monkeypatch.setattr("agent_team.serve._probe_git_branch", lambda cwd: "HEAD")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)


# Exact private surface serve._materialize_on_resume depends on.
_REQUIRED_SEAMS = (
    "_session_id",
    "_last_sent_index",
    "_session_jsonl_path",
)
_UID = "12345678-1111-2222-3333-444444444444"


def _real_model(*, session_id: str = _UID):
    """Build a real CLI model through sagent's stable public factory."""
    return AnthropicCLI(account="default").model(session_id=session_id)


def _session_flag(model):
    return (
        "_session_on_disk"
        if hasattr(model, "_session_on_disk")
        else "_session_initialized"
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
    # Recognize the current contract and the legacy contract supported by the
    # package's minimum version; unknown contracts must still fail this guard.
    assert hasattr(model, _session_flag(model))
    assert getattr(model, _session_flag(model)) is False
    if hasattr(model, "_session_on_disk"):
        assert hasattr(model, "_unanswered") and model._unanswered is None
    assert model._last_sent_index == 0
    # the path helpers are callable (used to align our write with the read path)
    assert callable(model._session_jsonl_path)
    path = model._session_jsonl_path()
    assert path == session_jsonl_path(_UID, cwd=Path.cwd())


# ---- _materialize_on_resume wiring -------------------------------------------

_MSGS = [
    UserMessage(text="u1"),
    AssistantMessage(text="a1"),
    UserMessage(text="u2"),
    AssistantMessage(text="a2"),
]


def _with_model(model):
    runtime = types.SimpleNamespace(
        context=lambda: types.SimpleNamespace(messages=list(_MSGS))
    )
    return types.SimpleNamespace(model=model, runtime=runtime)


def _fake_agent(*, home: Path, session_id=_UID, path_override=None, legacy=False):
    cwd = Path.cwd().resolve()
    correct = session_jsonl_path(session_id, cwd=cwd, home=home) if session_id else None
    reported = path_override if path_override is not None else correct
    model = types.SimpleNamespace(
        _session_id=session_id,
        _last_sent_index=0,
        _session_jsonl_path=lambda: reported,
    )
    if legacy:
        model._session_initialized = False
    else:
        model._session_on_disk = False
        model._unanswered = None
    return _with_model(model)


@pytest.mark.parametrize("legacy", [False, True])
def test_materialize_on_resume_flips_both_seams(tmp_path, legacy):
    agent = _fake_agent(home=tmp_path, legacy=legacy)
    assert _materialize_on_resume("tl", agent) is True
    assert getattr(agent.model, _session_flag(agent.model)) is True
    assert agent.model._last_sent_index == len(_MSGS)
    if not legacy:
        assert not hasattr(agent.model, "_session_initialized")
        assert agent.model._unanswered is None
    # the file was actually written at the provider's read path
    written = session_jsonl_path(_UID, cwd=Path.cwd().resolve(), home=tmp_path)
    assert written.exists()
    entries = [json.loads(x) for x in written.read_text().splitlines() if x.strip()]
    assert any(e.get("sessionId") == _UID for e in entries)


@pytest.mark.parametrize("mismatch", ["filename", "cwd", "projects"])
def test_materialize_on_resume_path_mismatch_does_not_flip(tmp_path, mismatch):
    correct = session_jsonl_path(_UID, cwd=Path.cwd(), home=tmp_path)
    wrong = {
        "filename": correct.with_name("other.jsonl"),
        "cwd": correct.parent.parent / "wrong-cwd" / correct.name,
        "projects": (
            correct.parent.parent.parent / "other" / correct.parent.name / correct.name
        ),
    }[mismatch]
    agent = _fake_agent(home=tmp_path, path_override=wrong)
    assert _materialize_on_resume("tl", agent) is False
    assert agent.model._session_on_disk is False
    assert agent.model._last_sent_index == 0
    assert not correct.exists() and not wrong.exists()
    assert list(tmp_path.iterdir()) == []


def test_materialize_on_resume_no_session_id_bails(tmp_path):
    agent = _fake_agent(home=tmp_path, session_id=None)
    assert _materialize_on_resume("user", agent) is False
    assert agent.model._session_on_disk is False


def test_materialize_on_resume_never_raises(tmp_path):
    broken = types.SimpleNamespace(
        model=types.SimpleNamespace(
            _session_id=_UID,
            _session_on_disk=False,
            _unanswered=None,
            _last_sent_index=0,
            _session_jsonl_path=lambda: session_jsonl_path(
                _UID, cwd=Path.cwd(), home=tmp_path
            ),
        ),
        runtime=types.SimpleNamespace(
            context=lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        ),
    )
    assert _materialize_on_resume("swe", broken) is False
    assert broken.model._session_on_disk is False
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "missing",
    ["_session_on_disk", "_last_sent_index", "_unanswered", "_session_jsonl_path"],
)
def test_materialize_on_resume_unknown_contract_bails(tmp_path, missing):
    agent = _fake_agent(home=tmp_path)
    delattr(agent.model, missing)
    before = vars(agent.model).copy()
    assert _materialize_on_resume("tl", agent) is False
    assert vars(agent.model) == before
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("failure", ["path", "write"])
def test_materialize_on_resume_provider_or_write_failure_bails(
    tmp_path, monkeypatch, failure
):
    agent = _fake_agent(home=tmp_path)

    def fail(*args, **kwargs):
        raise OSError("unavailable")

    if failure == "path":
        agent.model._session_jsonl_path = fail
    else:
        monkeypatch.setattr("agent_team.cli_session.materializer._atomic_write", fail)
    before = vars(agent.model).copy()
    assert _materialize_on_resume("tl", agent) is False
    assert vars(agent.model) == before
    assert list(tmp_path.iterdir()) == []


def test_materialize_on_resume_invalid_session_id_bails(tmp_path):
    agent = _fake_agent(home=tmp_path, session_id="../outside")
    assert _materialize_on_resume("tl", agent) is False
    assert agent.model._session_on_disk is False
    assert list(tmp_path.iterdir()) == []


def test_materialize_on_resume_config_root_mismatch_bails(tmp_path, monkeypatch):
    agent = _fake_agent(home=tmp_path)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "custom-config"))
    assert _materialize_on_resume("tl", agent) is False
    assert agent.model._session_on_disk is False
    assert agent.model._last_sent_index == 0
    assert list(tmp_path.iterdir()) == []


def test_materialize_on_resume_clears_stale_unanswered_input(tmp_path):
    agent = _fake_agent(home=tmp_path)
    agent.model._unanswered = UserMessage(text="old input awaiting a reply")
    assert _materialize_on_resume("tl", agent) is True
    assert agent.model._unanswered is None
    assert agent.model._last_sent_index == len(_MSGS)


def test_materialize_on_resume_unexpected_written_path_does_not_flip(
    tmp_path, monkeypatch
):
    agent = _fake_agent(home=tmp_path)
    monkeypatch.setattr(
        "agent_team.cli_session.materialize_session",
        lambda *args, **kwargs: (tmp_path / "unexpected.jsonl", []),
    )
    before = vars(agent.model).copy()
    assert _materialize_on_resume("tl", agent) is False
    assert vars(agent.model) == before


def test_materializer_explicit_config_dir_preserves_home_default(tmp_path, monkeypatch):
    home = tmp_path / "home"
    config_dir = tmp_path / "custom-config"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))
    default = session_jsonl_path(_UID, cwd=Path.cwd(), home=home)
    assert default.is_relative_to(home / ".claude")
    explicit = session_jsonl_path(
        _UID, cwd=Path.cwd(), home=home, config_dir=config_dir
    )
    path, _ = materialize_session(
        ModelRequest(messages=_MSGS),
        session_id=_UID,
        cwd=Path.cwd(),
        home=home,
        config_dir=config_dir,
    )
    assert path == explicit and path.exists()
    assert path.is_relative_to(config_dir)
    assert not default.exists()


@pytest.mark.parametrize("location", ["home", "custom", "account"])
async def test_real_provider_resumes_without_replaying_history(
    tmp_path, monkeypatch, location
):
    if location == "custom":
        config_dir = tmp_path / "custom-config"
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))
    else:
        config_dir = tmp_path / "home" / ".claude"
    model = _real_model()
    if location == "account":
        # A named account's constructor normally copies credentials into this
        # private HOME. Exercise its actual path/spawn rules without credentials.
        model._persistent_tmpdir = tmp_path / "account-home"
        config_dir = model._persistent_tmpdir / ".claude"
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "ignored-config"))
    expected = session_jsonl_path(_UID, cwd=Path.cwd(), config_dir=config_dir)
    if location == "custom" and not hasattr(model, "_session_on_disk"):
        # Legacy providers miscomputed arbitrary configuration-directory names;
        # refuse to materialize at their wrong path rather than claim a resume.
        reported = model._session_jsonl_path()
        assert reported != expected
        assert _materialize_on_resume("tl", _with_model(model)) is False
        assert not expected.exists() and not reported.exists()
        assert model._session_initialized is False
        assert model._last_sent_index == 0
        return
    assert model._session_jsonl_path() == expected
    if hasattr(model, "_unanswered"):
        model._unanswered = UserMessage(text="stale retry input")
    assert _materialize_on_resume("tl", _with_model(model)) is True
    assert expected.exists()
    restored = parse_jsonl_to_messages(expected)
    assert [(type(message), message.text) for message in restored] == [
        (type(message), message.text) for message in _MSGS
    ]
    assert getattr(model, _session_flag(model)) is True
    if hasattr(model, "_session_on_disk"):
        assert not hasattr(model, "_session_initialized")

    procs = []

    class FakeSubproc:
        def __init__(self, argv, **kwargs):
            self.argv = argv
            self.env = kwargs["env"]
            self.lines = []
            procs.append(self)

        async def start(self):
            pass

        async def close(self):
            pass

        async def write_line(self, line):
            self.lines.append(json.loads(line))

    bridge = types.SimpleNamespace(
        has_tools=False,
        url="http://fake.invalid/mcp",
        server_name="test",
        update_tools=lambda tools: None,
        set_publish=lambda publish: None,
        drain_detached_results=lambda: [],
    )
    model._tools_bridge = bridge
    monkeypatch.setattr(
        importlib.import_module(type(model).__module__), "Subproc", FakeSubproc
    )

    async def drain(proc, publish):
        return ModelResponse(message=AssistantMessage(text="done"))

    monkeypatch.setattr(model, "_drain_until_result", drain)

    # No newly appended inbound means no historical user input should run again.
    await model.stream(ModelRequest(messages=list(_MSGS)))
    assert procs == []
    await model.stream(ModelRequest(messages=[*_MSGS, UserMessage(text="new inbound")]))
    assert len(procs) == 1
    proc = procs[0]
    assert proc.argv[proc.argv.index("--resume") + 1] == _UID
    assert "--session-id" not in proc.argv
    assert len(proc.lines) == 1
    assert proc.lines[0]["message"]["content"] == "new inbound"
    assert model._last_sent_index == len(_MSGS) + 1
    if hasattr(model, "_unanswered"):
        assert model._unanswered is None
    if location == "account":
        assert proc.env["HOME"] == str(model._persistent_tmpdir)
    if location == "custom":
        assert proc.env["CLAUDE_CONFIG_DIR"] == str(config_dir)
