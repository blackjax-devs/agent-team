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
"""Tests for trace_writer.py — per-agent runtime event JSONL writer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import json

from agent_team.runtime import trace_writer


@dataclass
class _FakeAssistantMessage:
    text: str
    tool_calls: tuple[object, ...] = ()


@dataclass
class _FakeModelResponseComplete:
    message: _FakeAssistantMessage
    generation: int = 0


@dataclass
class _FakeWeirdEvent:
    """An event with a non-asdict-able field (e.g. asyncio task)."""

    name: str
    handle: object = None


class _StubAgentRuntime:
    def __init__(self) -> None:
        self.observers: list[object] = []


class _StubAgent:
    def __init__(self) -> None:
        self.runtime = _StubAgentRuntime()


def test_trace_path_under_sessions_dir(tmp_path: Path) -> None:
    p = trace_writer.trace_path_for("tl", sessions_dir=tmp_path)
    assert p == tmp_path / "tl.trace.jsonl"


def test_serializes_simple_event(tmp_path: Path) -> None:
    w = trace_writer.TraceWriter("tl", sessions_dir=tmp_path)
    w(_FakeAssistantMessage(text="hello"))
    lines = (tmp_path / "tl.trace.jsonl").read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["_event"] == "_FakeAssistantMessage"
    assert rec["_n"] == 0
    assert rec["text"] == "hello"
    assert rec["tool_calls"] == []
    assert rec["_ts"].endswith("Z")


def test_nested_dataclass_recursion(tmp_path: Path) -> None:
    w = trace_writer.TraceWriter("tl", sessions_dir=tmp_path)
    inner = _FakeAssistantMessage(text="hi", tool_calls=())
    w(_FakeModelResponseComplete(message=inner, generation=7))
    rec = json.loads((tmp_path / "tl.trace.jsonl").read_text().strip())
    assert rec["_event"] == "_FakeModelResponseComplete"
    assert rec["message"]["text"] == "hi"
    assert rec["generation"] == 7


def test_non_serializable_field_falls_back_to_repr(tmp_path: Path) -> None:
    """Asyncio tasks and similar end up as repr strings, not crashes."""
    import asyncio

    loop = asyncio.new_event_loop()
    try:

        async def _noop():
            pass

        task = loop.create_task(_noop())
        w = trace_writer.TraceWriter("tl", sessions_dir=tmp_path)
        w(_FakeWeirdEvent(name="x", handle=task))
        rec = json.loads((tmp_path / "tl.trace.jsonl").read_text().strip())
        assert rec["name"] == "x"
        # Asyncio task isn't dataclass/dict/list — falls through to repr.
        assert isinstance(rec["handle"], str)
        assert "Task" in rec["handle"] or "function" in rec["handle"]
    finally:
        loop.close()


def test_mapping_fields_serialize_as_objects(tmp_path: Path) -> None:
    """sagent holds a record's JSON fields read-only (``MappingProxyType``)."""
    from types import MappingProxyType

    w = trace_writer.TraceWriter("tl", sessions_dir=tmp_path)
    w(_FakeWeirdEvent(name="x", handle=MappingProxyType({"to": "user", "n": 1})))
    rec = json.loads((tmp_path / "tl.trace.jsonl").read_text().strip())
    assert rec["handle"] == {"to": "user", "n": 1}


def test_sagent_events_carry_the_fields_the_web_ui_reads(tmp_path: Path) -> None:
    """The viewers read ``text`` / ``source`` / ``message`` / ``error``."""
    from sagent.model.scripted import call, reply
    from sagent.runtime.types import ModelResponseComplete, ModelResponseError
    from sagent.session.types import AgentToAgentMessage, UserMessage

    w = trace_writer.TraceWriter("tl", sessions_dir=tmp_path)
    w(UserMessage(content="status report?"))
    w(AgentToAgentMessage(sender="swe", content="PR is up"))
    w(
        ModelResponseComplete(
            response=reply(
                "on it", calls=[call("Read", "c1", file_path="a.py"), call("Bash", "c2")]
            )
        )
    )
    w(ModelResponseError(exception=RuntimeError("claude exited")))
    user, peer, done, failed = (
        json.loads(line)
        for line in (tmp_path / "tl.trace.jsonl").read_text().splitlines()
    )
    assert (user["_event"], user["text"]) == ("UserMessage", "status report?")
    assert "source" not in user
    assert (peer["_event"], peer["text"], peer["source"]) == (
        "AgentToAgentMessage",
        "PR is up",
        "swe",
    )
    assert done["message"] == {
        "text": "on it",
        "tool_calls": [
            {"name": "Read", "args": {"file_path": "a.py"}},
            {"name": "Bash", "args": {}},
        ],
    }
    # sagent's own shape is still there, beside the derived fields.
    assert done["response"]["stop_reason"] == "tool_use"
    assert failed["error"] == "RuntimeError: claude exited"


def test_ordinal_increments_and_persists_across_instances(tmp_path: Path) -> None:
    w1 = trace_writer.TraceWriter("tl", sessions_dir=tmp_path)
    w1(_FakeAssistantMessage(text="a"))
    w1(_FakeAssistantMessage(text="b"))
    # New instance picks up the existing line count as starting ordinal.
    w2 = trace_writer.TraceWriter("tl", sessions_dir=tmp_path)
    w2(_FakeAssistantMessage(text="c"))
    lines = (tmp_path / "tl.trace.jsonl").read_text().splitlines()
    ns = [json.loads(l)["_n"] for l in lines]
    assert ns == [0, 1, 2]


def test_install_on_attaches_observer(tmp_path: Path) -> None:
    agent = _StubAgent()
    w = trace_writer.install_on(agent, "swe", sessions_dir=tmp_path)
    assert w in agent.runtime.observers
    # Firing the observer writes a record:
    w(_FakeAssistantMessage(text="hello"))
    rec = json.loads((tmp_path / "swe.trace.jsonl").read_text().strip())
    assert rec["text"] == "hello"


def test_concurrent_appends_atomic(tmp_path: Path) -> None:
    """Two writers on the same file produce no interleaved lines."""
    # Run two writers in subprocesses pushing 100 records each.
    import subprocess
    import sys

    target = tmp_path / "tl.trace.jsonl"
    script = f"""
import json, sys
from agent_team.runtime import trace_writer
tag = sys.argv[1]
w = trace_writer.TraceWriter('tl', sessions_dir={str(tmp_path)!r})
class E: pass
for i in range(100):
    e = E()
    e.tag = f'{{tag}}-{{i:03d}}'
    rec = trace_writer._to_jsonable(e)  # noqa: SLF001 — we cheat for the test
    # actually re-route through the public path with a fake dataclass:
    import dataclasses
    @dataclasses.dataclass
    class M:
        tag: str
    w(M(tag=f'{{tag}}-{{i:03d}}'))
"""

    procs = [
        subprocess.Popen([sys.executable, "-c", script, tag]) for tag in ("A", "B")
    ]
    for p in procs:
        assert p.wait() == 0
    lines = target.read_text().splitlines()
    assert len(lines) == 200
    # Every line must parse:
    for l in lines:
        json.loads(l)
