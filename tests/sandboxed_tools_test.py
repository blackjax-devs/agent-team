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
"""Tests for ``sandboxed_tools.py``: path-restricted Write and Edit.

These exercise the sandbox boundary directly, without going through a
full agent. The goal is to prove: (a) in-sandbox writes/edits delegate
to the underlying tool, (b) out-of-sandbox writes/edits reject with a
clear error before opening the file, (c) ``..`` path traversal cannot
escape, (d) symlinks pointing outside the sandbox cannot escape.
"""

from __future__ import annotations

from pathlib import Path

import asyncio
import os

import pytest

from agent_team import sandboxed_tools
from sagent.tools.types import ToolContext


def _run(tool, arguments, context: ToolContext | None = None):
    """Drive one tool call with a fresh agent's ``ToolContext``."""
    return asyncio.run(tool.call(context or ToolContext(), arguments))


def test_sandboxed_write_rejects_absolute_path_outside_sandbox(tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    outside = tmp_path / "outside.txt"
    tool = sandboxed_tools.SandboxedWrite(sandbox_root=sandbox)
    result = _run(tool, {"file_path": str(outside), "content": "hi"})
    assert result.is_error
    assert "outside the statistician sandbox" in result.text()
    assert not outside.exists()


def test_sandboxed_write_accepts_path_inside_sandbox(tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    target = sandbox / "experiment.py"
    tool = sandboxed_tools.SandboxedWrite(sandbox_root=sandbox)
    result = _run(tool, {"file_path": str(target), "content": "print('ok')\n"})
    assert not result.is_error, f"unexpected error: {result.text()}"
    assert target.read_text() == "print('ok')\n"


def test_sandboxed_write_resolves_relative_path_from_agent_cwd(tmp_path: Path) -> None:
    """A relative ``file_path`` is judged from the calling agent's cwd."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    tool = sandboxed_tools.SandboxedWrite(sandbox_root=sandbox)
    inside = _run(
        tool, {"file_path": "a.py", "content": "x\n"}, ToolContext(cwd=sandbox)
    )
    assert not inside.is_error, inside.text()
    assert (sandbox / "a.py").read_text() == "x\n"
    outside = _run(
        tool, {"file_path": "a.py", "content": "x\n"}, ToolContext(cwd=tmp_path)
    )
    assert outside.is_error
    assert not (tmp_path / "a.py").exists()


def test_sandboxed_write_blocks_dot_dot_traversal(tmp_path: Path) -> None:
    """``../outside.txt`` resolves above the sandbox and must reject."""
    sandbox = tmp_path / "sandbox"
    nested = sandbox / "nested"
    nested.mkdir(parents=True)
    escape_target = tmp_path / "pwned.txt"
    tool = sandboxed_tools.SandboxedWrite(sandbox_root=sandbox)
    result = _run(
        tool,
        {"file_path": str(nested / ".." / ".." / "pwned.txt"), "content": "x"},
    )
    assert result.is_error
    assert not escape_target.exists()


def test_sandboxed_write_blocks_symlink_escape(tmp_path: Path) -> None:
    """A symlink inside the sandbox pointing outside still rejects."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    outside_target = tmp_path / "outside.txt"
    outside_target.write_text("original\n")
    link_inside = sandbox / "trojan.txt"
    os.symlink(outside_target, link_inside)

    tool = sandboxed_tools.SandboxedWrite(sandbox_root=sandbox)
    result = _run(tool, {"file_path": str(link_inside), "content": "PWNED\n"})
    assert result.is_error
    # outside_target must be untouched:
    assert outside_target.read_text() == "original\n"


def test_sandboxed_write_rejects_missing_file_path() -> None:
    tool = sandboxed_tools.SandboxedWrite(sandbox_root=Path("/tmp"))
    result = _run(tool, {"content": "hi"})
    assert result.is_error
    assert "'file_path' is required" in result.text()


def test_sandboxed_edit_accepts_in_sandbox_target(tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    target = sandbox / "f.py"
    target.write_text("x = 1\ny = 2\n")
    tool = sandboxed_tools.SandboxedEdit(sandbox_root=sandbox)
    # Record the file as read, so Edit's read-before-write check passes.
    context = ToolContext()
    context.files.saw(target)
    result = _run(
        tool,
        {"file_path": str(target), "old_string": "x = 1", "new_string": "x = 42"},
        context,
    )
    assert not result.is_error, f"unexpected error: {result.text()}"
    assert target.read_text() == "x = 42\ny = 2\n"


def test_sandboxed_edit_rejects_out_of_sandbox(tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    outside = tmp_path / "production.py"
    outside.write_text("DANGER = True\n")
    tool = sandboxed_tools.SandboxedEdit(sandbox_root=sandbox)
    context = ToolContext()
    context.files.saw(outside)
    result = _run(
        tool,
        {
            "file_path": str(outside),
            "old_string": "DANGER = True",
            "new_string": "DANGER = False",
        },
        context,
    )
    assert result.is_error
    assert "outside the statistician sandbox" in result.text()
    # File contents unchanged:
    assert outside.read_text() == "DANGER = True\n"


def test_sandboxed_tools_keep_the_underlying_tool_identity(tmp_path: Path) -> None:
    """The model must see an ordinary ``Write`` / ``Edit``."""
    write = sandboxed_tools.SandboxedWrite(sandbox_root=tmp_path)
    edit = sandboxed_tools.SandboxedEdit(sandbox_root=tmp_path)
    assert (write.name, edit.name) == ("Write", "Edit")
    assert "file_path" in write.schema["properties"]
    assert "old_string" in edit.schema["properties"]


def test_sandbox_root_must_be_absolute() -> None:
    with pytest.raises(ValueError, match="must be absolute"):
        sandboxed_tools.SandboxedWrite(sandbox_root="relative/path")
    with pytest.raises(ValueError, match="must be absolute"):
        sandboxed_tools.SandboxedEdit(sandbox_root="relative/path")
