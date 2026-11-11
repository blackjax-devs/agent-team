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
"""Path-sandboxed variants of sagent's ``Write`` and ``Edit`` tools.

The statistician role is permitted to edit experiment scripts and
diagnostic notebooks under the configured ``sandbox_root`` only (from
the profile's ``[workspace].sandbox_root``). Without runtime
enforcement, the role's prompt-level "do not edit production code"
instruction is advisory — a misjudgment costs us a production
edit. These tools convert the boundary into a structural property
of the tool dispatch: any attempt to write or edit a path outside the
configured sandbox root resolves immediately as an error result,
without ever opening the file.

They inherit ``name``, ``description``, ``schema`` and the rest from the
underlying tool, so the model sees an ordinary ``Write`` / ``Edit`` — no
special prompt needed. Only the validation in ``call`` differs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import override

from sagent.session.types import EditResult, WriteResult
from sagent.tools.edit import Edit
from sagent.tools.types import ToolContext
from sagent.tools.write import Write
from treekle import PlainTree


def _is_within(child: Path, parent: Path) -> bool:
    """True if ``child`` is ``parent`` or any descendant.

    Both ends are resolved by the caller (follows symlinks, normalizes
    ``..``) so a sneaky ``../../etc/passwd`` argument cannot escape via
    path manipulation.
    """
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _resolved_root(sandbox_root: Path | str) -> Path:
    root = Path(sandbox_root)
    if not root.is_absolute():
        raise ValueError(f"sandbox_root must be absolute, got {sandbox_root!r}")
    return root.resolve()


def _refusal(
    context: ToolContext,
    arguments: Mapping[str, PlainTree],
    *,
    root: Path,
    verb: str,
) -> str | None:
    """Return why the call is refused, or ``None`` when its target is in ``root``."""
    raw_path = str(arguments.get("file_path", ""))
    if not raw_path:
        return "'file_path' is required."
    target = context.path(raw_path).resolve()
    if _is_within(target, root):
        return None
    return (
        f"Refused: {target} is outside the statistician sandbox "
        f"({root}). Statistician may only {verb} under "
        f"this root. Flag production-code changes to @swe via "
        f"AgentSend instead."
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class SandboxedWrite(Write):
    """``Write`` restricted to a configured root directory.

    Args:
        sandbox_root: Directory tree the tool is allowed to write
            inside. Must be an absolute path. Resolved once at
            construction.

    """

    sandbox_root: Path | str

    def __post_init__(self) -> None:
        object.__setattr__(self, "sandbox_root", _resolved_root(self.sandbox_root))

    @override
    async def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> WriteResult:
        refusal = _refusal(
            context, arguments, root=Path(self.sandbox_root), verb="write"
        )
        if refusal is not None:
            return WriteResult(call_id="", shown=refusal, is_error=True)
        return await Write.call(self, context, arguments)


@dataclass(frozen=True, slots=True, kw_only=True)
class SandboxedEdit(Edit):
    """``Edit`` restricted to a configured root directory. See ``SandboxedWrite``."""

    sandbox_root: Path | str

    def __post_init__(self) -> None:
        object.__setattr__(self, "sandbox_root", _resolved_root(self.sandbox_root))

    @override
    async def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> EditResult:
        refusal = _refusal(
            context, arguments, root=Path(self.sandbox_root), verb="edit"
        )
        if refusal is not None:
            return EditResult(call_id="", shown=refusal, is_error=True)
        return await Edit.call(self, context, arguments)
