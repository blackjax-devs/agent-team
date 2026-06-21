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
"""Profile loader — turns a profile dir (``team.toml`` + role prompts) into a
:class:`Profile` that drives all framework behavior.

Named ``team_profile`` (NOT ``profile`` — that shadows a stdlib module — and
NOT ``profiles`` — that clashes with the ``profiles/`` data dir).

A profile dir contains:

  - ``team.toml`` — the schema documented in ``profiles/default/team.toml``.
  - ``roles/<role>.md`` — per-role system-prompt templates with a single
    ``{{workspace}}`` placeholder that :func:`render_prompt` fills from the
    ``[workspace]`` table.

Resolution of the active profile dir (in :func:`load_profile`):

  1. the explicit ``profile_dir`` argument, else
  2. ``$AGENT_TEAM_PROFILE_DIR``, else
  3. the bundled default profile shipped inside the package
     (``agent_team/profiles/default``, resolved via importlib.resources).

``workspace_root`` and ``sandbox_root`` are resolved relative to the **launch
cwd** (so ``root = "."`` means "the dir the operator launched from"). Each
``role_prompts`` path is resolved relative to the **profile dir** (prompts are
profile data that ships with the profile, not with the cwd).
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from importlib.resources import files as _pkg_files
from pathlib import Path


def _default_profile_dir() -> Path:
    """Resolve the bundled default profile dir (``agent_team/profiles/default``).

    Uses :mod:`importlib.resources` rather than ``__file__`` arithmetic so the
    default profile resolves correctly when the package is INSTALLED into a
    venv (the profile ships as package data — see ``pyproject.toml``
    ``[tool.setuptools.package-data]``). For a regular (non-zip) install the
    returned ``Traversable`` is a real filesystem path; we coerce to
    :class:`pathlib.Path` so the rest of :func:`load_profile` (which opens
    ``team.toml`` and reads role prompts off disk) works unchanged.
    """
    return Path(_pkg_files("agent_team") / "profiles" / "default")


# The template token replaced by :func:`render_prompt` with the workspace block.
_WORKSPACE_TOKEN = "{{workspace}}"


@dataclass
class Profile:
    """A fully-resolved profile: everything the framework needs to boot.

    Attributes:
        name: Human label for the deployment (``[team].name``).
        session_id_namespace: String fed into the per-role uuid5 derivation
            (``[team].session_id_namespace``). Changing it per deployment keeps
            two deployments on the same machine/cwd from colliding on session
            files.
        workspace_root: Resolved absolute path to the project/monorepo root.
        workspace_repos: Sub-repo names for a monorepo; empty for a single
            project. Listed in the rendered ``{{workspace}}`` block when present.
        sandbox_root: Resolved absolute path the write-capable roles
            (statistician's ``SandboxedWrite``/``SandboxedEdit``) are confined to.
        workspace_description: One-line description injected into the
            ``{{workspace}}`` prompt block.
        roster: Ordered list of role labels to spin up (1..N).
        models: ``role label -> model id`` map.
        role_prompts: ``role label -> absolute Path`` of the prompt template.
        profile_dir: The resolved profile directory itself.
    """

    name: str
    session_id_namespace: str
    workspace_root: Path
    workspace_repos: list[str]
    sandbox_root: Path
    workspace_description: str
    roster: list[str]
    models: dict[str, str]
    role_prompts: dict[str, Path]
    profile_dir: Path
    # Optional per-role flags (e.g. {"statistician": {"sandbox": True}}).
    role_flags: dict[str, dict] = field(default_factory=dict)


def _resolve_profile_dir(profile_dir: str | os.PathLike | None) -> Path:
    if profile_dir is not None:
        return Path(profile_dir).expanduser().resolve()
    env = os.environ.get("AGENT_TEAM_PROFILE_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return _default_profile_dir().resolve()


def _resolve_roster(raw_roster, presets: dict[str, list[str]]) -> list[str]:
    """Expand a roster spec into a concrete ordered list of role labels.

    ``raw_roster`` may be:
      - a list of role labels (read verbatim), with any element that names a
        preset expanded in place, or
      - a single string naming a preset.
    """
    if isinstance(raw_roster, str):
        if raw_roster in presets:
            return list(presets[raw_roster])
        return [raw_roster]
    out: list[str] = []
    for item in raw_roster or []:
        if isinstance(item, str) and item in presets:
            out.extend(presets[item])
        else:
            out.append(item)
    return out


def load_profile(profile_dir: str | os.PathLike | None = None) -> Profile:
    """Load and resolve a :class:`Profile` from a profile dir.

    See the module docstring for the dir-resolution order. ``team.toml`` is
    parsed with :mod:`tomllib`; ``workspace_root``/``sandbox_root`` resolve
    relative to the launch cwd; each role prompt path resolves relative to the
    profile dir.
    """
    pdir = _resolve_profile_dir(profile_dir)
    toml_path = pdir / "team.toml"
    if not toml_path.exists():
        raise FileNotFoundError(f"profile is missing team.toml: {toml_path}")
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)

    team = data.get("team", {})
    workspace = data.get("workspace", {})
    presets = data.get("presets", {})
    models = dict(data.get("models", {}))
    roles_table = data.get("roles", {})

    name = str(team.get("name", "agent-team"))
    session_id_namespace = str(team.get("session_id_namespace", name))

    # workspace_root / sandbox_root resolve relative to the launch cwd.
    cwd = Path.cwd()
    root_raw = str(workspace.get("root", "."))
    workspace_root = (cwd / root_raw).resolve()
    sandbox_raw = str(workspace.get("sandbox_root", ".agent-team/sandbox"))
    sandbox_path = Path(sandbox_raw)
    sandbox_root = (
        sandbox_path.resolve()
        if sandbox_path.is_absolute()
        else (cwd / sandbox_path).resolve()
    )

    workspace_repos = [str(r) for r in workspace.get("repos", [])]
    workspace_description = str(
        workspace.get("description", "a software project")
    )

    # Roster: prefer [workspace].roster (where the schema puts it), fall back
    # to a top-level [roster] for forward-compat, then to all configured roles.
    raw_roster = workspace.get("roster")
    if raw_roster is None:
        raw_roster = data.get("roster")
    if raw_roster is None:
        raw_roster = list(roles_table.keys())
    roster = _resolve_roster(raw_roster, presets)

    # Role prompt templates resolve relative to the profile dir.
    role_prompts: dict[str, Path] = {}
    role_flags: dict[str, dict] = {}
    for role, spec in roles_table.items():
        if isinstance(spec, dict):
            prompt_rel = spec.get("prompt")
            flags = {k: v for k, v in spec.items() if k != "prompt"}
        else:
            prompt_rel = spec
            flags = {}
        if prompt_rel is None:
            prompt_rel = f"roles/{role}.md"
        role_prompts[role] = (pdir / str(prompt_rel)).resolve()
        if flags:
            role_flags[role] = flags

    return Profile(
        name=name,
        session_id_namespace=session_id_namespace,
        workspace_root=workspace_root,
        workspace_repos=workspace_repos,
        sandbox_root=sandbox_root,
        workspace_description=workspace_description,
        roster=roster,
        models=models,
        role_prompts=role_prompts,
        profile_dir=pdir,
        role_flags=role_flags,
    )


def _workspace_block(profile: Profile) -> str:
    """Build the prose block that replaces the ``{{workspace}}`` token.

    A single project yields one description line. A monorepo additionally lists
    its repos.
    """
    desc = profile.workspace_description
    if profile.workspace_repos:
        repos = " ".join(f"`{r}/`" for r in profile.workspace_repos)
        return f"{desc} (repos: {repos})"
    return desc


def render_prompt(profile: Profile, role: str) -> str:
    """Read ``role_prompts[role]`` and fill the ``{{workspace}}`` placeholder.

    Returns the filled system-prompt text. Raises ``KeyError`` if the role has
    no configured prompt path.
    """
    prompt_path = profile.role_prompts[role]
    text = prompt_path.read_text(encoding="utf-8")
    return text.replace(_WORKSPACE_TOKEN, _workspace_block(profile))
