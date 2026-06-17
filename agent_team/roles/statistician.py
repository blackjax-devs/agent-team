"""Statistician role factory.

Sandbox: ``profile.sandbox_root`` (from the profile's ``[workspace]``
table). Read-anywhere; Edit/Write are restricted to the sandbox root via
the wrappers in ``sandboxed_tools.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .common import MODEL_OPUS, build_agent

if TYPE_CHECKING:
    from ..team_profile import Profile


_ROLE = "statistician"


def build(profile: "Profile"):
    """Construct the Statistician Agent.

    Read-anywhere. Edit/Write restricted to ``profile.sandbox_root``
    via sandboxed tool wrappers (see ``sandboxed_tools.py``).
    Model + system prompt come from the loaded profile.
    """
    from ..team_profile import render_prompt

    from sagent import tools

    from .. import sandboxed_tools

    sandbox = profile.sandbox_root
    sandbox.mkdir(parents=True, exist_ok=True)
    return build_agent(
        role_name=_ROLE,
        system=render_prompt(profile, _ROLE),
        tools=[
            tools.Read(),
            tools.Grep(),
            tools.Glob(),
            tools.Bash(),
            tools.WebSearch(),
            tools.WebFetch(),
            sandboxed_tools.SandboxedWrite(sandbox_root=sandbox),
            sandboxed_tools.SandboxedEdit(sandbox_root=sandbox),
        ],
        model_id=profile.models.get(_ROLE, MODEL_OPUS),
        session_namespace=profile.session_id_namespace,
        peers=profile.roster,
    )
