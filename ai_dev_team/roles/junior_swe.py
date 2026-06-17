"""Junior-SWE role factory.

Same edit scope and tool set as senior SWE, but capped at
``max_tool_call_rounds=20`` to enforce the escalation discipline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .common import MODEL_HAIKU, build_agent

if TYPE_CHECKING:
    from ..team_profile import Profile


_ROLE = "junior-swe"
_MAX_ROUNDS_BEFORE_ESCALATION = 20


def build(profile: "Profile"):
    """Construct the Junior-SWE Agent.

    Model + system prompt come from the loaded profile.
    """
    from ..team_profile import render_prompt

    from sagent import tools

    return build_agent(
        role_name=_ROLE,
        system=render_prompt(profile, _ROLE),
        tools=[
            tools.Read(),
            tools.Edit(),
            tools.Write(),
            tools.Bash(),
            tools.Grep(),
            tools.Glob(),
        ],
        model_id=profile.models.get(_ROLE, MODEL_HAIKU),
        session_namespace=profile.session_id_namespace,
        peers=profile.roster,
        max_tool_call_rounds=_MAX_ROUNDS_BEFORE_ESCALATION,
    )
