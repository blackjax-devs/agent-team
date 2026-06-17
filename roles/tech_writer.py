"""Tech-writer role factory.

The role label is ``tech-writer`` (hyphenated). The Python module
filename uses an underscore because Python module names can't contain
hyphens; the role label + its profile prompt file stay hyphenated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .common import MODEL_HAIKU, build_agent

if TYPE_CHECKING:
    from team_profile import Profile


_ROLE = "tech-writer"


def build(profile: "Profile"):
    """Construct the Tech-Writer Agent.

    Model + system prompt come from the loaded profile.
    """
    from team_profile import render_prompt

    from sagent import tools

    return build_agent(
        role_name=_ROLE,
        system=render_prompt(profile, _ROLE),
        tools=[
            tools.Read(),
            tools.Grep(),
            tools.Glob(),
            tools.Edit(),
            tools.Write(),
            tools.Bash(),
            tools.WebSearch(),
            tools.WebFetch(),
        ],
        model_id=profile.models.get(_ROLE, MODEL_HAIKU),
        session_namespace=profile.session_id_namespace,
        peers=profile.roster,
    )
