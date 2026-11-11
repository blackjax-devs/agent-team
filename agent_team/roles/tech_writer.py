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
"""Tech-writer role factory.

The role label is ``tech-writer`` (hyphenated). The Python module
filename uses an underscore because Python module names can't contain
hyphens; the role label + its profile prompt file stay hyphenated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .common import MODEL_HAIKU, build_agent

if TYPE_CHECKING:
    from ..team_profile import Profile


_ROLE = "tech-writer"


def build(profile: "Profile", **agent_kwargs):
    """Construct the Tech-Writer Agent.

    Model + system prompt come from the loaded profile.
    """
    from ..team_profile import render_prompt

    from sagent.tools.bash import Bash
    from sagent.tools.edit import Edit
    from sagent.tools.glob_tool import Glob
    from sagent.tools.grep import Grep
    from sagent.tools.read import Read
    from sagent.tools.web_fetch import WebFetch
    from sagent.tools.web_search import WebSearch
    from sagent.tools.write import Write

    return build_agent(
        role_name=_ROLE,
        system=render_prompt(profile, _ROLE),
        tools=[
            Read(),
            Grep(),
            Glob(),
            Edit(),
            Write(),
            Bash(),
            WebSearch(),
            WebFetch(),
        ],
        model_id=profile.models.get(_ROLE, MODEL_HAIKU),
        peers=profile.roster,
        **agent_kwargs,
    )
