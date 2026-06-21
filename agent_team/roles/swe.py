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
"""SWE role factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .common import MODEL_SONNET, build_agent

if TYPE_CHECKING:
    from ..team_profile import Profile


_ROLE = "swe"


def build(profile: "Profile"):
    """Construct the SWE Agent.

    SWE has full code-edit scope across the workspace (excluding the
    statistician's sandbox). Tool set covers reading, editing, running
    tests, and coordinating with peers via the plugin's MCP server
    (``mcp__sagent_chat__sagent_send``).

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
        model_id=profile.models.get(_ROLE, MODEL_SONNET),
        session_namespace=profile.session_id_namespace,
        peers=profile.roster,
    )
