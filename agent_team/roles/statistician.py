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


def build(profile: "Profile", **agent_kwargs):
    """Construct the Statistician Agent.

    Read-anywhere. Edit/Write restricted to ``profile.sandbox_root``
    via sandboxed tool wrappers (see ``sandboxed_tools.py``).
    Model + system prompt come from the loaded profile.
    """
    from ..team_profile import render_prompt

    from sagent.tools.bash import Bash
    from sagent.tools.glob_tool import Glob
    from sagent.tools.grep import Grep
    from sagent.tools.read import Read
    from sagent.tools.web_fetch import WebFetch
    from sagent.tools.web_search import WebSearch

    from .. import sandboxed_tools

    sandbox = profile.sandbox_root
    sandbox.mkdir(parents=True, exist_ok=True)
    return build_agent(
        role_name=_ROLE,
        system=render_prompt(profile, _ROLE),
        tools=[
            Read(),
            Grep(),
            Glob(),
            Bash(),
            WebSearch(),
            WebFetch(),
            sandboxed_tools.SandboxedWrite(sandbox_root=sandbox),
            sandboxed_tools.SandboxedEdit(sandbox_root=sandbox),
        ],
        model_id=profile.models.get(_ROLE, MODEL_OPUS),
        peers=profile.roster,
        **agent_kwargs,
    )
