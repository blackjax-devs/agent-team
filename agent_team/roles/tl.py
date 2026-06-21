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
"""TL role factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .common import MODEL_OPUS, build_agent

if TYPE_CHECKING:
    from ..team_profile import Profile


_ROLE = "tl"


def build(profile: "Profile"):
    """Construct the TL Agent.

    TL is read-only by scope (no Edit/Write); coordinates by reading
    code, inspecting git history, and routing work to peers via the
    plugin's MCP server (``mcp__sagent_chat__sagent_send``). Bash is
    included for read-only verbs (git log/diff/show, ls, cat, grep,
    journalctl), but TL must not run mutating commands.

    Peer messaging + self-defer + status-report live in the MCP
    server's tool catalog and are NOT in ``tools=[...]``. See
    ``mcp_sagent/server.py`` and the per-agent ``--mcp-config``
    threaded by :func:`common.build_agent`.

    Model + system prompt come from the loaded profile.
    """
    from ..team_profile import render_prompt

    from sagent import tools

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
        ],
        model_id=profile.models.get(_ROLE, MODEL_OPUS),
        session_namespace=profile.session_id_namespace,
        peers=profile.roster,
    )
