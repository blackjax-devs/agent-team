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
"""agent_team — a configurable, Bayesian-aware multi-agent dev team.

Built on `sagent <https://github.com/rekursiv-ai/sagent>`_ + the Claude CLI.
The default profile targets building and debugging apps that use BlackJAX, but
the workspace, roster, models, and prompts are all config (a *profile dir*).

Console entry points (see ``pyproject.toml [project.scripts]``):

  - ``agent-team``       → :func:`agent_team.serve.main`
  - ``agent-team-merge`` → :func:`agent_team.merge_jsonl.main`

The default profile and the web UI ship inside this package
(``agent_team/profiles/default`` + ``agent_team/web``) and are resolved at
runtime via :mod:`importlib.resources`, so the framework works when installed
into a venv from any cwd — no reliance on a source-tree path.
"""

from __future__ import annotations

__version__ = "0.1.0"
