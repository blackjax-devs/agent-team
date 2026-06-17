"""ai_dev_team — a configurable, Bayesian-aware multi-agent dev team.

Built on `sagent <https://github.com/rekursiv-ai/sagent>`_ + the Claude CLI.
The default profile targets building and debugging apps that use BlackJAX, but
the workspace, roster, models, and prompts are all config (a *profile dir*).

Console entry points (see ``pyproject.toml [project.scripts]``):

  - ``ai-dev-team``       → :func:`ai_dev_team.serve.main`
  - ``ai-dev-team-merge`` → :func:`ai_dev_team.merge_jsonl.main`

The default profile and the web UI ship inside this package
(``ai_dev_team/profiles/default`` + ``ai_dev_team/web``) and are resolved at
runtime via :mod:`importlib.resources`, so the framework works when installed
into a venv from any cwd — no reliance on a source-tree path.
"""

from __future__ import annotations

__version__ = "0.1.0"
