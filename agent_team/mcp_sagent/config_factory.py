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
"""Generate per-role MCP config files at agent build time.

Each agent's ``claude --print`` subprocess needs its own ``--mcp-config``
pointing at the MCP server with ``SAGENT_ROLE=<label>`` in the
subprocess env. We write one JSON file per role to the plugin's
``sessions/`` dir (gitignored) and return the path; :mod:`runtime.build`
threads it through to the role's :func:`build_agent` call.

The config format follows Claude Code's standard ``mcpServers`` shape:

  {
    "mcpServers": {
      "sagent": {
        "command": "<python>",
        "args": ["<plugin>/mcp_sagent/server.py"],
        "env": {"SAGENT_ROLE": "tl"}
      }
    }
  }

The ``sagent`` key becomes the namespace prefix the model sees, so
tools appear as ``mcp__sagent_chat__sagent_send`` etc. Don't rename without
also updating the role onboarding text.
"""

from __future__ import annotations

from importlib.resources import files as _pkg_files
from pathlib import Path

import json
import sys


# The MCP server is launched as a module (``python -m
# agent_team.mcp_sagent.server``) rather than by file path, so it works when
# the package is INSTALLED (a relative import inside ``server.py`` would fail
# if it were run as a bare script). This is the canonical spawn form; see
# :data:`SERVER_MODULE` and ``write_role_config`` / ``common._sagent_mcp_server_entry``.
SERVER_MODULE = "agent_team.mcp_sagent.server"

# Filesystem path to ``server.py`` inside the (installed) package — kept for
# debugging / forensics only. Resolved via importlib.resources so it points at
# the installed location, NOT a source-tree path. Not used for the spawn
# command (that uses ``-m SERVER_MODULE``).
SERVER_SCRIPT = Path(_pkg_files("agent_team") / "mcp_sagent" / "server.py")

# Per-role mcp.json files live in the data dir, NOT the plugin code
# dir. Resolved via :mod:`agent_team.mcp_sagent.delivery` so this module
# agrees with where ``main.jsonl`` and the trace files land.
from . import delivery


SESSIONS_DIR = delivery.SESSIONS_DIR


def write_role_config(
    role: str,
    *,
    python: str | None = None,
    serve_url: str = "http://127.0.0.1:8767",
) -> Path:
    """Write ``sessions/<role>.mcp.json`` with the role + serve URL baked in. Returns the path."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    out = SESSIONS_DIR / f"{role}.mcp.json"
    config = {
        "mcpServers": {
            "sagent_chat": {
                "command": python or sys.executable,
                "args": ["-m", SERVER_MODULE],
                "env": {
                    "SAGENT_ROLE": role,
                    "SAGENT_HTTP_URL": serve_url,
                },
            }
        }
    }
    out.write_text(json.dumps(config, indent=2))
    return out


def warmup_env_override(role: str) -> dict[str, str]:
    """Env additions for the warmup window: suppress audit writes.

    Set ``SAGENT_SUPPRESS_AUDIT=1`` on the CLI subprocess's env so
    the MCP server's tool handlers skip ``append_record`` and
    ``inbox.push_back`` during the bootstrap turn. ``runtime.build``
    is responsible for clearing this env var after warmup completes.

    Currently a no-op stub: env vars on a subprocess can't be flipped
    after launch. We instead use the in-process delivery layer's
    suppression toggle (set by :mod:`runtime.build` directly, bypassing
    the env-var hop). Kept here as a marker for the contract.
    """
    return {"SAGENT_SUPPRESS_AUDIT": "1"}
