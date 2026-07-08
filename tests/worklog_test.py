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
"""Tests for the three worklog hooks: worklog-lint, worklog-index, worklog-tags.

Each hook is exercised against a fixture mini-worklog: one happy path and
one failure case per hook.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixture: mini-worklog root
# ---------------------------------------------------------------------------


def make_mini_worklog(tmp_path: Path) -> Path:
    """Create a minimal but valid worklog tree under tmp_path."""
    root = tmp_path

    # worklog/README.md — exempt from frontmatter requirement
    (root / "worklog").mkdir()
    (root / "worklog" / "README.md").write_text("# Worklog\n")
    (root / "worklog" / "INDEX.md").write_text("# Worklog Index\n")

    # WORKLOG.md — the dashboard
    (root / "WORKLOG.md").write_text(
        textwrap.dedent("""\
        ---
        status: CURRENT
        date: 2026-01-01
        tags: [worklog, process]
        model: null
        author: tl
        supersedes: []
        related: []
        ---

        # Dashboard
        """)
    )

    # One valid thread
    threads = root / "worklog" / "threads"
    threads.mkdir()
    (threads / "my-feature.md").write_text(
        textwrap.dedent("""\
        ---
        status: CURRENT
        date: 2026-01-02
        tags: [process, worklog]
        model: null
        author: tl
        supersedes: []
        related:
          - WORKLOG.md
        ---

        # My feature thread
        """)
    )

    return root


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_hook(module: str, root: Path, extra_args: list[str] | None = None) -> tuple[int, str]:
    """Run a worklog hook module and return (returncode, combined output)."""
    cmd = [
        sys.executable,
        "-m",
        module,
        "--root",
        str(root),
    ]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout + result.stderr


# ---------------------------------------------------------------------------
# worklog-lint
# ---------------------------------------------------------------------------


class TestWorkloglint:
    def test_happy_path(self, tmp_path):
        root = make_mini_worklog(tmp_path)
        rc, out = run_hook("agent_team.worklog.tools.lint_worklog", root)
        assert rc == 0, f"Expected clean lint, got:\n{out}"

    def test_missing_frontmatter_fails(self, tmp_path):
        root = make_mini_worklog(tmp_path)
        bad = root / "worklog" / "threads" / "no-frontmatter.md"
        bad.write_text("# No frontmatter here\n")
        rc, out = run_hook("agent_team.worklog.tools.lint_worklog", root)
        assert rc == 1, "Expected lint failure for missing frontmatter"
        assert "missing YAML frontmatter" in out

    def test_invalid_status_fails(self, tmp_path):
        root = make_mini_worklog(tmp_path)
        bad = root / "worklog" / "threads" / "bad-status.md"
        bad.write_text(
            textwrap.dedent("""\
            ---
            status: INVALID_STATUS
            date: 2026-01-03
            tags: [process]
            model: null
            author: tl
            supersedes: []
            related: []
            ---
            # Bad status
            """)
        )
        rc, out = run_hook("agent_team.worklog.tools.lint_worklog", root)
        assert rc == 1
        assert "invalid status" in out


# ---------------------------------------------------------------------------
# worklog-index
# ---------------------------------------------------------------------------


class TestWorkloglndex:
    def test_happy_path_generates_index(self, tmp_path):
        """INDEX.md is generated and contains expected sections."""
        root = make_mini_worklog(tmp_path)
        index_path = root / "worklog" / "INDEX.md"
        index_path.unlink()  # remove the dummy so the tool creates the real one

        rc, out = run_hook("agent_team.worklog.tools.build_index", root)
        # First run: INDEX.md was missing or stale -> exits 1 and writes it
        assert index_path.exists(), "INDEX.md was not created"
        content = index_path.read_text()
        assert "# Worklog Index" in content
        assert "## Recent changes" in content

    def test_check_mode_stale(self, tmp_path):
        """--check exits non-zero when INDEX.md is absent."""
        root = make_mini_worklog(tmp_path)
        (root / "worklog" / "INDEX.md").unlink()

        rc, out = run_hook("agent_team.worklog.tools.build_index", root, ["--check"])
        assert rc == 1, "Expected non-zero exit for missing INDEX.md"

    def test_check_mode_up_to_date(self, tmp_path):
        """--check exits 0 when INDEX.md is current."""
        root = make_mini_worklog(tmp_path)
        # First write a correct INDEX.md
        run_hook("agent_team.worklog.tools.build_index", root)
        # Now check — must be up to date
        rc, out = run_hook("agent_team.worklog.tools.build_index", root, ["--check"])
        assert rc == 0, f"Expected up-to-date, got:\n{out}"


# ---------------------------------------------------------------------------
# worklog-tags
# ---------------------------------------------------------------------------


class TestWorklogTags:
    def test_happy_path_no_changes(self, tmp_path):
        """--check exits 0 when no files need tag changes."""
        root = make_mini_worklog(tmp_path)
        rc, out = run_hook("agent_team.worklog.tools.normalize_tags", root, ["--check"])
        assert rc == 0, f"Expected no tag drift, got:\n{out}"

    def test_check_fails_on_unnormalized_tags(self, tmp_path):
        """--check exits 1 when lintrc substitution would change tags."""
        root = make_mini_worklog(tmp_path)
        # Create a lintrc that substitutes "process" -> "workflow"
        lintrc = root / "worklog" / ".lintrc.yaml"
        lintrc.write_text("tag_substitutions:\n  process: workflow\n")

        rc, out = run_hook("agent_team.worklog.tools.normalize_tags", root, ["--check"])
        assert rc == 1, "Expected tag-drift failure with substitution in lintrc"
        assert "process" in out or "workflow" in out

    def test_rewrite_applies_substitutions(self, tmp_path):
        """Without --check, tags are rewritten in-place per lintrc."""
        root = make_mini_worklog(tmp_path)
        lintrc = root / "worklog" / ".lintrc.yaml"
        lintrc.write_text("tag_substitutions:\n  worklog: log\n")

        thread = root / "worklog" / "threads" / "my-feature.md"
        original = thread.read_text()
        assert "worklog" in original

        rc, out = run_hook("agent_team.worklog.tools.normalize_tags", root)
        assert rc == 0
        rewritten = thread.read_text()
        assert "worklog" not in rewritten
        assert "log" in rewritten
