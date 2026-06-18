#!/usr/bin/env bash
#
# setup.sh — install the agent-team AGENT PACK into a target repo.
#
# Drops a Bayesian-aware Claude Code subagent team (tl / swe / junior-swe /
# statistician / tech-writer) into <target>/.claude/agents/, plus the
# statistician's methodology checklists into <target>/.claude/checklists/.
# Then you run the team from the target repo with:  claude --agent tl
#
# Idempotent and non-destructive: it never deletes anything and refuses to
# overwrite an existing file unless you pass --force.
#
# Usage:
#   bash setup.sh [TARGET_DIR] [--force] [--workspace "DESCRIPTION"] [--docs]
#
#   TARGET_DIR              Repo to install into (default: current directory).
#   --force                 Overwrite existing pack files (still never deletes).
#   --workspace "DESC"      Fill the {{WORKSPACE}} placeholder in the agent
#                           defs with DESC (default: a generic line).
#   --docs                  Put checklists under <target>/docs/agent-checklists/
#                           instead of <target>/.claude/checklists/.

set -euo pipefail

# --- resolve this script's own directory (the pack source) -------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AGENTS_SRC="$SCRIPT_DIR/.claude/agents"
# Checklists live one level up, in the repo's shared checklists/ directory.
CHECKLISTS_SRC="$SCRIPT_DIR/../checklists"

# --- defaults ----------------------------------------------------------------
TARGET="$PWD"
FORCE=0
DOCS_MODE=0
WORKSPACE_DESC="this project (a repository that uses BlackJAX for Bayesian inference)"
TARGET_SET=0

# The methodology files that ship with the pack (the public, portable ones).
CHECKLIST_FILES="AGENT_CHECKLIST.md STATISTICIAN_BAYESIAN_WORKFLOW.md STATISTICIAN_DIAGNOSTICS_RECIPE.md"

# --- parse args --------------------------------------------------------------
while [ "$#" -gt 0 ]; do
  case "$1" in
    --force) FORCE=1; shift ;;
    --docs) DOCS_MODE=1; shift ;;
    --workspace)
      [ "$#" -ge 2 ] || { echo "error: --workspace needs an argument" >&2; exit 2; }
      WORKSPACE_DESC="$2"; shift 2 ;;
    --workspace=*) WORKSPACE_DESC="${1#--workspace=}"; shift ;;
    -h|--help)
      sed -n '2,30p' "$0"; exit 0 ;;
    -*)
      echo "error: unknown option '$1'" >&2; exit 2 ;;
    *)
      if [ "$TARGET_SET" -eq 0 ]; then TARGET="$1"; TARGET_SET=1; shift
      else echo "error: unexpected argument '$1'" >&2; exit 2; fi ;;
  esac
done

# --- validate ----------------------------------------------------------------
[ -d "$AGENTS_SRC" ] || { echo "error: cannot find pack agents at $AGENTS_SRC" >&2; exit 1; }
[ -d "$TARGET" ] || { echo "error: target directory does not exist: $TARGET" >&2; exit 1; }
TARGET="$(cd "$TARGET" && pwd)"

# Guard against a common footgun: running this from inside agent-pack/ with no
# TARGET makes TARGET=$PWD=the pack's own source dir, so it would copy the agent
# defs onto themselves ("skip (exists)") and scatter a stray .claude/checklists/
# into the source. Refuse and point at the real usage.
if [ "$TARGET" = "$SCRIPT_DIR" ]; then
  echo "error: TARGET is the agent-pack's own source dir ($TARGET)." >&2
  echo "       Pass the repo you want to install the team INTO, e.g.:" >&2
  echo "         bash $0 /path/to/your-repo --workspace \"the BlackJAX library\"" >&2
  exit 2
fi

if [ "$DOCS_MODE" -eq 1 ]; then
  CHECKLISTS_DST="$TARGET/docs/agent-checklists"
  CHECKLISTS_REL="docs/agent-checklists"
else
  CHECKLISTS_DST="$TARGET/.claude/checklists"
  CHECKLISTS_REL=".claude/checklists"
fi
AGENTS_DST="$TARGET/.claude/agents"

echo "Installing agent-team agent pack"
echo "  target:     $TARGET"
echo "  agents  ->  $AGENTS_DST"
echo "  checklists -> $CHECKLISTS_DST"
echo "  workspace:  $WORKSPACE_DESC"
[ "$FORCE" -eq 1 ] && echo "  mode:       --force (existing files will be overwritten)"
echo

mkdir -p "$AGENTS_DST" "$CHECKLISTS_DST"

# install_file SRC DST  — copy non-destructively, filling {{WORKSPACE}}.
install_file() {
  src="$1"; dst="$2"
  if [ -e "$dst" ] && [ "$FORCE" -eq 0 ]; then
    echo "  skip (exists): ${dst#$TARGET/}   (use --force to overwrite)"
    return 0
  fi
  # Substitute {{WORKSPACE}} and {{CHECKLISTS_PATH}} on the way in. Use a
  # literal-safe sed by escaping each replacement's sed-special chars
  # (& and the / delimiter and backslash).
  esc=$(printf '%s' "$WORKSPACE_DESC" | sed -e 's/[\/&\\]/\\&/g')
  esc_cl=$(printf '%s' "$CHECKLISTS_REL" | sed -e 's/[\/&\\]/\\&/g')
  sed -e "s/{{WORKSPACE}}/$esc/g" -e "s/{{CHECKLISTS_PATH}}/$esc_cl/g" "$src" > "$dst"
  echo "  wrote:         ${dst#$TARGET/}"
}

# --- agents (always {{WORKSPACE}}-substituted) -------------------------------
for f in "$AGENTS_SRC"/*.md; do
  [ -e "$f" ] || continue
  install_file "$f" "$AGENTS_DST/$(basename "$f")"
done

# --- checklists (verbatim copies; no placeholder) ----------------------------
if [ -d "$CHECKLISTS_SRC" ]; then
  for name in $CHECKLIST_FILES; do
    src="$CHECKLISTS_SRC/$name"
    if [ ! -e "$src" ]; then
      echo "  warn (missing source): $name — skipped" >&2
      continue
    fi
    dst="$CHECKLISTS_DST/$name"
    if [ -e "$dst" ] && [ "$FORCE" -eq 0 ]; then
      echo "  skip (exists): ${dst#$TARGET/}   (use --force to overwrite)"
    else
      cp "$src" "$dst"
      echo "  wrote:         ${dst#$TARGET/}"
    fi
  done
else
  echo "  warn: checklist source dir not found ($CHECKLISTS_SRC) — agents installed without methodology files" >&2
fi

# --- usage hint --------------------------------------------------------------
cat <<EOF

Done. The agent team is installed under .claude/ in the target repo.

Now run, from inside $TARGET:

  claude --agent tl "finalize the open PR on this branch"
  claude --agent tl "debug why my NUTS sampler diverges on the funnel model"

The 'tl' (Tech Lead) agent is the entry point — it plans and fans out to
swe / junior-swe / statistician / tech-writer via the Task tool. You can also
invoke a specialist directly, e.g.  claude --agent statistician "review this kernel".
EOF
