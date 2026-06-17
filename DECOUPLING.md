# Extraction / decoupling work map (M2)

This repo was scaffolded by copying the role-agnostic framework from the
BlackJAX team's internal channel (`claude-config/sagent-channel/`). The copied
code still has BlackJAX couplings + assumes a flat in-repo layout. This file
tracks the work to make it a clean, config-driven public library.

Plan of record: `claude-config/project/worklog/decisions/2026-06-17-m2-ai-dev-team-public-library.md`.

## Status (2026-06-17) — core decoupling DONE ✅
**§1 (profile loader), §2 (all couplings), §3 (roster/single-agent), and the §6
sanitization sweep are complete + adversarially verified** (`team_profile.py`
loader; statistician sandbox, session namespace, models, peer list, and role
prompts all profile-driven; `serve.py` builds only the roster; single-agent
roster boots). Verified: **0 BlackJAX coupling hits** in framework code
(`bin/ mcp_sagent/ roles/ runtime/ sandboxed_tools.py web/`); loader + render
+ build + serve-import smoke pass; 20 tests pass. `roles/*.md` moved to
`profiles/default/roles/*.md` with a `{{workspace}}` template.

**Remaining:** §4 (methodology checklists — public general core + private
addendum, **incl. routing OOM/machine-specific guidance to private**), §5 (agent
pack), §6 *packaging restructure* (flat → package + console script), §7 (sample
app + further prompt generalization), §8 (nightly sagent CI + clean-clone verify).

## 1. Profile mechanism (the core of M2)
- [ ] Add a profile loader: read `AI_DEV_TEAM_PROFILE_DIR` → parse `team.toml`
      (`profiles/default/team.toml` defines the schema). Expose roster, models,
      workspace, sandbox_root, session_id_namespace, role prompts.
- [ ] Route everything below through it (no hardcoded BlackJAX values).

## 2. Kill the hardcoded couplings
| File | Coupling | → becomes |
|---|---|---|
| `roles/statistician.py` | `_monorepo_root()` `parents[3]` walk + hardcoded `tuningfork/experiments` sandbox | `workspace_root` / `sandbox_root` from `team.toml` (default = launch cwd) |
| `roles/common.py` | MODEL map + uuid5 namespace `"blackjax-chat:<role>"` | from `[models]` + `session_id_namespace` |
| `roles/*.md` | "You are X on the **BlackJAX monorepo** (`blackjax/ sampling-book/ tuningfork/`)" | a templated `{{workspace}}` block filled from `[workspace].description` + repos |
| `bin/serve.py` | role registry hardcoded to the 5 | driven by `[roster]` (enables single-agent) |
| `web/` | role colors/labels | from the roster/profile |

## 3. Roster / single-agent (S-decision)
- [ ] `serve.py` builds only the roles in `roster` (1..N). Handle the no-peers
      case (a single agent talks to the operator only — no peer-messaging targets).
- [ ] Presets (`dev-team`, `statistician-only`) selectable via CLI flag / env.

## 4. Methodology (the public value)
- [ ] Bring in the general checklists from claude-config (sanitized): the
      general core of `AGENT_CHECKLIST.md` + `STATISTICIAN_BAYESIAN_WORKFLOW.md`
      + `STATISTICIAN_DIAGNOSTICS_RECIPE.md`. Carve BlackJAX-specific bits into a
      private addendum that stays in claude-config.
- [ ] Wire them as role mandatory-reads.

## 5. Agent pack (Claude Code native surface)
- [ ] `agent-pack/`: sanitized generic `.claude/agents/*.md` + a setup script
      that drops them + the checklists into a target repo. (Distinct from the
      sagent channel — gives ephemeral hierarchical subagent fan-out.)

## 6. Packaging / restructure
- [ ] Decide: keep the flat layout (`bin/ mcp_sagent/ roles/ runtime/`) or move
      to a proper package (`ai_dev_team/`). The latter unlocks a clean
      `[project.scripts]` console entry (`ai-dev-team serve …`).
- [ ] Sanitize: `grep -ri 'blackjax|tuningfork|/home/jp|@email'` → only
      generic/templated hits before going public.

## 7. Default profile content
- [ ] Write the sanitized `profiles/default/roles/*.md` (generic Bayesian-aware
      prompts; the copied `roles/*.md` are the BlackJAX-flavored starting point).
- [ ] A 1-file sample app so a clean clone boots + coordinates out of the box.

## 8. CI + verification
- [ ] **Nightly sagent CI** (build against the git-ref / `main`, run the e2e
      smoke + provider conformance) to catch API drift.
- [ ] Clean-clone verification (both surfaces; `statistician-only` + `dev-team`).
- [ ] Then: create the public `blackjax-devs/ai-dev-team` repo + push.

## Dependency note
sagent is pinned to a **git-ref** (merged-main `06603e2`) because PyPI `0.1.6`
lags the #177 provider work. Repin to `sagent>=0.1.7` once a release ships it.
