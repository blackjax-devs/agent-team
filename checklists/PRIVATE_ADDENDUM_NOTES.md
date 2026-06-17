# Private Addendum Notes — what was stripped from the public checklists

This manifest lists everything removed from the BlackJAX-team internal checklists when producing the public, portable versions in this directory. Each entry names the **source file + section/line** and the **bucket** it falls in, so `claude-config` can host these as a private addendum that re-attaches the repo-/machine-specific specifics on top of the public methodology.

- **Bucket A** = blackjax/repo-specific (repo names, paths, `make` targets, marker discipline specifics, monorepo git, per-repo test invocations, findings/PR numbers, worklog substrate mechanics).
- **Bucket B** = machine/environment-specific (OOM/memory limits, `systemd`/cgroup tactics, absolute `/home/jp/...` paths, resource tuning, hardware assumptions).

Public versions KEEP all process discipline (genericized) and ALL Bayesian methodology. BlackJAX *API examples* were kept as illustrations (default profile is blackjax-aware) but made non-load-bearing.

---

## AGENT_CHECKLIST.md

### Bucket A — repo/tooling-specific
- **§ 0 Identify the repo** — the concrete 3-repo table (`blackjax/` → `pytest -n auto`; `tuningfork/` → `make test-fast` etc.; `sampling-book/` → light `pytest`); the "monorepo root is NOT a git repo / `git status` exits 128" specifics; `cd <subdir> && git …` relative-form rule. Public version genericized to "identify which project / its documented test runner / whether the launch dir is a git repo."
- **§ 1** — `Read tuningfork/CONTRIBUTING.md` for marker discipline/fixtures; `cd <subdir> && git status --short` exact form.
- **§ 2 Anti-narration** — the "(n=4 since P4.8)" instance counter.
- **§ 2 Scratch-file rule** — the `META-007 / META-013` lesson IDs.
- **§ 2 Monorepo git rule** — entire subsection: `cd tuningfork && git status`, the pre-approved-verb list "per subdir", mutating-verb approval gate. Public version kept the read-only-vs-mutating distinction without the per-subdir/monorepo framing.
- **§ 4 Test invocation — blackjax/** block: `JAX_PLATFORM_NAME=cpu uv run pytest -n auto -vv --benchmark-disable tests/`; the `tests/mcmc/test_sampling.py -k test_hmc` example.
- **§ 4 tuningfork `make` targets** — the entire `make test-fast / test / test-slow / test-e2e / test-full / clean-orphans` table + "NEVER bare `pytest -n auto`" + the per-target descriptions + `-n 2` / `-n 1` worker counts. (Genericized to "use the project's documented, possibly worker-capped test command.")
- **§ 4 / § 4.5 / § 7.5** — `JAX_PLATFORM_NAME=cpu` env var mandate (blackjax-specific); the `tuningfork`-specific `make test-slow` HARD RULE framing tied to specific targets.
- **§ 4.5** — citations to specific dated incidents ("2026-05-26", "PR #75", "792 passed, 14m29s", "[gwN] node down… exit 144") and the `xdist`/`-n 2`/`-n0`/`-n auto` exact worker numbers as load-bearing prescriptions. Public keeps the *pattern* (background, serial on shared box) without the exact dated incident ledger.
- **§ 4.5 trial-run budget** — the `gp_regression` `n_warmup=1000, n_samples=1000, max_doublings=12` worked example, the "Phase 0 2026-05-12" date, "MCLMC rather than continued NUTS tuning" specifics, "user-policy 2026-05-12" reference.
- **§ 4.6 smoke-E2E** — the `gp_regression × laplace_mhmc recert` worked example, `run_recipe_to_idata`, `az.convert_to_inference_data` / arviz 1.x specifics, the "(n=4+, last 2026-05-28)" counter, "36.2 min" figures, the `worklog/lessons/subagent-behaviour/2026-05-24-...md` link. (Kept the generic "stale-API-after-upgrade" failure-mode description.)
- **§ 5 Pre-sweep orphan check** — `META-014`; the exact orphan signature `python -u -c "import sys;exec(eval(sys.stdin.readline()))"`; the `pgrep -af 'python.*sys;exec(eval' && pkill -9 ...` command; `make clean-orphans`. (Genericized to "clean up stray processes / prefer the project's clean-up target.")
- **§ 6 Marker discipline (tuningfork only)** — the entire `fast`/`slow`/`e2e` marker table with wall-time thresholds, `pytestmark`, `@pytest.mark.requires_posteriordb`, the exact discipline-gate command `pytest tests -m "not fast and not slow and not e2e" --collect-only`, and the `tests/fixtures.py` helper names (`make_rng`, `rng_key`, `mvn_5d_logdensity`, `mvn_5d_init`). (Kept the generic "if the project uses speed/scope markers, one per test + empty-uncategorized gate" pattern.)
- **§ 7.5 Before merging a PR** — the exact touched-path list (`tuningfork/warmup/`, `tuningfork/base_method/`, `tuningfork/recipes/_recipe_runner.py`, `tuningfork/recipes/_templates/{warmups,samplers}/`, `blackjax.<symbol>` call sites); the `gh -R blackjax-devs/tuningfork run list --workflow test-slow ...` command; the META evidence block (blackjax PR #923 rename, PRs #41/#42/#43, recovered in #46, `tuningfork/warmup/window_adaptation_low_rank_imm.py:184`). (Kept the generic upstream-rename-audit gate.)
- **Reference index** — `<repo>/CLAUDE.md`, `tuningfork/CONTRIBUTING.md`, `WORKLOG.md` "always read at session start" pointer.
- **Throughout** — every `worklog/decisions/...` and `worklog/lessons/...` substrate link; "TL"/"SWE"/"junior-swe"/"statistician"/"tech-writer" role names where load-bearing (softened to "the lead" / role-neutral in public).

### Bucket B — machine/environment-specific
- **§ 4 / § 4.5** — the "8 workers OOM-kill 16 GB sessions" (META-010), "orphan Python REPLs hold GBs" (META-014), "8-core shared box already hosting 4 agents", "memory-tight box shared with a concurrent agent OOM-kills a worker", "roughly doubles peak memory" — all specific memory/core/OOM hardware claims. (Kept the *generic* "parallelism multiplies peak memory; run serial on a shared box" guidance without the GB/core numbers.)
- **§ 4 PYTHONUNBUFFERED** — "4 KB buffer", "~60 min sweep" specifics (kept the generic block-buffering explanation).
- **§ 5** — "7.7 GB RSS for 11 min on 2026-05-10 and OOM-killed 3 sessions" memory figures.
- **§ 4.5 HARD RULE** — "(user-mandated 2026-05-29, after repeated OOM crashes)", "crashed both a TL and a SWE session by OOM on 2026-05-29" — dated machine-crash ledger.
- **Monitor tool / `run_in_background` harness specifics** — kept generically ("launch in background, harness re-invokes you") but the BlackJAX-team's exact tool names / `systemd --scope` style tactics (none were explicitly present but the cgroup/scope guidance class belongs here if added later).
- **Absolute paths** — all `/tmp/slow_gate.log`-style examples kept as generic `/tmp/...`; any `/home/jp/...` absolute paths (none survived) would go here.

---

## STATISTICIAN_BAYESIAN_WORKFLOW.md

This file was almost entirely portable methodology and is kept nearly verbatim. Stripped items:

### Bucket A — repo-specific
- **Step 5 Pathfinder pattern** — "consult `blackjax/vi/pathfinder.py` for the current return type" → genericized to "the sampler's `pathfinder` module".
- **Step 1 / References SBC note** — "the statistician's per-task budget is too tight for it" (internal role-budget framing) → softened to "reserve it for when the per-task budget allows".
- BlackJAX API calls (`blackjax.window_adaptation`, `blackjax.nuts`, `blackjax.vi.pathfinder`, `blackjax.diagnostics.*`) were **KEPT** as illustrations per the task spec, with an added header note that the methodology is library-agnostic.

### Bucket B
- None. (No memory/hardware/absolute-path content in this file.)

---

## STATISTICIAN_DIAGNOSTICS_RECIPE.md

Mostly portable methodology, kept near-verbatim. Stripped items:

### Bucket A — repo-specific
- **Operational-rule header** — kept; the "cert"/"groundtruth"/"benchmark reference-sample" framing was generalized ("a benchmark groundtruth" kept as an *example* of a downstream artifact, not as the project's own gate).
- **Lens 1 case-study block** — the two `worklog/lessons/case-studies/stoch_vol/2026-05-12-*.md` links. Converted to inline prose case study (kept the lesson, dropped the path).
- **Lens 2 case-study** — eight-schools `worklog`-style link → inline prose.
- **§ 2 R̂ implementation note** — "`blackjax.diagnostics.potential_scale_reduction` is the classic Gelman-Rubin" naming + "BlackJAX gap tracked at blackjax-devs/blackjax#912" issue link. Genericized to "some libraries' default R̂ helper computes classic G-R — check which one yours computes." (Kept the entire classic-vs-modern calibration explanation + `arviz.rhat` recommendation.)
- **§ 2 ESS** — "`blackjax.diagnostics.effective_sample_size` returns" → "a bulk/tail ESS helper returns".
- **§ 2 Divergences** — "The cert protocol uses a 0.1% threshold" → "a reasonable inspection threshold is ~0.1%".
- **§ 2 Tree-depth case study** — the `worklog/.../gp_regression/2026-05-12-under-warmupped-imm-203d-latent.md` link + horseshoe link → inline prose.
- **§ 5 Case studies** — the entire `worklog/lessons/case-studies/` substrate-link list + contribution-guide / README cross-link mechanics. Converted to inline prose anchors (kept the lessons, dropped the worklog paths + write-up workflow specifics).
- **§ 6a Paper-to-code mapping** — "`blackjax/mcmc/hmc.py:87`" concrete file:line example → generic `<file>:<line>`.
- **§ 6b** — "checking whether the **BlackJAX implementation** matches the paper" → "the implementation". (The NumPyro/Stan/TFP cross-reference table is general and was KEPT.)
- **§ 7 Test robustness** — "`BlackJAXTest.next_key()` in the BlackJAX test suite", "(when writing blackjax tests)" → genericized to "a shared test-suite RNG helper / `next_key()`-style fixture". (The `blackjax.nuts` code example was KEPT as illustration.)

### Bucket B
- None. (The ~50 h CPU wall-time case study in § 5 was kept as a *statistical-vs-wall-time* methodology lesson, genericized; no machine-specific resource tuning or absolute paths were present.)
