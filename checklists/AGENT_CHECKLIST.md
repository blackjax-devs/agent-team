# Agent Checklist — universal process rules

**Read this file at every checkpoint below. Do not rely on memory — a `Read` call is a hard attention anchor; recall is not.**

This file is the source of truth for the process boilerplate that would otherwise be repeated inline in every spawn brief. Spawn briefs point here instead of repeating the rules.

---

## Checkpoints — when you MUST re-read this file

1. **At task start** — before your first tool call.
2. **Before each `git commit`** — verify lint + commit-message format + relevant gates.
3. **Before reporting back to the lead** — verify final-report format + all gates passed.

---

## 0. Identify the repo

A workspace may contain more than one independent project. Before editing, know **which** project you are in and what its conventions are:

- What is the **test runner** for this project? (It is not always `pytest -n auto` — some suites cap worker count or scope the suite behind a task runner. Find the project's documented command.)
- Is the directory you launched in actually a git repo, or is it a parent directory holding several sibling repos? (`git status` at a non-repo root exits 128.) Run git verbs against the specific project directory, not a wrapper directory.
- Does the project have its own contributor doc (`CONTRIBUTING.md`, `CLAUDE.md`, etc.)? Read it before touching code.

The general rule: **never assume the command from one project carries to another.** Look it up per project.

---

## 1. Before starting a task

- **Read this file** — yes, even if the spawn brief reminded you.
- **Identify the repo** (§ 0). If the project has a contributor doc, read it for content rules (test layout, fixtures, any tagging/marker discipline).
- **Check git state**: `git status --short && git log --oneline -3` in the target project. If the worktree is dirty or HEAD is unexpected, surface to the lead before editing.
- **Note the file ownership boundaries** the spawn brief gives you (edit-list + don't-touch list). Stick to them.

---

## 2. Implementation discipline

### Anti-narration rule
Do NOT write transitional narration between tool calls — no "Now let me…", "Good, now I'll…", "Perfect! The linter ran…". Just emit the next tool call. Narration correlates with the harness's end-of-turn heuristic firing; it has truncated work mid-task. Emit results, not commentary.

### Scratch-file rule
Any multi-line Python — **especially with `#` comments** — goes into a scratch file (e.g. `/tmp/foo.py`), then run that file. NEVER multi-line `python -c "..."` containing `#` (path-validation guards can reject it). Single-line `python -c "..."` is OK.

### Two-commit structure
For each logical task, prefer **two commits**: commit-1 = code + tests; commit-2 = registration / wiring (`__init__.py`, registry, exports). Bisect-clean, easy to review.

### Git hygiene
Use read-only git verbs (`status`, `diff`, `log`, `branch`, `show`, `blame`, `ls-files`, `rev-parse`) freely. Treat mutating verbs (`commit`, `push`, `reset`, `checkout`, `merge`) as requiring deliberate intent — confirm before running them.

### File ownership
The spawn brief lists exactly which files you may edit and which you must not touch. If you discover the brief is wrong (e.g., a file outside your list needs edits), STOP and surface to the lead — do not silently expand scope.

---

## 3. Before every commit

- **Lint / pre-commit (no `--no-verify`)**: run the project's pre-commit / lint suite (commonly `uv run pre-commit run --all-files`). Fix every failure before committing.
- **Commit message format**:
  ```
  <imperative summary in present tense>

  Finding: <what surprised you / what was wrong / what you discovered>
  Fix: <what you changed and why>

  Co-Authored-By: <model name> <noreply@anthropic.com>
  ```
  The `Finding/Fix` lines are mandatory — they enrich history with trial-and-error context.

---

## 4. Test invocation

**Use the project's documented test command.** Do not assume `pytest -n auto` is safe everywhere — some suites deliberately cap worker count or wrap the test run behind a task runner for good reasons. Look up the project's command (§ 0) before running it.

For a single file or `-k` filter, prefer single-process over a parallel runner — it is faster to start and easier to read.

### Long-running scripts: unbuffered output is mandatory

When invoking a long-running script and redirecting stdout to a file or `tee`, you MUST disable output buffering (`PYTHONUNBUFFERED=1`, or `python -u`):

```bash
PYTHONUNBUFFERED=1 <runner> long_script.py > log 2>&1 &
```

Python defaults to **block-buffered** output when stdout is a pipe (not a TTY). Per-iteration `print(...)` lines sit in a buffer until the process exits — defeating the purpose of `tee` and hiding real-time progress + warnings during the run. A multi-minute sweep can produce only the build prelude in the log because per-iteration prints sat unflushed.

Belt-and-suspenders: scripts that print progressively should also call `sys.stdout.flush()` after each per-iteration print so the env-var dependency is not load-bearing.

### 4.5. Long-running jobs: background execution (never block on polling)

Test suites, benchmarks, and any process expected to run longer than ~10 seconds **must** run in the background. Never block a turn on synchronous execution of a long-running command.

**Anti-patterns (observed repeatedly):**

1. **Synchronous test-suite invocation**: a parallel test run forked synchronously in a turn — the worker blocks for the whole run, and on a shared box the parallel workers can exhaust memory.
2. **Self-matching `pgrep -f` wait-loop**: `pgrep -f "<pattern>" && pkill ...` where `<pattern>` appears in the polling command's own process line → the command matches itself → the loop never exits.
3. **Self-matching `until ! pgrep` blocking wait-loop**: `until ! pgrep -f "<pattern>"; do sleep 10; done` run *synchronously* — the shell running the loop has `<pattern>` in its own command line, so `pgrep -f` always matches itself, the negation is never true, and the worker blocks forever. This is instance 2 recurring under pressure. **The durable fix is the prescriptive pattern below: never hand-roll a wait-loop at all.**

**THE ONE SANCTIONED WAY to run a long job / test gate (copy this; do not improvise):**

Do NOT write any `until …; do sleep; done`, any `pgrep`/`pkill` wait-loop, or any synchronous blocking command for a >10 s job. Instead, **launch the job in the background** and let the harness re-invoke you on exit — that is the *only* polling mechanism you need:

```
# background job
PYTHONUNBUFFERED=1 <test/sweep command> 2>&1 | tee /tmp/gate.log
```

The harness notifies you when it exits; you then `Read /tmp/gate.log` for the result. If you need a *bounded* wait for a specific condition, use a monitor that exits on the condition — never a foreground `until` loop. If you ever find yourself typing `pgrep`, `pkill`, or `until` into a foreground command, **stop** — you are re-walking instance 2/3.

**Procedure:**

- Run any job expected to take >10 s in the background. **Disable output buffering** (see § 4) so buffered output doesn't prevent log-marker polling from seeing completion signals.
- Poll by **file marker** (touch a sentinel file when done) or **log patterns** (`grep` the output file for a completion message), never by `pgrep -f` of a pattern in the polling command.
- If you must `pgrep`, use a pattern strictly disjoint from the polling command's own name/args.
- For test suites: prefer the project's scoped/worker-capped command; on a shared box, run serial rather than maximally parallel — parallelism roughly multiplies peak memory and can crash a near-complete run.

> **HARD RULE — heavy test gates run in the background, serial on a shared box.**
> A heavy/slow test gate MUST run in the background — **never inline/synchronously**, by ANY role. **When another agent/session may be active on the box, run the gate serially, not maximally parallel.** Parallel workers roughly multiply peak memory; on a memory-tight shared box that can kill a worker near completion. Serial is slower wall-time but robust — prefer correctness over speed for a merge gate. Serialize compute: one heavy job on the box at a time; the lead gates this.

> **`systemd-run --scope` is for HEAVY jobs only — not a default wrapper.** OOM-isolation
> scope-wrapping (`systemd-run --user --scope … -- bash -c '<cmd>'`) is the right tool when a
> background job is genuinely heavy (multi-GB RSS) and an OOM would otherwise cascade up and kill
> the worker. But wrapping a *light* job is harmful: a ~1.2 GB german_credit re-cert wrapped in a
> scope was repeatedly killed by **scope lifecycle management (not OOM)** and had to be relaunched
> as a plain background job to complete. Rule: wrap only jobs you expect to be memory-heavy; launch
> light jobs (≲ a few GB) as plain background processes.

### Trial-run budget: 10 minutes

A **trial run** (downscaled invocation to confirm the approach works before committing to a production-scale wall) should generally finish in ≤ 10 minutes. The point of a trial is to **fail fast** — if a trial takes longer than that, something is wrong that more wall won't fix.

When you launch a trial:

1. **Estimate up front**: extrapolate from prior trials / per-step costs. If the estimate exceeds 10 min, the trial is the wrong size — scale down until the estimate fits.
2. **Check at 5 min**: is the job producing output? Are diagnostics (CPU usage, intermediate prints) consistent with a healthy run?
3. **At 10 min, evaluate**: kill or continue. Continuing past 10 min should be a *deliberate* call, not a *drift* ("I forgot it was running").
4. **At 30 min**, the trial is no longer "fast feedback" — it's a small production run. Surface to the lead or to the user.

Production sweeps (the long ground-truth / production emit runs) are the exception — those can run for hours by design. The 10-min rule applies to **diagnostic trials** specifically.

### 4.6. Smoke the WHOLE script end-to-end at small N before any production-scale launch

For any script whose expected wall is **>30 min**, you MUST **first run the full script at a downscaled N** to verify EVERY code path — compute, post-processing, output write — succeeds. Only then launch at production N.

**Why this is distinct from § 4.5 (trial-run budget):**
- Trial-run-budget catches **compute** failures (won't converge, too slow, deadlock).
- This rule catches **post-processing** failures — a stale API call after a dependency upgrade, a `KeyError` in the metric block, a typo in the output path, a missing dependency for the final write, etc. The compute succeeded; the script crashed AFTER, after hours of wall.

**The failure mode this rule prevents:** a long script launches, the heavy compute completes successfully — and then the script crashes at the next line because of a stale API call, a `float()` of a `None` field, a missing key in a comparison dict, etc. Wall cost: 30 min to many hours of compute lost. The fix is always one line and was always discoverable at small N in 2 minutes.

**Pattern (required):**

1. Copy the production script verbatim → `<script>_smoke_e2e.py`.
2. Override the size knobs to tiny values (e.g., 5 and 20). For multi-phase pipelines, override every phase.
3. Run with the **same env / same imports / same downstream code paths** as production.
4. **Verify**: exit code 0; all output files written; output contents contain the expected fields (not just NaNs / empty dicts); any assertion-driven gates actually evaluate rather than erroring on missing fields.
5. Only after smoke-E2E PASSES → launch the production-N run.

**Persist intermediate artifacts inside the script**: any long-compute script SHOULD persist the heavy compute outputs to disk BEFORE running the post-processing block. That way if a downstream call fails despite the smoke, a fix can re-load the persisted outputs instead of paying the compute again.

**Time cost**: 1-5 min smoke → saves potentially hours of wasted compute.

**Don't conflate with § 4.5**: trial-run-budget is a HARD STOP for diagnostic compute (kill at 10 min). Smoke-E2E is a PRE-FLIGHT for production compute (verify the script flow at tiny N first). Both apply.

---

## 5. Pre-sweep orphan check

Before any heavy run you invoke directly: check for and clean up stray processes left by prior sessions (orphaned REPLs, detached workers). A leaked background process can hold significant memory and starve concurrent sessions. Prefer the project's clean-up target if one exists; otherwise identify the orphan pattern and kill it before launching.

---

## 6. Test tagging discipline (when the project uses test markers)

If the project categorizes tests by speed/scope (e.g. `fast` / `slow` / `e2e` markers), every new test gets exactly **one** category marker; use a module-level default if every test in the file is the same kind. Add any additive capability markers (e.g. "requires external cache") as needed.

**Discipline gate (run after every test-touching commit, if the project uses markers):** collect the tests that match *no* category marker. The set must be empty — if anything appears, you missed a marker. Adapt the exact command to the project's marker names.

Reuse the project's shared fixtures (RNG helpers, toy distributions) rather than reinventing them.

---

## 7. Before final report

- **Re-read this file** (yes, again — the report is the most truncation-prone moment).
- **Format**: keep it tight — a few bullets for implementation roles; a structured review for analysis roles (see the methodology checklists); audit-style for documentation roles.
- **Required content**:
  1. Commit SHAs in order
  2. Validation results (test counts + any discipline gate + lint pass)
  3. Anything unexpected (gotchas, deviations from brief, items for follow-up)
- **Do NOT** narrate next steps. Stop after the required content.

---

## 7.5. Before merging a PR (lead / human gate)

An async heavy gate that runs on push but is NOT a blocking PR-merge check can let a PR land green on the fast checks while still breaking the heavy gate. The lead / merging human is the gate.

**For any PR that touches the project's load-bearing surfaces (the core run path, adaptation/warmup code, templates, or that adds a new call site against an upstream dependency's API):**

1. **Confirm the heavy gate is green on the PR head** — either run it locally on the PR branch, or check the heavy-gate workflow run for the PR's most recent push commit.
2. **For upstream-import-surface changes**: grep all callsites (in both the library and the tests) for the affected symbol before merging — template/example updates don't catch wrapper-internal callsites.
3. **If the heavy gate is unavailable** (no compute budget, infra issue): explicitly annotate the PR body with `[pre-merge heavy gate skipped because <reason>]`. Otherwise: don't merge.

**Why**: an upstream rename-without-alias can go undetected across several PRs when the rename audit checks the example templates but misses a wrapper-internal call that isn't exercised by the fast tests. The wall cost of the pre-merge gate is minutes; the cost of skipping is multi-PR regression debugging the next time an upstream dependency renames or removes a symbol.

---

## 8. When you are stuck

After 2 unsuccessful attempts at the same fix:
1. Write a `BLOCKED.md` in your working directory:
   - What you tried (each attempt + outcome)
   - What you believe the root cause is
   - What additional information would unblock you
2. Stop. Do not loop on the same approach a third time.

`BLOCKED.md` is a hard escalation signal the lead acts on immediately.

---

## Reference index

- **Project rules per repo**: the project's own `CLAUDE.md` / `CONTRIBUTING.md`
- **Bayesian workflow**: `STATISTICIAN_BAYESIAN_WORKFLOW.md`
- **Diagnostics reference**: `STATISTICIAN_DIAGNOSTICS_RECIPE.md`
