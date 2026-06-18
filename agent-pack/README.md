# Agent Pack — a Bayesian-aware Claude Code team for your repo

The **agent pack** is the *native Claude Code* surface of `agent-team`: a
drop-in set of subagent definitions you install into your **own** repository so
that Claude Code gives you a Bayesian-aware engineering team — a Tech Lead that
plans and fans out to specialists.

It's for **Persona B**: *"I'm building or debugging an app that uses
[BlackJAX](https://blackjax-devs.github.io/blackjax/), and I want help that
understands sampling, not just code."*

## What it is (and how it differs from the channel)

There are two ways to use `agent-team`:

| | **Agent pack** (this) | **sagent channel** |
|---|---|---|
| Surface | Native Claude Code subagents | A long-running multi-agent server |
| Topology | **Hierarchical + ephemeral** — one entry agent (`tl`) spawns short-lived subagents via the **Task** tool, each finishes and returns | **Peer + persistent** — named agents in a shared inbox message each other (`@swe`, `@statistician`) |
| Where it runs | Inside *your* repo, on demand | A standing deployment |
| Server? | **No** | Yes |
| Best for | Finalizing a PR, debugging a sampler, adding a feature in your own codebase | An always-on team coordinating ongoing work |

The pack is the lightweight option: no server, nothing standing. You run
`claude --agent tl ...`, the Tech Lead plans, spawns the subagents it needs, they
do their bit and disappear, and you get a synthesized answer.

## The team

| Agent | Model | Role |
|-------|-------|------|
| `tl` | opus | **Entry point.** Plans, decides architecture, delegates to the others via the Task tool, synthesizes results. |
| `swe` | sonnet | Implements multi-file work; runs the implement → commit → test → fix loop. |
| `junior-swe` | haiku | Simple, well-scoped edits; escalates to `swe` when a task grows. |
| `statistician` | sonnet | Algorithm correctness, math-to-code review, and Bayesian-workflow diagnosis (divergences, R-hat, ESS, geometry, tuning). Points at the methodology checklists. |
| `tech-writer` | haiku | Docstrings, guides, notebook QA — the final documentation gate. |

The statistician ships with two methodology checklists it is told to read before
any investigation:

- `STATISTICIAN_BAYESIAN_WORKFLOW.md` — the procedural workflow (prior predictive
  → fit → read diagnostics → reparameterize → tune → recommend).
- `STATISTICIAN_DIAGNOSTICS_RECIPE.md` — how to *read* a chain (traceplots, the
  diagnostic hierarchy, the reparameterize-before-tuning rule).

A shared `AGENT_CHECKLIST.md` (process discipline) installs alongside them.

## Install

`setup.sh` installs the pack **into a target repo** — the repo you want the team
to work on. Pass that repo as the first argument (the `TARGET`), and set
`--workspace`:

```bash
bash /path/to/agent-pack/setup.sh  <TARGET-REPO>  --workspace "DESCRIPTION"
```

It copies the agent defs into `<TARGET>/.claude/agents/` and the methodology
checklists into `<TARGET>/.claude/checklists/`. Idempotent and non-destructive:
it refuses to overwrite without `--force` and never deletes anything.

> **`TARGET` is the repo you install *into*, resolved relative to your cwd.** So
> from `~/test1`, `setup.sh blackjax …` installs into `~/test1/blackjax/`. The
> script **refuses to install into the pack's own source dir** — a common slip
> when you run it from *inside* `agent-pack/` with no target (it would otherwise
> copy the agent defs onto themselves). Run it from your repo root, or pass an
> explicit target.

### `--workspace` — set it (it scopes the whole team)

`--workspace "DESC"` fills the `{{WORKSPACE}}` placeholder in every agent prompt,
telling the team *what it's working on*. This drives how the Tech Lead routes the
work, so a vague or wrong description mis-scopes everything. Omit it and you get a
generic "a repository that uses BlackJAX" line.

- **An app that uses BlackJAX** (the common case):
  `--workspace "my-app, a Stan-to-BlackJAX port of a hierarchical model"`
- **The BlackJAX library itself** (e.g. reviewing a sampler PR) — say so explicitly,
  or the agents assume you're debugging a *user's* app, not library internals:
  `--workspace "the BlackJAX probabilistic-programming library (JAX-based MCMC/VI)"`

Verify it took: the top of `<TARGET>/.claude/agents/tl.md` should echo your
description.

### Other options

```bash
bash setup.sh ../my-app --force                 # overwrite an earlier install
bash setup.sh ../my-app --docs                  # checklists under docs/agent-checklists/ instead
bash setup.sh                                    # install into the current dir (run from your repo root)
```

## Invoke

From inside the target repo:

```bash
# Entry point — the Tech Lead plans and fans out to the specialists:
claude --agent tl "finalize the open PR on this branch"
claude --agent tl "debug why my NUTS sampler diverges on the eight-schools model"
claude --agent tl "add a non-centered reparameterization helper and tests"

# Or talk to a specialist directly:
claude --agent statistician "review this MCLMC kernel against the paper"
claude --agent tech-writer "QA the docstrings in src/inference/ before I merge"
```

The Tech Lead is the recommended entry: hand it a goal, and it decides which
specialists to spawn, gives each a scoped brief, and assembles their output. For
a sampling problem it routes to the statistician (geometry/diagnostics) rather
than treating it as a plain code bug — which is the whole point of a
Bayesian-aware team.
