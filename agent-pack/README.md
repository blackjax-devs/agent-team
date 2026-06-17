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

From anywhere, point `setup.sh` at the repo you want the team in:

```bash
bash setup.sh /path/to/your/repo
```

This copies the agent defs into `<your-repo>/.claude/agents/` and the
methodology checklists into `<your-repo>/.claude/checklists/`. It is idempotent
and never destructive — it refuses to overwrite an existing file unless you pass
`--force`, and it never deletes anything.

Options:

```bash
bash setup.sh                                   # install into the current directory
bash setup.sh ../my-app                         # install into a sibling repo
bash setup.sh ../my-app --workspace "my-app, a Stan-to-BlackJAX port"
bash setup.sh ../my-app --force                 # overwrite an earlier install
bash setup.sh ../my-app --docs                  # checklists under docs/agent-checklists/ instead
```

`--workspace "DESC"` fills the `{{WORKSPACE}}` placeholder in each agent's prompt
so the team knows what it's working on. Omit it and the agents default to a
generic "a repository that uses BlackJAX" line.

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
