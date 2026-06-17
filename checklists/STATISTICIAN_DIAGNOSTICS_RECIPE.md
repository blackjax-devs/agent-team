# Statistician — Model Diagnostics Common Recipe

**This file teaches you how to *read* a chain.** It is not a lookup table from signal to fix. The thresholds and formulas matter, but the interpretive practice matters more — and that practice is contextual. The same number can mean a healthy chain in one model and a broken sampler in another.

For procedural workflow (Step 1 → Step 9), see `STATISTICIAN_BAYESIAN_WORKFLOW.md`. This file is the reference companion you re-read at workflow Step 3 (read the diagnostics) and Step 4 (form hypotheses).

> The code examples use the [BlackJAX](https://github.com/blackjax-devs/blackjax) API as concrete illustrations. The interpretive practice is library-agnostic.

> **Operational rule — don't edit the model when the diagnosis is downstream of the sampler.** When a chain misbehaves, the diagnostic chain of causality usually runs *prior → posterior → metric → step → divergence*. If your diagnosis lands on "the prior is too wide" or "the parameterization is wrong," that's a *modeling* claim, not a *sampling* claim. Before editing the model, ask three things: (1) does the model currently produce scientifically meaningful posteriors? (2) has any downstream artifact — a paper, a benchmark groundtruth, a publication-ready summary — been computed against this exact model? (3) is the geometry pathology you found *known* for this model class (e.g. funnel necks in centered hierarchies, unit-root tails in AR(1))? If the answer to (1) is yes, or to (2) is yes, or to (3) is yes, your response should land on the **sampler / config / gate** side first, not the **model** side. For benchmark / reference-sample projects this hardens into a "model-freeze after groundtruth" rule; for general Bayesian work it's the soft form: editing the model is a one-way door and the fix should be motivated by *modeling* evidence, not by *sampling* convenience. The § 5 worked example is the canonical case — a model edit was tried, made things worse, and was reverted in favour of a sampler-side adjustment.

---

## Three lenses to bring to every chain

Before any specific diagnostic, ground yourself in three questions. Most misdiagnoses are misdiagnoses of the question, not of the math.

### Lens 1 — Location, not count

Five divergences clustered in the same parameter-space tail tell you *exactly* what to fix. Fifty divergences scattered uniformly across the posterior likely signal numerical precision, not the model geometry. The count is a screening signal; the *position* is the diagnostic.

The first thing to do when a run fails on divergences (or rhat, or low ESS for one parameter) is **cluster the failing transitions in the parameter space you wrote the model in**. Compute mean / variance / tail-fraction of each parameter conditional on `is_divergent`, then sort by `|Δ(mean) / σ_non-div|`. The parameters at the top of that list are where the model is fighting the sampler. Often that points at a prior, a transformation, or an initialization — not at the sampler at all.

> **Case study (AR(1) unit-root)**: a 503-D non-centered recursive AR(1) stochastic-volatility model put divergences right at the unit-root of the persistence parameter φ. Two prior architectural diagnoses missed it; the cluster check took 3 minutes and was decisive. And the cluster diagnosis was correct but the obvious-looking fix (tighten the prior) made things *worse* — a correct diagnosis is not automatically a correct fix.

### Lens 2 — Thresholds are starting points, not gates

`R̂ < 1.01`. `bulk-ESS / num_chains > 100`. `E-BFMI > 0.3`. `divergences = 0`. These thresholds are calibrated for the **average well-specified hierarchical model running long enough to be a publication**. They are not laws of physics. Two real situations where treating them as gates wastes turns:

- **A weakly identified hyperparameter** (e.g. a Cauchy scale at the top of a 3-level hierarchy with 8 groups) can have R̂=1.05 at n=4000 simply because the prior is doing most of the work and the chain has nothing to mix toward. You can fail "the threshold" forever without sampling failing.
- **A 503-D state-space model** can produce 0.26% divergences when the prior is wide enough to admit a low-probability tail — those divergences are *correctly* flagging a region of bad geometry, but they aren't preventing coverage. R̂ and ESS confirm coverage is fine.

When a threshold trips, your first question is **"is the threshold meaningful for *this* parameter in *this* model?"** Not "what knob lowers it."

> **Case study (eight-schools)**: a non-centered eight-schools model at a strict-zero divergence gate — a single funnel-neck excursion per 40 000 draws is a *stochastic* feature of the geometry, not a *systematic* failure. The right "fix" is reading the threshold, not chasing it.

### Lens 3 — A signal is always downstream of something else

Every diagnostic is a *symptom*. Before you interpret it, ask what it's downstream of:

- **Divergences** are downstream of the *symplectic leapfrog integrator failing to track the gradient* — specifically, the local error scales as `O(ε³ · ‖∇²log π‖)` per step, where `‖∇²log π‖` is the operator norm of the Hessian. So "step-size too large for the local curvature" is shorthand for "ε too large relative to the local Lipschitz constant of `∇log π`." That curvature is downstream of the **posterior**, which is downstream of the **prior × likelihood × parameterization × init**.
- **Low ESS** is downstream of *autocorrelation*, which is downstream of *poorly conditioned target geometry*, which is downstream of correlation structure in the model that the metric isn't capturing.
- **R̂ > 1.01** is downstream of *between-chain disagreement*, which can be modes, slow mixing, or just an underpowered run.
- **Max-tree-depth saturation** is downstream of *small step × long trajectory*, which is downstream of *anisotropic geometry the diagonal IMM can't capture*.

When you find yourself reaching for the immediate-cause knob (`target_acceptance += 0.05`, `max_num_doublings += 2`), back up at least one level. The "right knob" is usually two levels up the chain of causality.

---

## 1. Reading traceplots

A traceplot is a visual geometry probe. Look at each chain's trajectory over iterations, then ask *what the shape tells you about the posterior*.

| What you see | What the geometry is likely doing | Confirm by |
|---|---|---|
| Fuzzy caterpillar — chains overlap, no trend | Healthy mixing across the bulk | Rank plot uniform; ESS matches `n/τ_int` |
| Slow river meander — smooth drift, long correlations | High autocorrelation; the metric isn't capturing a correlation in the target | 2D scatter of the slow pair; ESS << n |
| Chain trapped in one region, others elsewhere | Multimodality or bad initialization | Histogram per chain; rank plot bimodal |
| Funnel shape — spread varies with another parameter | Hierarchical funnel — variance of leaf params couples to a scale param | 2D scatter (scale, leaf) shows the funnel directly |
| Intermittent spikes to extreme values | Divergences fire when the chain visits a sharp region | Overlay `is_divergent` on the 2D scatter |
| Flat segments — no movement | Chain stuck; logdensity NaN, or init far from posterior | Check `logdensity_fn` at init; check first-step acceptance |

**Rank plots** (rank-normalized traces) are more sensitive than raw traceplots for heavy-tailed posteriors. A uniform rank histogram across chains is the convergence signal you want; a deviation from uniform tells you which chain is the outlier.

A traceplot is a *first-pass* tool. It will show you what to investigate, but it will rarely identify the root cause on its own. Always pair it with at least one 2D scatter of the parameters the traceplot suggests are interacting.

---

## 2. Reading the diagnostic signals

Each diagnostic has a sharp definition, a default threshold, and a list of typical underlying causes. The defaults are useful starting points. The *meaning* of a tripped threshold depends on the context — the three lenses above are what you bring to read it.

### R̂ (modern: rank-normalized split-R̂, Vehtari et al. 2021)

**Implementation note (read this).** Some libraries' default R̂ helper computes the **classic Gelman-Rubin (1992)** statistic — a plain between-chain / within-chain variance ratio, *no* rank normalization, *no* folding. The widely-cited **R̂ < 1.01** threshold was calibrated for the *modern* Vehtari 2021 statistic, which is more sensitive: rank-normalization makes it robust to heavy tails, and folding (`z = |x − median(x)|`) catches scale discrepancies between chains that classic G-R misses (two chains with the same mean but different variance pass classic G-R; folded R̂ correctly fails them). For the modern statistic, use `arviz.rhat(samples)`. The classic statistic remains useful (cheaper, intuitive) but should not be used with the 1.01 threshold unless you understand the calibration mismatch. The loose threshold for classic G-R is more like 1.05; for the modern statistic, 1.01 is right. **Check which one your library's helper actually computes before applying a threshold to it.**

**What it measures.** Between-chain disagreement relative to within-chain variance. Modern variants add rank normalization (robustness to heavy tails) and folding (scale-discrepancy detection).

**When it trips, ask:**
- Are the chains in *different modes*? (Histogram per chain.)
- Is the model under-warmupped for this parameter? (Look at the per-parameter trace.)
- Is this parameter weakly identified — so the prior is doing the work, and "between-chain disagreement" is the prior, not the posterior? (Compare per-chain means to the prior mean. If they're all near the prior mean with little spread, R̂=1.05 is partly the prior leaking through.)
- Is the model under-sampled? (Watch R̂ as you scale n. If it shrinks with n, the chains are mixing; they just need more draws.)

**When it's fine to accept R̂ > 1.01.** A weakly-informed hyperparameter at the top of a hierarchy, or a parameter that is *intentionally* multi-modal under your prior, can sit above 1.01 without breaking the inference for parameters you actually report. Be explicit about which parameters need which thresholds.

### Effective sample size — bulk and tail

A bulk/tail ESS helper returns `(bulk_ess, tail_ess)`. Bulk governs mean/median estimates; tail governs quantile estimates. The default is **bulk-ESS / num_chains > 100** (so ≥ 400 for 4 chains).

**MCSE** = `posterior_std / sqrt(bulk_ess)` is the right calibration: it tells you which decimal places of a reported posterior mean are real and which are Monte-Carlo noise.

**When it trips, ask:**
- Bulk-ESS low but R̂=1: chain is mixing slowly across the bulk — autocorrelation is high. Look for *strong correlations the diagonal IMM can't capture* or *long memory in the latent state*. The fix is usually a reparameterization or a different metric, not more samples.
- Tail-ESS low with healthy bulk-ESS: the chain visits the bulk but rarely visits the tails. Heavy tails or hard-to-reach corners. More samples *might* help; a sampler change (slice, MCLMC, full-rank metric) often helps more.
- Both ESS low and divergences present: the chain is failing to explore *and* the local geometry is sharp. Treat the geometry first (parameterization), then re-evaluate ESS.

**Context matters.** A `min-bulk-ESS = 200` is fine if you care about the mean and don't quote tail quantiles. A `tail-ESS = 200` is unacceptable if you report a 95% credible interval.

### Divergences

`info.is_divergent` is a boolean per draw. A divergence means: during a single Hamiltonian trajectory, the integrator's energy error exceeded the divergence threshold (default 1000) — the leapfrog blew up.

**What it measures.** The local step size was too large for the local curvature *somewhere along the trajectory*.

**When it trips, the most important thing to do first is *find the divergences in parameter space*** (Lens 1). Once you know where they live, the interpretation is contextual:

- **Clustered in a tail of one or two parameters** → prior is too wide, the chain visited a low-probability region with bad geometry. Fix: tighten the prior, change the parameterization at that boundary, or accept the divergence rate as a structural signal of the prior choice.
- **Clustered at the funnel neck of a hierarchical scale** → centered-vs-non-centered parameterization story. Fix: switch parameterization.
- **Scattered uniformly across the posterior** → the sampler / metric is bottlenecked globally. Fix: full-rank or low-rank metric, MCLMC, or split into more chains.
- **Clustered near the init** → bad initialization, not a model problem. Fix: MAP-warmup, longer warmup, or different init.

**When it's fine to accept a small divergence rate.** 1 / 40 000 in a model with a funnel-like prior, with all other diagnostics healthy and the divergence not in a region you care about reporting, is a *signal* of the model's geometry, not a bug. A reasonable inspection threshold is ~0.1%; below that it's a flag for inspection, not a hard stop.

```python
# Cluster check — first thing to do on a divergence-rate failure
import jax.numpy as jnp
div_mask = is_divergent
non_div_mask = ~div_mask
delta = jnp.array([
    (positions[div_mask, i].mean() - positions[non_div_mask, i].mean())
    / positions[non_div_mask, i].std()
    for i in range(positions.shape[1])
])
top_offenders = jnp.argsort(jnp.abs(delta))[::-1][:5]
# Print: (param_index, Δ/σ, fraction-of-divs-in-|z|>2-tail)
```

### Energy / E-BFMI

E-BFMI = `Var(ΔH) / Var(H)` where `H` is the leapfrog Hamiltonian. The Betancourt (2016) threshold is **E-BFMI > 0.3**.

**What it measures.** Whether momentum resampling between trajectories is exploring the energy landscape efficiently. Low E-BFMI means consecutive trajectories have similar energy, so the chain explores the energy axis slowly.

**When it trips, ask:**
- Is the posterior *heavy-tailed* in a parameter that drives the kinetic energy? Heavy tails (Cauchy-like) cause the chain to spend many trajectories in the bulk and few in the tail.
- Is the kinetic energy structurally coupled to one parameter (a scale, in a hierarchy)? Look at `energy` vs that parameter as a 2D scatter.

E-BFMI and divergences typically flag **different failure modes** — divergences at sharp local curvature (the integrator overshoots), low E-BFMI at heavy energy tails (momentum resampling can't keep up). The canonical fixes differ, so don't reach for the same knob by default. But they are **not mutually exclusive**: a hierarchical funnel produces both simultaneously, because the funnel's *neck* is locally sharp (divergences fire at the constriction) while the *log-scale* parameter has a heavy tail (low E-BFMI from energy variability). Same topological feature, two integrator-failure signatures.

### Acceptance rate

- Mean acceptance near 0 → step is too large; the integrator is rejecting everything. (Or `logdensity_fn` returns NaN.)
- Mean acceptance near 1 → step is too small; the chain crawls. Or geometry is trivial and the run is over-tuned.
- Bimodal acceptance (mean=0.85 but bottom 5% are near 0) → the chain occasionally enters a sharp region. This is upstream of a divergence problem.

### Tree-depth saturation (NUTS)

`info.num_trajectory_expansions` near `max_num_doublings` (default 10 → 1023 leapfrog steps per draw) means NUTS isn't finding the no-U-turn criterion before the budget runs out.

**When it trips, ask:**
- Is the *step* so small that the trajectory needs many doublings to reach U-turn? Often a downstream signal of the metric being mis-adapted (the IMM is too tight in one direction).
- Is there *correlation structure* the diagonal IMM can't capture? A 2D scatter of any two strongly correlated parameters will look like a thin ellipse — the chain needs many leapfrog steps to traverse the long axis.
- Is `max_num_doublings` simply too low for the natural trajectory length of this model? GPs, ODE models, and high-d state spaces commonly want `max_num_doublings = 12–15`.

**Why trajectory length and condition number couple.** The number of leapfrog steps NUTS needs before it can detect a no-U-turn condition scales roughly with `√κ`, where `κ` is the condition number of the local metric (ratio of largest to smallest eigenvalue of the inverse mass matrix). When the IMM is well-adapted to an anisotropic posterior, `κ ≈ 1` and U-turn fires quickly; when it isn't, `κ` blows up and the chain crawls along the elongated axis until it's used its tree-depth budget. This is why "raise `n_warmup`" and "raise `max_num_doublings`" come paired for high-d latents: better IMM adaptation lowers `κ`, *and* the tree-depth ceiling has to give the chain room to traverse whatever `κ` it actually adapts to.

> **Case study (latent-GP regression)**: a 203-D latent-GP regression saturated the depth-10 ceiling on every draw; the IMM never adapted and `step_size` collapsed to 1e-6. The visible signal is at the bottom (depth saturation); the cause is two levels up (`n_warmup=500` is insufficient for d=203). Rule of thumb from this and an analogous horseshoe (d=204) case: for d > 100, plan `n_warmup ≥ 2000` and consider `max_num_doublings = 12–15`.

---

## 3. Geometry exploration

Before changing any sampler parameter, *understand the posterior geometry*. The diagnostics above tell you *what* is wrong; the plots below tell you *where*.

### Cluster divergences in parameter space (first thing, every time)

```python
import jax.numpy as jnp
import matplotlib.pyplot as plt

div = info.is_divergent
# Per-parameter signed standardized difference (div minus non-div mean)
delta = (positions[div].mean(0) - positions[~div].mean(0)) / positions[~div].std(0)
top_k = jnp.argsort(jnp.abs(delta))[::-1][:5]

# Quick visual on top offenders
for i in top_k:
    fig, ax = plt.subplots()
    ax.hist(positions[~div, i], bins=50, alpha=0.5, label="non-div", density=True)
    ax.hist(positions[ div, i], bins=50, alpha=0.5, label="div",     density=True)
    ax.set_title(f"param {i}: Δ/σ = {float(delta[i]):.2f}")
    ax.legend()
```

A divergence cluster with `|Δ/σ| > 2` and tail-fraction > 50% is a strong signal that the problem is local to that parameter's tail. Read against the prior on that parameter — often the prior is wider than the data warrants.

### 2D marginal contour + divergence overlay

For the top 2 offenders from the cluster check, plot the joint:

```python
plt.scatter(positions[~div, i], positions[~div, j], s=2, alpha=0.3, label="samples")
plt.scatter(positions[ div, i], positions[ div, j], s=8, color="red", label="div")
```

Elongated, curved, or funnel-shaped contours signal correlations or scale-coupling that the diagonal IMM cannot capture. Divergences at the narrow end of a funnel = funnel-neck pathology. Divergences along a ridge = unidentified or curved correlation.

### Conditional slice plots

Fix one parameter at its posterior mean and plot `logdensity_fn` as a function of another. This reveals whether curvature is roughly uniform (good for default step sizes) or varies dramatically across the space.

### Marginal vs joint — always check both

A marginal can look healthy while the joint hides a funnel. Always pair traceplots and marginal histograms with at least one 2D scatter of the parameters the diagnostic singled out.

---

## 4. Re-parameterization first — knob-tuning second, last

In nearly every diagnostic above, the canonical first fix is *change the model*, not change the sampler. Tuning `target_acceptance` from 0.8 to 0.99 on a funnel masks the symptom for that step-size regime; the underlying geometry is unchanged, and the next stress test (more data, a tighter prior, a new seed) will surface it again.

The cardinal sin in MCMC tuning is treating geometric pathology as a step-size problem.

| Symptom (what you see) | Right fix (one level up) | Wrong fix (don't do this first) |
|---|---|---|
| Funnel → divergences | Non-centered parameterization | `target_acceptance = 0.99` |
| Boundary geometry at `sigma → 0` | `log_sigma` reparameterization | `step_size = 0.001` |
| Heavy tail in α | Stronger / informative prior on α; reparameterize α | `n_samples = 100000` |
| Multimodality | Overdispersed init; SMC tempered | More chains, same init |
| Strong correlations | Whitening / Cholesky factor; or low-rank IMM | `max_num_doublings += 2` |
| Divergences clustered at unit-root in an AR(1) | Tighten prior away from the boundary | Change samplers |
| Depth saturation on a high-d latent | Longer warmup; raise `max_num_doublings` modestly | Lower `target_acceptance` |

When you find yourself reaching for the wrong-fix column, stop and re-read the three lenses at the top of this file. For the procedural "now what do I do?" question, return to `STATISTICIAN_BAYESIAN_WORKFLOW.md` Step 4.

---

## 5. Case studies — where these lenses get put into practice

Worked examples are most useful as one-finding-per-file write-ups, indexed by **diagnostic symptom** — *"my chain is doing X; has anyone seen this before?"* — and also by model. Anchor cases to revisit when the corresponding lens is what's tripping:

- **Lens 1 (location not count)** → AR(1) unit-root divergence cluster. The 3-minute cluster check that two prior architectural diagnoses had skipped.
- **A correct diagnosis is not a fix** → the failed prior-tightening swap on that same φ. The cluster diagnosis pointed at tightening the prior; production re-run *tripled* the divergence count.
- **Lens 3 (signals are downstream of something else)** → diagonal-IMM on an AR(1) boundary. The condition-number signal was real but downstream; reading the upstream signal changed the response.
- **Default warmup budgets are calibrated for the average-d model** → under-warmupped IMM at d=203. Depth-10 saturation downstream of un-adapted IMM downstream of insufficient warmup budget.
- **Wall-time failures are a different category from statistical failures** → a 203-D latent-GP with healthy R̂/ESS/divergences but ~50 h wall on CPU. A clean chain can still be the wrong production choice.

When you write up a new finding, keep it to one finding per file, dated, with status, and cross-linked from the per-model and symptom indexes.

---

## 6. Algorithm correctness review (PR review mode)

When reviewing a new or modified MCMC/VI algorithm — not diagnosing a user model, but checking whether the implementation matches the paper — follow this protocol.

### 6a. Paper-to-code mapping

1. Locate the original paper. Find the algorithm statement (usually a numbered algorithm block).
2. Map each variable in the pseudocode to the corresponding variable in the implementation.
3. Verify in order: loop structure, acceptance criterion, momentum refreshment schedule, step-size update rule, normalization constants.
4. Flag any discrepancy with a precise diff: *"Paper Eq. 4 uses ε/2 for the half-step; code at `<file>:<line>` uses `step_size` (full step) — this is a bug."*

### 6b. Cross-reference checklist

Check against at least two independent reference implementations for any HMC/NUTS variant:

| Reference | Where to look |
|---|---|
| NumPyro | `numpyro/infer/mcmc.py`, `numpyro/infer/hmc.py` |
| Stan | Stan Reference Manual, *HMC* chapter + CmdStan source |
| TFP | `tensorflow_probability/python/mcmc/` |
| Original author's repo | Usually linked in the paper abstract |

A discrepancy is not always wrong — document the design decision. A match does not always mean correct — implementations can have the same bug.

### 6c. Structured review output format

```
## Algorithm Correctness
- [PASS/FAIL/WARN] <check name>: <finding>

## Code-Math Alignment
- Paper: <equation ref> — Code: <file:line> — Status: [MATCH/MISMATCH]

## Test Coverage
- [PASS/FAIL/WARN] <what is or isn't tested>

## Statistical Properties Verified
- <property verified, method used, conclusion>

## Recommendations
- <prioritized list: BLOCKER / MINOR>
```

---

## 7. Test robustness against stochasticity (when writing sampler tests)

MCMC tests fail non-deterministically if poorly designed. Enforce:

- Use a date-seeded or fixture-provided RNG (a shared test-suite RNG helper / `next_key()`-style fixture) so failures are reproducible.
- Test **statistical properties**, not exact values: `ESS > threshold`, `R-hat < 1.01`, `mean within 2 standard errors of truth`.
- Numerical tolerance: use `jnp.allclose(actual, expected, atol=1e-3)` not exact equality.
- For adaptation tests: verify the *final adapted parameters* are in a sane range, not that they equal a hard-coded value.
- Use `chex.assert_max_traces(n=2)` to prevent recompilation bugs from masking statistical errors.
- For tail-quantile tests: bump `n_samples` rather than tightening `atol`. Tight `atol` on a 200-sample chain is a coin flip.

```python
# Good stochastic test pattern
def test_nuts_samples_std_normal(self):
    key = self.next_key()
    algorithm = blackjax.nuts(
        logdensity_fn=std_normal_logdensity,
        step_size=0.1,
        inverse_mass_matrix=jnp.ones(2),
    )
    _, samples = blackjax.util.run_inference_algorithm(
        key, algorithm, num_steps=2000, initial_position=jnp.zeros(2),
    )
    # Test statistical properties, not exact values
    self.assertAllClose(jnp.mean(samples.position, axis=0), jnp.zeros(2), atol=0.1)
    self.assertAllClose(jnp.std(samples.position, axis=0), jnp.ones(2), atol=0.1)
```

---

## 8. References

- Vehtari et al., *Rank-normalization, folding, and localization* (Bayesian Analysis, 2021) — defines rank-normalized split-R-hat and the 1.01 threshold
- Betancourt, *Diagnosing Suboptimal Cotangent Disintegrations in HMC* (arXiv:1604.00695) — E-FMI theory and interpretation
- Betancourt, *Identifying the Optimal Integration Time in HMC* (arXiv:1601.00225) — `max_num_doublings`, trajectory length tuning
- Stan Reference Manual, *Diagnostics* chapter — acceptance rate targets, divergence definitions
- Gelman & Rubin, *Inference from Iterative Simulation* (Statistical Science, 1992) — original R-hat (superseded by Vehtari 2021 for practical use)
- Stan User's Guide, *§ 2.5 Stochastic volatility models* — the canonical NCP form + the `Beta(20, 1.5)`-shifted phi prior for daily financial volatility (the stoch-vol worked example above)
