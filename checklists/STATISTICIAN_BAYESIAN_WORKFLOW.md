# Statistician — Modern Bayesian Workflow Checklist

**This file is the procedural workflow.** For diagnostic-signal interpretation (what does *this* signal mean), see `STATISTICIAN_DIAGNOSTICS_RECIPE.md`. For universal process rules (anti-narration, scratch files, commit format), see `AGENT_CHECKLIST.md`.

The workflow is **iterative investigation**, not a script. The order below names the kinds of questions to ask and the tools to reach for; the emphasis depends on what each step reveals. Do not cargo-cult — every step should be motivated by what you saw at the previous step.

> The code examples below use the [BlackJAX](https://github.com/blackjax-devs/blackjax) API (`blackjax.window_adaptation`, `blackjax.nuts`, …) as concrete illustrations. The *methodology* is library-agnostic; translate the calls to whatever sampler you use.

---

## The Folk Theorem (Gelman)

> When you have computational problems, often there's a problem with your model.

Divergences, poor mixing, and slow chains are *informative* — they reveal something about the posterior geometry. **Resist the reflex to tune knobs before reading what the diagnostics are telling you.** Tuning `adapt_delta` to 0.99 on a funnel model wastes compute and masks the real problem.

---

## Step 1 — Prior predictive checks (BEFORE you touch MCMC)

Simulate from the prior alone. Do the synthetic datasets look scientifically plausible? If the prior generates wildly implausible observations, no warmup budget will save you. Fix the model, not the sampler.

### Check the **pushforward**, not just the marginals

The most common failure mode is verifying priors *on the parameter scale* and missing that they push forward through a non-linear observation model into something extreme on the *data scale*. A standard normal on a logit-probability looks fine marginally and pushes forward to a bimodal Beta-like distribution piling up at 0 and 1. A `Normal(0, 5)` on a log-rate looks reasonable and pushes forward to median rates of `e^5 ≈ 148` events with massive variance.

**Always plot the simulated *data* against the domain of the likelihood**, not just the priors against the parameter scale:

```python
# Joint pushforward: priors → parameters → simulated observations
def simulate_one(key):
    k1, k2 = jax.random.split(key)
    params = sample_prior(k1)
    y_sim = likelihood_sample(k2, params)   # forward through the observation model
    return params, y_sim

keys = jax.random.split(key, 200)
params_draws, y_sim = jax.vmap(simulate_one)(keys)

# Check both scales:
# 1. Marginal priors (parameter scale) — what most students check
# 2. Pushforward to data scale — what reveals the actual bugs
for i in range(min(20, len(y_sim))):
    plt.hist(y_sim[i], bins=30, alpha=0.1, color="blue")
plt.title("Prior-predictive data; should look like plausible measurements")
```

If pushforward data looks implausible (extreme outliers, all-zeros, all-extremes), the prior is fighting the likelihood — revise the prior, the link function, or both. Loop.

If priors *and* their pushforward look reasonable but the model still misbehaves later, the bug is elsewhere (likelihood, indexing, Jacobian). Move on.

> **Out of scope but worth knowing**: *Simulation-based calibration* (SBC; Modrák et al. 2023) is the principled Step-0 check that would catch indexing errors in hierarchical models, missing Jacobian adjustments, and mis-specified likelihoods that no traceplot can detect. It works by simulating fake data from the prior, fitting the model to each simulated dataset, and checking that the rank statistics of the true parameters under the posterior are uniformly distributed. Run it once when validating a new model implementation against a reference. Not part of routine diagnostic work — SBC is compute-expensive (one fresh MCMC fit per simulated dataset, typically 100+ datasets), so reserve it for when the per-task budget allows.

---

## Step 2 — Fail fast: probe runs as diagnostics

Run **1 chain × (200 warmup + 200 sampling)** before committing to a long run. 200 samples will tell you almost everything important about whether the model is broken:

- Divergences at 200 will still be there at 10,000. Stop and fix the model.
- Chains stuck far from the posterior at 200 will not improve with more samples alone.
- If 200 samples look clean, scale up.

**Probe pattern (BlackJAX)** — `window_adaptation` handles step-size + mass-matrix tuning automatically:

```python
import blackjax, jax, jax.numpy as jnp

# Step 1: warmup (adaptation)
warmup = blackjax.window_adaptation(blackjax.nuts, logdensity_fn)
key, subkey = jax.random.split(key)
(state, adapted_params), _ = warmup.run(subkey, initial_position, num_steps=200)

# Step 2: sampling with adapted parameters
nuts = blackjax.nuts(logdensity_fn, **adapted_params)
key, subkey = jax.random.split(key)
_, (samples, info) = blackjax.util.run_inference_algorithm(
    subkey,
    nuts.init(state.position),
    nuts,
    num_steps=200,
    transform=lambda state, info: (state.position, info),
    progress_bar=False,
)
# samples has shape (200, D); note: NOT samples.position here because we extracted it in transform

# Diagnostics — [None] adds the chain dimension required by these functions
ess  = blackjax.diagnostics.effective_sample_size(samples[None])    # (1, 200, D) -> (D,)
rhat = blackjax.diagnostics.potential_scale_reduction(samples[None]) # needs >=2 chains for real use
print(f"min ESS: {jnp.min(ess):.0f}  max R-hat: {jnp.max(rhat):.3f}")
print(f"divergences: {jnp.sum(info.is_divergent)}  mean accept: {jnp.mean(info.acceptance_rate):.3f}")
```

Note: `samples[None]` / `samples.position[None]` adds a leading chain dimension required by `effective_sample_size` and `potential_scale_reduction`. For K parallel chains use `jnp.stack([chain1, ..., chainK])` producing shape `(K, T, D)`. R-hat is only meaningful with K >= 4 chains from overdispersed starts (see Step 6).

If divergences > 0 OR R-hat > 1.05 OR min ESS << 100 → go to Step 3 (read what the geometry is telling you), NOT Step 4 (tuning).

---

## Step 3 — Read the diagnostics, not the symptoms

Open `STATISTICIAN_DIAGNOSTICS_RECIPE.md` for the traceplot table + diagnostic hierarchy + geometry-exploration recipes. Identify:

- **Where** divergences cluster (overlay on 2D marginal — they live at the geometry bottleneck).
- **What** the slow-mixing direction looks like (rank plot per parameter).
- **Whether** the geometry is sharp (divergences) or flat (low E-FMI). These are *opposite* signals — fixes differ.

Output of Step 3 should be a one-sentence diagnosis: *"Funnel in (sigma, theta) at sigma → 0"* / *"Heavy left tail in alpha"* / *"Bimodality in beta_4 vs beta_5"*. Without that sentence, do not proceed to Step 4.

---

## Step 4 — Reparameterize BEFORE tuning

The single highest-leverage fix for hierarchical models is **non-centered parameterization**:

```python
# Centered (problematic when data is weak):
# theta[i] ~ Normal(mu, sigma)
# → geometry has a funnel in (mu, sigma) space at small sigma

# Non-centered (separates geometry):
# delta[i] ~ Normal(0, 1)
# theta[i] = mu + sigma * delta[i]
```

### Why this works (the geometric intuition)

In the **centered** form, the group means `theta[i]` are sampled *conditionally* on the population scale `sigma`. As `sigma → 0`, the conditional variance of `theta[i]` vanishes, pinching the joint `(sigma, theta)` posterior into a narrow neck — the classic Neal's funnel. An HMC sampler whose step size is tuned to the wide *top* of the funnel will overshoot the walls of the narrow neck, producing divergences clustered at small `sigma`. Tuning the step size down to handle the neck makes the chain crawl across the top — there is no single step size that works for both regions.

In the **non-centered** form, `delta[i] ∼ Normal(0, 1)` is sampled in a *flat, isotropic* space that's algebraically independent of `sigma`; the deterministic shift `theta[i] = mu + sigma · delta[i]` is applied after the fact. The sampler never sees the funnel geometry — it sees an uncorrelated standard-normal cloud, which is trivial. The funnel is still there in the posterior over `(mu, sigma, theta)`; it's just been moved out of the sampler's working space and into a deterministic post-hoc transformation.

**When NCP is the right call vs. wrong call**: NCP helps when the data is *weak* relative to the prior (so the funnel shape dominates the posterior). When the data is *strong*, the likelihood overpowers the funnel and the centered form is fine — and NCP can even hurt because the relationship between `theta` and `delta` is no longer 1:1 informative. The probe in Step 2 will tell you which regime you're in: if divergences cluster at small `sigma`, NCP. If divergences cluster somewhere else, look at what *that* somewhere-else is telling you.

Other common reparameterizations:
- **Log-transform positive parameters** (`sigma → log_sigma`) — removes boundary geometry. Same intuition: the boundary at `sigma = 0` is a hard wall the sampler can't cross; in log-space the boundary disappears.
- **Logit-transform bounded parameters** (`phi ∈ (-1, 1) → atanh(phi)`) — removes the analogous two-sided boundary geometry, e.g. for AR(1) persistence near the unit root.
- **Stick-breaking** for simplex parameters.
- **Cholesky factor** for covariance matrices — preserves PSD without quadratic cost.
- **Whitening** — premultiply by a known mass matrix when you have a good guess of the posterior covariance.

After reparameterization, re-run Step 2 (probe). Most of the time, divergences vanish and ESS climbs without changing a single tuning knob.

---

## Step 5 — Starting position

Starting position affects warmup convergence speed, especially in high dimensions. In order of preference:

1. **Pathfinder** — runs VI to find a good approximation before MCMC even starts. Also provides a warm-start mass matrix for adaptation. Use for any non-trivial model.
2. **MAP estimate** — optimize first, disperse chains around the mode. Useful when Pathfinder is too expensive.
3. **Prior samples** — acceptable when likelihood is strong relative to prior width.
4. **Zero / constant** — only for standardized models where the posterior is near zero.

**Pathfinder warm-start pattern (BlackJAX):**

```python
pathfinder = blackjax.vi.pathfinder.as_top_level_api(logdensity_fn)
key, subkey = jax.random.split(key)
pf_state, pf_info = pathfinder.run(subkey, initial_position, num_steps=300)
# pf_state.position  — use as starting point for MCMC
# pf_info.inverse_mass_matrix  — pass to window_adaptation as warm-start mass matrix

warmup = blackjax.window_adaptation(
    blackjax.nuts, logdensity_fn,
    initial_step_size=0.5,
    # If the sampler exposes mass_matrix_init, pass pf_info.inverse_mass_matrix here.
    # Otherwise use as the initial inverse_mass_matrix in the nuts kernel directly.
)
(state, adapted_params), _ = warmup.run(subkey2, pf_state.position, num_steps=500)
```

If the Pathfinder API changes, consult the sampler's `pathfinder` module for the current return type.

### Validate Pathfinder's approximation with Pareto-k

Pathfinder fits a normal approximation to the posterior; that approximation can be inadequate when the true posterior has heavy tails, funnels, or multiple modes. A bad approximation gives you a starting *position* that's still better than random, but the *warm-start inverse mass matrix* will be overly confident (too narrow) and will mislead `window_adaptation`. The diagnostic is the **Pareto-k** shape parameter of the importance-weight tail distribution (Vehtari, Gelman & Gabry 2017):

```python
# After pathfinder.run() returns pf_state:
# pathfinder.sample returns (samples, log_q) — the approximation log-density
# is computed alongside the draws, no separate logpdf call needed.
pf_samples, log_q = blackjax.vi.pathfinder.sample(key, pf_state, num_samples=1000)
log_p = jax.vmap(logdensity_fn)(pf_samples)        # target log-density at the draws
log_ratios = log_p - log_q                         # importance log-weights

_, pareto_k = blackjax.diagnostics.psis_weights(log_ratios)
print(f"Pathfinder pareto-k = {float(pareto_k):.2f}")
```

Interpretation (cited from the `psis_weights` docstring):

| `pareto_k` | Pathfinder approximation quality | What to do |
|---|---|---|
| **< 0.5** | Reliable. Tails of the proposal cover the target. | Trust the Pathfinder warm-start IMM. Proceed. |
| **0.5 – 0.7** | Moderate. Tails marginally under-covered. | Use Pathfinder for the starting position, but let `window_adaptation` re-adapt the IMM from scratch rather than using `pf_info.inverse_mass_matrix` as warm-start. |
| **> 0.7** | Unreliable. The normal approximation is losing mass in the tails (likely a funnel, multimodality, or heavy tails). | Use Pathfinder for the starting position *only*; throw out the IMM. Increase the warmup budget; the Pathfinder IMM would mislead adaptation. |
| **inf** (tail too small) | The approximation is so bad PSIS couldn't fit a tail. | Treat as `> 0.7`; ignore the Pathfinder IMM entirely. |

Reading the diagnostic before committing to the warm-start IMM is the difference between Pathfinder helping and Pathfinder secretly hurting downstream adaptation.

---

## Step 6 — Chains vs samples (resource allocation)

This is a budget decision, not a fixed formula. Think about what question you're answering:

- **R-hat diagnosis requires ≥ 4 chains** from overdispersed starts. There is no substitute.
- **Tail quantile estimation** benefits more from additional chains than from additional samples per chain.
- **Mean / median estimation** benefits more from additional samples once chains agree (R-hat < 1.01).
- **Parallel hardware**: chains parallelize perfectly via `jax.vmap`; use available devices before adding samples.
- **Expensive logdensity** (GP, neural net): minimize total gradient evaluations; invest budget in warmup quality over raw sample count.

Rule of thumb for most problems: **4 chains × 1000 post-warmup samples** is a sensible starting point. Scale up only where diagnostics suggest it's needed — don't run 10,000 samples by default.

---

## Step 7 — Compare algorithms (only after a clean baseline)

Once you have a clean (divergence-free, R-hat < 1.01) baseline, benchmarking different algorithms is about **efficiency**: min ESS per second across all dimensions. The metric that matters is the *worst-mixing dimension* (min ESS), since that bottleneck drives effective run length.

```python
import time

t0 = time.perf_counter()
# Use the same warmup + sampling pattern from Step 2, substituting the algorithm under test.
warmup = blackjax.window_adaptation(algo_class, logdensity_fn)
(state, adapted_params), _ = warmup.run(key, initial_position, num_steps=500)
algo = algo_class(logdensity_fn, **adapted_params)
_, (samples, _) = blackjax.util.run_inference_algorithm(
    key2, algo.init(state.position), algo, num_steps=1000,
    transform=lambda s, i: (s.position, i), progress_bar=False,
)
t1 = time.perf_counter()

ess = blackjax.diagnostics.effective_sample_size(samples[None])
min_ess_per_s = jnp.min(ess) / (t1 - t0)
```

Algorithm starting points — calibrate to the problem, not a fixed recipe:

| Problem class | Default algorithm |
|---|---|
| High-dim smooth posteriors | NUTS + `window_adaptation` |
| Hierarchical with weak data | NUTS + MEADS, non-centered |
| Very high-dim (d > 1000) | MCLMC + MCLMC tuning |
| Multimodal or complex geometry | SMC tempered + NUTS kernel |
| Heavy-tailed (low E-FMI despite tuning) | Slice sampling or add regularizing priors |

---

## Step 8 — Posterior predictive checks (after clean sampling)

Once you have clean samples (divergence-free, R-hat < 1.01), check whether the fitted model actually generates data that looks like the observations. This is the key loop-closing step in the Bayesian workflow — it can reveal model misspecification that the prior predictive check (Step 1) or sampler diagnostics cannot.

```python
# For each posterior sample, simulate a new dataset and compare to observed data.
# Shape: posterior_params has shape (T, D), y_obs has shape (N,).
def simulate_one(key, params):
    # generate a dataset of size N under the model parameterized by params
    return likelihood_sample(key, params)   # returns shape (N,)

keys = jax.random.split(key, num_samples)
y_rep = jax.vmap(simulate_one)(keys, posterior_samples)   # shape (T, N)

# Visual checks: overlay histogram of y_rep vs y_obs; plot test statistics
import matplotlib.pyplot as plt
plt.hist(y_obs, density=True, alpha=0.5, label="observed")
for i in range(0, len(y_rep), len(y_rep)//20):
    plt.hist(y_rep[i], density=True, alpha=0.05, color="blue")
plt.legend()
```

Targeted PPC test statistics (go beyond histograms):
- **Mean, SD, min, max** — cover location, spread, and extremes.
- **Skewness, kurtosis** — catch distributional mismatch.
- **Proportion of zeros / boundary values** — important for count or constrained data.

If PPCs fail: the model is misspecified. Revise the likelihood or add a component, then restart from Step 1.

---

## Step 9 — Recommendation output

After investigation, produce a recommendation grounded in **what you found**, not just what numbers won a grid search. The reasoning matters as much as the configuration. **Use this exact template** for the recommendation block:

```
Problem geometry: <what the traceplots / contour plots revealed>
Key issue identified: <e.g., funnel in (sigma, theta); heavy tails in likelihood>
Fix applied: <e.g., non-centered reparameterization; Pathfinder init>

Recommended configuration:
  Algorithm + warmup: <e.g., NUTS + window_adaptation>
  num_warmup / num_samples / num_chains: <N / N / N>
  Starting position: <method and why>
  Any non-default parameters: <e.g., max_num_doublings=10>

Why this works: <1-2 sentences on the geometry reason, not just the ESS number>
Watch for: <failure mode that could reappear; what to check if it degrades>
Runner-up: <alternative and the condition under which to prefer it>
```

---

## Appendix — When reparameterization isn't possible

Sometimes a parameter truly is bounded and cannot be freely reparameterized (e.g., a probability that must stay in [0,1] and the model is already in logit-space). In that case:

1. Confirm the boundary is genuinely structural, not just a modeling choice.
2. If divergences persist after confirming no further reparameterization is possible, *then* increase `adapt_delta` — but only to 0.95. Beyond 0.95 the step size becomes so small that ESS collapses.
3. Consider a different algorithm: slice sampling is provably correct on constrained domains; NUTS is not always the best choice for hard boundaries.

---

## References

- Gelman et al., *Bayesian Workflow* (arXiv:2011.01808) — the umbrella reference for everything in this doc.
- Gabry, Simpson, Vehtari, Betancourt & Gelman, *Visualization in Bayesian Workflow* (JRSS-A 2019) — the canonical reference for prior-pushforward checks and PPC visualizations (Step 1, Step 8).
- Vehtari et al., *Rank-normalization, folding, and localization* (Bayesian Analysis, 2021) — modern R̂ (Step 3 diagnosis).
- Vehtari, Gelman & Gabry, *Practical Bayesian model evaluation using LOO-CV and WAIC* (Statistics and Computing, 2017) — PSIS + the Pareto-k diagnostic used to validate Pathfinder (Step 5).
- Zhang, Carpenter, Gelman & Vehtari, *Pathfinder: Parallel quasi-Newton variational inference* (JMLR 2022) — Pathfinder algorithm + warm-start mass matrix (Step 5).
- Betancourt, *A Conceptual Introduction to Hamiltonian Monte Carlo* + *Identifying the Optimal Integration Time* — HMC geometry primer (Step 3, Step 4).
- Hoffman & Gelman, *NUTS* (JMLR 2014).
- Modrák et al., *Simulation-Based Calibration Checking* (Bayesian Analysis, 2023) — the canonical Step-0 check for *novel model implementations* (not for re-using a model on new data). Validates that `logdensity_fn + sampler` recover known ground truth by simulating data from the prior and checking rank uniformity of true parameters under the posterior. *Out of scope for routine diagnostic work* — SBC is compute-expensive (100+ MCMC fits); run it once when shipping a new model implementation against a reference, not on every diagnostic task.
