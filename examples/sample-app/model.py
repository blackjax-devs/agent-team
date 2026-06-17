"""A minimal BlackJAX app: sample the posterior of a Normal mean.

Data are drawn from N(mu=3, sigma=1). We put a wide Normal prior on `mu` and
run NUTS to recover it. There is a MILD, intentional issue for the team to
debug: the NUTS `step_size` below is far too large for this geometry, so the
sampler's acceptance probability collapses and the chain barely moves / mixes
terribly. The fix is to run window adaptation (or hand-tune a smaller step
size) instead of hardcoding `step_size=5.0`.
"""

import jax
import jax.numpy as jnp
import blackjax

KEY = jax.random.PRNGKey(0)

# --- data ----------------------------------------------------------------
true_mu, sigma = 3.0, 1.0
data = true_mu + sigma * jax.random.normal(KEY, shape=(200,))


# --- model ---------------------------------------------------------------
def logdensity_fn(mu):
    """Unnormalized log posterior of the mean `mu` (wide Normal prior)."""
    log_prior = jax.scipy.stats.norm.logpdf(mu, loc=0.0, scale=10.0)
    log_lik = jnp.sum(jax.scipy.stats.norm.logpdf(data, loc=mu, scale=sigma))
    return log_prior + log_lik


# --- sampler -------------------------------------------------------------
# BUG (intentional, mild): step_size is hardcoded and far too large for this
# tightly-peaked posterior. Acceptance collapses and the chain doesn't mix.
# Recommended fix: use blackjax.window_adaptation to tune step_size + mass matrix.
nuts = blackjax.nuts(logdensity_fn, step_size=5.0, inverse_mass_matrix=jnp.ones(1))


def inference_loop(rng_key, kernel, initial_state, num_samples):
    @jax.jit
    def one_step(state, key):
        state, info = kernel(key, state)
        return state, (state, info)

    keys = jax.random.split(rng_key, num_samples)
    _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)
    return states, infos


if __name__ == "__main__":
    init = nuts.init(jnp.array([0.0]))
    states, infos = inference_loop(KEY, nuts.step, init, 1000)
    samples = states.position[200:]  # drop warmup
    print(f"posterior mean estimate: {jnp.mean(samples):.3f} (true {true_mu})")
    print(f"mean acceptance prob:    {jnp.mean(infos.acceptance_rate):.3f}")
