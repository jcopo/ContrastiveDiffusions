# pylint: disable=assignment-from-no-return
from dataclasses import dataclass
from typing import NamedTuple, Tuple

from diffuse.diffusion.sde import SDEState
import jax
import optax
from diffuse.base_forward_model import MeasurementState
from diffuse.denoisers.cond import CondDenoiser, CondDenoiserState
from jax import numpy as jnp
from jaxtyping import Array, PRNGKeyArray
from optax import GradientTransformation
import matplotlib.pyplot as plt

from design.forward_model import DesignForwardModel

def plot_lines(array: jnp.ndarray, fractions: list = None, title: str = "Trajectory Evolution") -> None:
    """
    Plot frames at different fractions of a trajectory array.

    Args:
        array: Array with shape (n_steps, height, width, channels) or (n_steps, height, width)
        fractions: List of fractions [0.0, 1.0] to plot
        title: Title for the plot
    """
    if fractions is None:
        fractions = [0.1 * i for i in range(10)]

    n = len(fractions)
    fig, axs = plt.subplots(1, n, figsize=(n * 3, 3))
    fig.suptitle(title)

    for idx, fraction in enumerate(fractions):
        frame_index = int(fraction * (array.shape[0] - 1))
        axs[idx].imshow(array[frame_index].squeeze(), cmap="gray")
        axs[idx].set_title(f"f={fraction:.1f}")
        axs[idx].axis("off")

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(2)
    plt.close(fig)

class BEDState(NamedTuple):
    denoiser_state: CondDenoiserState
    cntrst_denoiser_state: CondDenoiserState
    design: Array
    opt_state: optax.OptState


@dataclass
class ExperimentOptimizer:
    denoiser: CondDenoiser
    cntrst_denoiser: CondDenoiser
    mask: DesignForwardModel
    optimizer: GradientTransformation
    base_shape: Tuple[int, ...]

    def init(self, rng_key: PRNGKeyArray, n_samples: int, n_samples_cntrst: int) -> BEDState:
        """Initialize the experiment optimizer."""
        design = self.mask.init_design(rng_key)
        opt_state = self.optimizer.init(design)

        key_pos, key_t, key_c = jax.random.split(rng_key, 3)
        thetas, cntrst_thetas = _init_start_time(key_pos, n_samples, n_samples_cntrst, self.base_shape)
        denoiser_state = jax.vmap(self.denoiser.init, in_axes=(0, 0, None))(
            thetas, jax.random.split(key_t, n_samples), n_samples
        )
        cntrst_denoiser_state = jax.vmap(self.denoiser.init, in_axes=(0, 0, None))(
            cntrst_thetas, jax.random.split(key_c, n_samples_cntrst), n_samples_cntrst
        )

        return BEDState(
            denoiser_state=denoiser_state,
            cntrst_denoiser_state=cntrst_denoiser_state,
            design=design,
            opt_state=opt_state,
        )

    def step(
        self,
        state: BEDState,
        rng_key: PRNGKeyArray,
        measurement_state: MeasurementState,
        plot_callback,
        step_idx: int,
        measurement_idx: int,
    ) -> BEDState:
        # Setup and extract state
        design, opt_state = state.design, state.opt_state
        rng_key, key_t, key_c, key_noise = jax.random.split(rng_key, 4)

        # Step joint sampler: sample p(θ|D_k), one particle per vmap lane
        n_samples = state.denoiser_state.integrator_state.position.shape[0]
        keys_t = jax.random.split(key_t, n_samples)
        denoiser_state = jax.vmap(self.denoiser.step, in_axes=(0, 0, None))(keys_t, state.denoiser_state, measurement_state)
        t_current = self.denoiser.integrator.timer(denoiser_state.integrator_state.step)

        # θ estimates E[θ|x_t, D_k]: denoiser-polymorphic (amortized samplers bind D_k)
        thetas = jax.vmap(self.denoiser.denoise, in_axes=(0, 0, None))(
            denoiser_state.integrator_state.position, t_current, measurement_state
        )

        # Construct superposed measurement for pooled posterior
        #    Combines: old measurement D_k + contrastive measurements from current θ estimates
        # Create masks
        new_mask = self.mask.make(design)
        new_part_mask = new_mask * (1 - measurement_state.mask_history)
        superposed_mask = measurement_state.mask_history + new_part_mask

        # Contrastive measurements y_i = A_ξ θ_i + σ ε_i on the newly observed region.
        # Kept at clean (time-0) scale: FPS guidance rescales targets by α_t internally.
        noise = jax.random.normal(key_noise, thetas.shape)
        y_superposed = measurement_state.y + (thetas + self.mask.std * noise) * new_part_mask

        # Step pooled posterior sampler: sample p(θ|D_k, {y_i}, ξ). Each contrastive
        # particle is guided toward its round-robin assigned superposed measurement.
        n_cntrst = state.cntrst_denoiser_state.integrator_state.position.shape[0]
        assignments = jnp.arange(n_cntrst) % n_samples
        cntrst_measurement_state = MeasurementState(y=y_superposed[assignments], mask_history=superposed_mask)
        keys_c = jax.random.split(key_c, n_cntrst)
        contrastive_state = jax.vmap(self.cntrst_denoiser.step, in_axes=(0, 0, MeasurementState(0, None)))(
            keys_c, state.cntrst_denoiser_state, cntrst_measurement_state
        )

        # Contrastive θ estimates E[θ|x_t, D_k ∪ y_i]: per-particle conditioning
        cntrst_thetas = jax.vmap(self.cntrst_denoiser.denoise, in_axes=(0, 0, MeasurementState(0, None)))(
            contrastive_state.integrator_state.position,
            self.denoiser.integrator.timer(contrastive_state.integrator_state.step),
            cntrst_measurement_state,
        )

        # Update design via gradient ascent on EIG, with the observation variance
        # annealed by the Tweedie posterior uncertainty at the current time
        t_scalar = t_current[0]
        sigma_t = self.denoiser.model.noise_level(t_scalar)
        alpha_t = self.denoiser.model.signal_level(t_scalar)
        std_eff = jnp.sqrt(self.mask.std**2 + (sigma_t / alpha_t) ** 2)
        design, opt_state, ys, val_eig = calculate_and_apply_gradient(
            thetas, cntrst_thetas, design, self.mask, self.optimizer, opt_state, noise, std_eff
        )

        # Create updated state
        state_next = BEDState(
            denoiser_state=denoiser_state,
            cntrst_denoiser_state=contrastive_state,
            design=design,
            opt_state=opt_state,
        )

        if plot_callback is not None:
            jax.experimental.io_callback(
                plot_callback,
                None,
                state_next.denoiser_state.integrator_state.position,
                y_superposed,
                superposed_mask,
                step_idx,
                t_current[0],
                measurement_idx,
            )

        return state_next, val_eig

    def get_design(
        self,
        state: BEDState,
        rng_key: PRNGKeyArray,
        measurement_state: MeasurementState,
        n_loop_opt: int,
        plot_callback,
        measurement_idx: int,
    ):
        # Each trajectory takes n_steps iterations; restart happens at the start of
        # the iteration that follows a completed trajectory, so the scan ends exactly
        # on an end-of-trajectory state (t = eps) and never steps past it.
        n_steps_per_traj = self.denoiser.integrator.timer.n_steps
        total_steps = n_loop_opt * n_steps_per_traj

        def step_fn(state, tup):
            rng_key, idx = tup
            current_step = state.denoiser_state.integrator_state.step[0]
            should_restart = (current_step >= n_steps_per_traj) & (idx > 0)
            state = jax.lax.cond(
                should_restart,
                lambda s: restart_state(s, rng_key, self.denoiser),
                lambda s: s,
                state,
            )
            state, val_eig = self.step(state, rng_key, measurement_state, plot_callback, idx, measurement_idx)
            t_current = self.denoiser.integrator.timer(state.denoiser_state.integrator_state.step[0])

            return state, (state.design, val_eig, t_current)

        keys = jax.random.split(rng_key, total_steps)
        return jax.lax.scan(step_fn, state, (keys, jnp.arange(total_steps)))


def restart_state(state, rng_key, denoiser):
    """
    Restart the state of the denoiser and cntrst_denoiser to t0
    """
    n_thetas = state.denoiser_state.integrator_state.position.shape[0]
    n_cntrst_thetas = state.cntrst_denoiser_state.integrator_state.position.shape[0]
    base_shape = state.denoiser_state.integrator_state.position.shape[1:]
    rng_key, rng_key_t, rng_key_c = jax.random.split(rng_key, 3)
    thetas, cntrst_thetas = _init_start_time(rng_key, n_thetas, n_cntrst_thetas, base_shape)
    denoiser_state = jax.vmap(denoiser.init, in_axes=(0, 0, None))(
        thetas, jax.random.split(rng_key_t, n_thetas), n_thetas
    )
    cntrst_denoiser_state = jax.vmap(denoiser.init, in_axes=(0, 0, None))(
        cntrst_thetas, jax.random.split(rng_key_c, n_cntrst_thetas), n_cntrst_thetas
    )
    return BEDState(
        denoiser_state=denoiser_state,
        cntrst_denoiser_state=cntrst_denoiser_state,
        design=state.design,
        opt_state=state.opt_state,
    )


def _init_start_time(
    key_init: PRNGKeyArray,
    n_samples: int,
    n_samples_cntrst: int,
    ground_truth_shape: Tuple[int, ...],
) -> Tuple[Array, Array]:
    """
    Initialize thetas for just the start time of the conditional sampling
    """
    key_t, key_c = jax.random.split(key_init)
    thetas = jax.random.normal(key_t, (n_samples, *ground_truth_shape))
    cntrst_thetas = jax.random.normal(key_c, (n_samples_cntrst, *ground_truth_shape))
    return thetas, cntrst_thetas


def calculate_and_apply_gradient(
    thetas: Array,
    cntrst_thetas: Array,
    design: Array,
    mask: DesignForwardModel,
    optx_opt: GradientTransformation,
    opt_state: optax.OptState,
    noise: Array,
    std_eff: Array,
):
    grad_xi_score = jax.value_and_grad(information_gain, argnums=2, has_aux=True)
    (val_eig, ys), grad_xi = grad_xi_score(thetas, cntrst_thetas, design, mask, noise, std_eff)
    updates, new_opt_state = optx_opt.update(grad_xi, opt_state, design)
    new_design = optax.apply_updates(design, updates)
    # Keep the whole square in frame: a center outside [s/2, H - s/2] wastes
    # the fraction of the mask that falls off the image.
    half = mask.size / 2
    new_design = jnp.clip(new_design, half, mask.img_shape[0] - half)
    return new_design, new_opt_state, ys, val_eig


def information_gain(theta: Array, cntrst_theta: Array, design: Array, mask: DesignForwardModel, noise: Array, std_eff: Array):
    r"""
    EIG gradient surrogate, eq. (10)-(11) of the CoDiff paper.

    Surrogate whose ∇_ξ is the estimator
        (1/N) Σ_i [ g(ξ,y_i,θ_i,θ_i) − Σ_j w_ij g(ξ,y_i,θ_i,θ'_j) ]
    with y_i = A_ξ θ_i + σ ε_i reparametrized (ε_i fixed w.r.t. ξ) and
    w_ij the SNIS weights p(θ'_j|y_i,ξ)/q_{ξ,N}(θ'_j), uniform pooling ν_ℓ = 1/N.

    std_eff is the time-annealed observation std σ_eff(t)² = σ_y² + σ_t²/α_t²
    (ΠGDM-style): it propagates the Tweedie posterior uncertainty of the θ
    estimates into the likelihood, so early-t updates (where θ̂ is dominated by
    score error amplified by 1/α_t) contribute ≈ 0 gradient, and the exact
    estimator is recovered as t → 0.
    """
    # Create measurement state with design mask
    mask_design = mask.make(design)
    measurement_state = MeasurementState(y=jnp.zeros_like(theta), mask_history=mask_design)

    # Joint samples y_i = A_ξ θ_i + σ_eff ε_i
    y_pred_ref = mask.apply(theta, measurement_state)
    y_ref = y_pred_ref + std_eff * noise

    # log p(y_i|θ_i, ξ)
    logprob_ref = jax.vmap(lambda y, y_pred: -0.5 * jnp.sum((y - y_pred) ** 2) / (std_eff**2))(y_ref, y_pred_ref)

    # log p(y_i|θ'_j, ξ) for all pairs (i, j)
    y_pred_cntrst = jax.vmap(mask.apply, in_axes=(0, None))(cntrst_theta, measurement_state)
    logprob_pairs = jax.vmap(
        lambda y: jax.vmap(lambda y_pred: -0.5 * jnp.sum((y - y_pred) ** 2) / (std_eff**2))(y_pred_cntrst)
    )(y_ref)

    # SNIS weights: log w̃_ij = log p(y_i|θ'_j) − (1/N) Σ_ℓ log p(y_ℓ|θ'_j), normalized over j
    log_weights = logprob_pairs - jnp.mean(logprob_pairs, axis=0, keepdims=True)
    log_weights = log_weights - jax.scipy.special.logsumexp(log_weights, axis=1, keepdims=True)
    weights = jax.lax.stop_gradient(jnp.exp(log_weights))

    # Σ_j w_ij log p(y_i|θ'_j, ξ)
    weighted_logprobs = jnp.sum(weights * logprob_pairs, axis=1)

    return (logprob_ref - weighted_logprobs).mean(), y_ref


@dataclass
class ExperimentRandom:
    denoiser: CondDenoiser
    mask: DesignForwardModel
    base_shape: Tuple[int, ...]

    def init(self, rng_key: PRNGKeyArray, n_samples: int, n_samples_cntrst: int):
        design = self.mask.init_design(rng_key)
        return BEDState(
            denoiser_state=None,
            cntrst_denoiser_state=None,
            design=design,
            opt_state=None,
        )

    def get_design(
        self,
        state: BEDState,
        rng_key: PRNGKeyArray,
        measurement_state: MeasurementState,
        **kwargs,
    ):
        n_steps = self.denoiser.integrator.timer.n_steps
        n_particles = jax.device_count() * 5
        design = self.mask.init_design(rng_key)
        cond_denoiser_state, _ = self.denoiser.generate(rng_key, measurement_state, n_steps, n_particles)

        return BEDState(
            denoiser_state=cond_denoiser_state,
            cntrst_denoiser_state=cond_denoiser_state,
            design=design,
            opt_state=None,
        ), _
