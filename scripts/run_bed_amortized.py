"""Sequential CoDiff BED run; saves per-measurement records to npz."""
import argparse
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx, serialization
from huggingface_hub import hf_hub_download

from diffuse.diffusion.sde import Flow, SDEState
from diffuse.integrator.deterministic import DDIMIntegrator
from diffuse.predictor import Predictor
from diffuse.timer import VpTimer

from design.bayesian_design import ExperimentOptimizer
from design.square_mask import SquareMaskDesign
from models.pixel_dit import PixelDiT

HF_REPO = "jcopo/mnist"
HF_WEIGHTS = "model.msgpack"
# Pinned: an unpinned download silently resolves to a stale cache entry on
# offline compute nodes, which loads the wrong weights instead of failing.
HF_REVISION = "7eb5205898b7a86361ff3bfda633072a412680c4"

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--num_meas", type=int, default=6)
parser.add_argument("--n_samples", type=int, default=151)
parser.add_argument("--n_samples_cntrst", type=int, default=150)
parser.add_argument("--n_t", type=int, default=6)
parser.add_argument("--n_loop_opt", type=int, default=30)
parser.add_argument("--mask_size", type=int, default=7)
parser.add_argument("--base_lr", type=float, default=2.0)
parser.add_argument("--design_mode", type=str, default="optimized", choices=["optimized", "random"])
parser.add_argument("--n_eval_particles", type=int, default=64)
parser.add_argument("--n_t_eval", type=int, default=8)
parser.add_argument("--weights", type=str, default=None, help="local model.msgpack; defaults to the Hub copy")
parser.add_argument("--out", type=str, required=True)
args = parser.parse_args()

print("devices:", jax.devices(), flush=True)

score_net = PixelDiT(rngs=nnx.Rngs(0))
_, _state = nnx.split(score_net)
_weights = args.weights or hf_hub_download(HF_REPO, HF_WEIGHTS, revision=HF_REVISION)
with open(_weights, "rb") as f:
    _state.replace_by_pure_dict(serialization.from_bytes(_state.to_pure_dict(), f.read()))
nnx.update(score_net, _state)
score_net.eval()
network_fn = lambda x, t: score_net(x, t=t).output  # no cond = unconditional
print(f"PixelDiT loaded from {_weights}", flush=True)

# Ground truth from MNIST, in the score network's data space [-1, 1]
data = np.load("dataset/mnist.npz")
xs = 2.0 * jnp.array(data["X"]).reshape(-1, 28, 28, 1) - 1.0
key = jax.random.PRNGKey(args.seed)
key, sample_key = jax.random.split(key)
ground_truth = jax.random.choice(sample_key, xs)

shape = ground_truth.shape
flow = Flow(tf=1.0)
timer = VpTimer(eps=1e-4, tf=1.0, n_steps=args.n_t)
integrator = DDIMIntegrator(model=flow, timer=timer)
predictor = Predictor(model=flow, network=network_fn, prediction_type="velocity")
mask = SquareMaskDesign(args.mask_size, shape, std=0.1)
from dataclasses import dataclass

from diffuse.denoisers.cond import CondDenoiser


@dataclass
class AmortizedDenoiser(CondDenoiser):
    """Direct conditional sampling: the measurement state IS the conditioning.

    No guidance — each step integrates the conditional v-field with
    c = [y, mask_history]. The pooled/contrastive case conditions each particle
    on its assigned superposed measurement via vmap over the MeasurementState.
    """

    def step(self, rng_key, state, measurement_state):
        del rng_key
        cond = jnp.concatenate([measurement_state.y, measurement_state.mask_history], axis=-1)
        pred = Predictor(
            model=self.model,
            network=lambda x, t: score_net(x, t=t, cond=cond).output,
            prediction_type="velocity",
        )
        return state._replace(integrator_state=self.integrator(state.integrator_state, pred))

    def denoise(self, position, t, measurement_state):
        """Conditional Tweedie E[θ|x_t, c]: bind the measurement in the score."""
        cond = jnp.concatenate([measurement_state.y, measurement_state.mask_history], axis=-1)
        pred = Predictor(
            model=self.model,
            network=lambda x, tt: score_net(x, t=tt, cond=cond).output,
            prediction_type="velocity",
        )
        return self.model.tweedie(SDEState(position, t), pred.score).position


denoiser = AmortizedDenoiser(integrator=integrator, model=flow, predictor=predictor, forward_model=mask, x0_shape=shape)
cntrst_denoiser = denoiser
eval_timer = VpTimer(eps=1e-4, tf=1.0, n_steps=args.n_t_eval)
eval_denoiser = AmortizedDenoiser(integrator=DDIMIntegrator(model=flow, timer=eval_timer), model=flow, predictor=predictor, forward_model=mask, x0_shape=shape)

# One cosine over the actual horizon (n_loop_opt * n_t design updates): the old
# fixed-length SGDR cycles made the final lr depend on where the horizon fell
# inside a cycle, coupling base_lr and n_loop_opt.
schedule = optax.cosine_decay_schedule(args.base_lr, decay_steps=args.n_loop_opt * args.n_t, alpha=0.05)
optimizer = optax.chain(optax.adam(learning_rate=schedule, b1=0.9, b2=0.98), optax.scale(-1))

eo = ExperimentOptimizer(denoiser=denoiser, cntrst_denoiser=cntrst_denoiser, mask=mask, optimizer=optimizer, base_shape=shape)


@jax.jit
def optimize_one_measurement(key, measurement_state):
    key_init, key_opt = jax.random.split(key)
    exp_state = eo.init(key_init, args.n_samples, args.n_samples_cntrst)
    return eo.get_design(exp_state, key_opt, measurement_state, args.n_loop_opt, None, 0)


import dm_pix


@jax.jit
def eval_posterior(key, measurement_state):
    """Fresh posterior samples given D_k; weighted SSIM/PSNR."""
    state, _ = eval_denoiser.generate(key, measurement_state, args.n_t_eval, args.n_eval_particles)
    samples = state.integrator_state.position
    weights = jax.nn.softmax(state.log_weights)
    gt01 = (ground_truth + 1.0) / 2.0
    s01 = jnp.clip((samples + 1.0) / 2.0, 0.0, 1.0)
    ssim = jnp.sum(jax.vmap(dm_pix.ssim, in_axes=(None, 0))(gt01, s01) * weights)
    psnr = jnp.sum(jax.vmap(dm_pix.psnr, in_axes=(None, 0))(gt01, s01) * weights)
    return samples, ssim, psnr


measurement_state = mask.init(key)
records = {"ground_truth": np.asarray(ground_truth)}
designs_seq, ys_seq, masks_seq, samples_seq, design_trajs, eig_trajs, t_trajs = [], [], [], [], [], [], []
ssims, psnrs = [], []

for k in range(args.num_meas):
    key, key_opt, key_eval = jax.random.split(key, 3)
    t0 = time.time()
    if args.design_mode == "random":
        design = mask.init_design(key_opt)
    else:
        optimal, (design_traj, eig_traj, t_traj) = optimize_one_measurement(key_opt, measurement_state)
        jax.block_until_ready(optimal.design)
        design = optimal.design
        design_trajs.append(np.asarray(design_traj))
        eig_trajs.append(np.asarray(eig_traj))
        t_trajs.append(np.asarray(t_traj))
    measurement_state = mask.update_measurement(measurement_state, ground_truth, design)

    samples, ssim, psnr = eval_posterior(key_eval, measurement_state)
    jax.block_until_ready(samples)
    print(
        f"measurement {k + 1}/{args.num_meas}: design={np.asarray(design)} "
        f"ssim={float(ssim):.4f} psnr={float(psnr):.2f} ({time.time() - t0:.1f}s)",
        flush=True,
    )

    designs_seq.append(np.asarray(design))
    ys_seq.append(np.asarray(measurement_state.y))
    masks_seq.append(np.asarray(measurement_state.mask_history))
    samples_seq.append(np.asarray(samples))
    ssims.append(float(ssim))
    psnrs.append(float(psnr))

records.update(
    designs=np.stack(designs_seq),
    ys=np.stack(ys_seq),
    masks=np.stack(masks_seq),
    samples=np.stack(samples_seq),
    ssims=np.asarray(ssims),
    psnrs=np.asarray(psnrs),
)
if design_trajs:
    records.update(
        design_trajs=np.stack(design_trajs),
        eig_trajs=np.stack(eig_trajs),
        t_trajs=np.stack(t_trajs),
    )
np.savez(args.out, **records)
print("saved:", args.out, flush=True)
