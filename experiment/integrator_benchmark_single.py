"""Run one (solver, config) benchmark of the torus neural SDE and dump JSON.

Intended to be launched as a fresh subprocess so peak-memory measurement is
not contaminated by JIT caches from other solver runs in the same process.
Mirrors ``../CFEES-Neural-SDEs/experiment/integrator_benchmark_single.py``
adapted to the torus model.

Usage::

    python -m experiment.integrator_benchmark_single \
        --num-angles 7 --n-steps 50 --solver cfees25 --n-reps 10
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import psutil

from experiment.config import Solvers
from experiment.factories import build_solver
from models.torus_nsde import TorusNeuralSDE


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024**2)


def _hlo_temp_bytes(grad_fn, model, context, key) -> int | None:
    """Compile the gradient graph once and read XLA's scratch buffer size."""
    try:
        compiled = grad_fn.lower(model, context, key).compile()
        return int(compiled.compiled.memory_analysis().temp_size_in_bytes)
    except Exception:  # noqa: BLE001
        return None


def run(
    num_angles: int,
    n_steps: int,
    solver: Solvers,
    n_reps: int,
    *,
    context_length: int = 20,
    hidden_dim: int = 128,
    ctx_dim: int = 64,
    dt: float = 0.05,
    batch_size: int = 32,
) -> dict:
    """Benchmark one ``(solver, num_angles, n_steps)`` forward+backward pass.

    Returns a dict containing:
    - ``compile_footprint_mb``: ``rss(post-compile) - rss(post-model-built)``
    - ``rep_peak_incr_mb``: peak additional RSS during timed reps
    - ``rep_peak_rss_mb``: absolute peak RSS during reps
    - ``rss_stage_{a,b,c,d}_mb``: RSS snapshots at successive stages
    - ``warmup_s`` and summary statistics of ``n_reps`` timed steps
    - ``temp_bytes``: XLA compile-time scratch (if available)
    """
    rss_stage_a = rss_mb()

    model = TorusNeuralSDE(
        num_angles=num_angles,
        hidden_dim=hidden_dim,
        ctx_dim=ctx_dim,
        n_steps=n_steps,
        dt=dt,
        solver=build_solver(solver),
        diffusion_scale=0.1,
        key=jax.random.key(0),
    )
    context = jax.random.uniform(
        jax.random.key(1),
        (batch_size, context_length, num_angles),
        minval=-jnp.pi,
        maxval=jnp.pi,
    )
    jax.block_until_ready(context)

    gc.collect()
    rss_stage_b = rss_mb()  # model + data on device, pre-compile

    def loss_fn(m: eqx.Module, ctx: jax.Array, k: jax.Array) -> jax.Array:
        sample_keys = jax.random.split(k, ctx.shape[0])
        preds = jax.vmap(lambda c, sk: m(c, sk))(ctx, sample_keys)
        return jnp.sum(preds**2)

    grad_fn = eqx.filter_jit(eqx.filter_value_and_grad(loss_fn))
    rep_keys = jax.random.split(jax.random.key(42), n_reps + 1)

    temp_bytes = _hlo_temp_bytes(grad_fn, model, context, rep_keys[0])

    t0 = time.time()
    loss, grads = grad_fn(model, context, rep_keys[0])
    jax.block_until_ready((loss, grads))
    t_warmup = time.time() - t0

    gc.collect()
    rss_stage_c = rss_mb()  # post-compile, post-warmup

    times: list[float] = []
    rep_peak_rss = rss_stage_c
    for i in range(n_reps):
        t0 = time.time()
        loss, grads = grad_fn(model, context, rep_keys[i + 1])
        jax.block_until_ready((loss, grads))
        times.append(time.time() - t0)
        rep_peak_rss = max(rep_peak_rss, rss_mb())

    rss_stage_d = rss_mb()  # after all reps

    return {
        "num_angles": num_angles,
        "n_steps": n_steps,
        "solver": solver.value,
        "context_length": context_length,
        "hidden_dim": hidden_dim,
        "ctx_dim": ctx_dim,
        "dt": dt,
        "batch_size": batch_size,
        "warmup_s": float(t_warmup),
        "mean_s": float(np.mean(times)),
        "std_s": float(np.std(times)),
        "median_s": float(np.median(times)),
        "min_s": float(np.min(times)),
        "max_s": float(np.max(times)),
        "n_reps": n_reps,
        "times_s": [float(t) for t in times],
        "compile_footprint_mb": float(rss_stage_c - rss_stage_b),
        "rep_peak_incr_mb": float(rep_peak_rss - rss_stage_c),
        "rep_peak_rss_mb": float(rep_peak_rss),
        "rss_stage_a_mb": float(rss_stage_a),
        "rss_stage_b_mb": float(rss_stage_b),
        "rss_stage_c_mb": float(rss_stage_c),
        "rss_stage_d_mb": float(rss_stage_d),
        "temp_bytes": temp_bytes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-angles", type=int, required=True)
    parser.add_argument("--n-steps", type=int, required=True)
    parser.add_argument("--solver", type=str, required=True)
    parser.add_argument("--n-reps", type=int, default=10)
    parser.add_argument("--context-length", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--ctx-dim", type=int, default=64)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    out = run(
        args.num_angles,
        args.n_steps,
        Solvers(args.solver),
        args.n_reps,
        context_length=args.context_length,
        hidden_dim=args.hidden_dim,
        ctx_dim=args.ctx_dim,
        dt=args.dt,
        batch_size=args.batch_size,
    )
    json.dump(out, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
