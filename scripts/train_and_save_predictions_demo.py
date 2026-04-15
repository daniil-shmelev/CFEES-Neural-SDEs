"""Train one CFEES25 cell and save test predictions plus context angles.

Regenerates ``results/rna_predictions_demo.npz`` so the figure in
``report/generate_figures.py::fig_predictions`` is self-contained and
aligned with the current dataset cache.

Usage::

    python scripts/train_and_save_predictions_demo.py
"""

from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from datasets.rna.dataset import RNATorsionDataset
from experiment.config import Devices, Experiments, Solvers, make_config
from experiment.factories import make_loader, make_model
from experiment.rna_losses import (
    make_wrapped_energy_score_loss,
    make_wrapped_mae_metric,
    make_wrapped_mse_loss,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = PROJECT_ROOT / "results" / "rna_predictions_demo.npz"


def main() -> int:
    config = make_config(
        experiment=Experiments.RNA,
        epochs=80,
        batch_size=64,
        learning_rate=1e-4,
        seed=0,
        device=Devices.GPU,
        hidden_dim=256,
        ctx_dim=128,
        n_steps=50,
        dt=0.05,
        solver=Solvers.CFEES25,
        diffusion_scale=0.1,
        context_length=20,
        residues_per_state=1,
        max_chains=1000,
    )
    n_mc_samples = 16
    warmup_steps = 1000
    weight_decay = 1e-4
    use_energy_score = False
    energy_score_samples = 8

    train_ds = RNATorsionDataset(
        split="train",
        max_chains=config.max_chains,
        context_length=config.context_length,
        residues_per_state=config.residues_per_state,
    )
    metadata = train_ds.metadata()

    key = jax.random.key(config.seed)
    model = make_model(config, metadata, key)
    if use_energy_score:
        loss_fn = make_wrapped_energy_score_loss(n_samples=energy_score_samples)
    else:
        loss_fn = make_wrapped_mse_loss()
    metric_fn = make_wrapped_mae_metric()
    train_loader = make_loader(config, "train")
    train_loader_next = jax.jit(train_loader.next)
    val_loader = make_loader(config, "val")
    val_loader_next = jax.jit(val_loader.next)

    total_train_steps = train_loader.steps_per_epoch * config.epochs
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.learning_rate,
        warmup_steps=min(warmup_steps, total_train_steps // 4),
        decay_steps=max(total_train_steps - warmup_steps, 1),
        end_value=config.learning_rate * 0.05,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=weight_decay),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    key, train_key, val_key = jax.random.split(key, 3)
    train_state = train_loader.init_state(train_key)
    val_state = val_loader.init_state(val_key)
    print(
        f"train={train_loader.steps_per_epoch * config.batch_size}"
        f" val={val_loader.steps_per_epoch * config.batch_size}"
        f" n_mc={n_mc_samples} warmup={warmup_steps}/{total_train_steps}",
        flush=True,
    )

    @eqx.filter_jit
    def train_step(m, opt_state, batch, mask, step_key):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(m, batch, mask, step_key)
        updates, new_opt_state = optimizer.update(grads, opt_state, m)
        return eqx.apply_updates(m, updates), new_opt_state, loss

    eval_metric = eqx.filter_jit(metric_fn)

    best_ckpt_path = PROJECT_ROOT / "results" / "rna_best_model.eqx"
    best_ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    best_model = model
    best_val = float("inf")
    for epoch in range(config.epochs):
        total = 0.0
        for _ in range(train_loader.steps_per_epoch):
            key, step_key = jax.random.split(key)
            batch, train_state, mask = train_loader_next(train_state)
            model, opt_state, loss = train_step(model, opt_state, batch, mask, step_key)
            total += float(loss)
        val_metric = 0.0
        for _ in range(val_loader.steps_per_epoch):
            key, step_key = jax.random.split(key)
            batch, val_state, mask = val_loader_next(val_state)
            val_metric += float(eval_metric(model, batch, mask, step_key))
        val_metric /= val_loader.steps_per_epoch
        if val_metric < best_val:
            best_val = val_metric
            best_model = model
            eqx.tree_serialise_leaves(best_ckpt_path, best_model)
        print(
            f"epoch {epoch + 1:2d}/{config.epochs}  "
            f"train_loss={total / train_loader.steps_per_epoch:.4f}  "
            f"val_mae={val_metric:.4f}  best_val={best_val:.4f}",
            flush=True,
        )
    model = best_model

    test_ds = RNATorsionDataset(
        split="test",
        max_chains=config.max_chains,
        context_length=config.context_length,
        residues_per_state=config.residues_per_state,
    )
    arrays = test_ds.as_array_dict()
    context_angles = jnp.asarray(arrays["context_angles"])
    context_bases = jnp.asarray(arrays["context_bases"])
    target_angles = jnp.asarray(arrays["target_angles"])
    target_bases = jnp.asarray(arrays["target_bases"])

    # Test-time averaging: draw n_mc_samples stochastic rollouts per context
    # and circular-mean them on the torus.
    base_key = jax.random.key(config.seed + 99)
    per_sample_keys = jax.random.split(base_key, n_mc_samples)

    @jax.jit
    def predict_ensemble(ctx_a, ctx_b, tgt_b, keys):
        def one_sample(k):
            sub_keys = jax.random.split(k, ctx_a.shape[0])
            return jax.vmap(
                lambda a, cb, tb, kk: model(a, cb, tb, kk)
            )(ctx_a, ctx_b, tgt_b, sub_keys)
        samples = jax.vmap(one_sample)(keys)  # (n_mc, n_test, d)
        mean_sin = jnp.mean(jnp.sin(samples), axis=0)
        mean_cos = jnp.mean(jnp.cos(samples), axis=0)
        return jnp.arctan2(mean_sin, mean_cos)

    predicted = jax.device_get(
        predict_ensemble(context_angles, context_bases, target_bases, per_sample_keys)
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_PATH,
        predicted=np.asarray(predicted),
        actual=np.asarray(target_angles),
        context_angles=np.asarray(context_angles),
        context_bases=np.asarray(context_bases),
        target_bases=np.asarray(target_bases),
    )
    print(f"saved {OUT_PATH}", flush=True)
    print(f"n_test = {predicted.shape[0]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
