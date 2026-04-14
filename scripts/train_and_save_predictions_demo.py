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
from experiment.factories import make_loader, make_model, make_prediction_fn
from experiment.rna_losses import make_wrapped_mae_metric, make_wrapped_mse_loss

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = PROJECT_ROOT / "results" / "rna_predictions_demo.npz"


def main() -> int:
    config = make_config(
        experiment=Experiments.RNA,
        epochs=20,
        batch_size=32,
        learning_rate=3e-4,
        seed=0,
        device=Devices.GPU,
        hidden_dim=128,
        ctx_dim=64,
        n_steps=50,
        dt=0.05,
        solver=Solvers.CFEES25,
        diffusion_scale=0.1,
        context_length=20,
        residues_per_state=1,
        max_chains=100,
    )

    train_ds = RNATorsionDataset(
        split="train",
        max_chains=config.max_chains,
        context_length=config.context_length,
        residues_per_state=config.residues_per_state,
    )
    metadata = train_ds.metadata()

    key = jax.random.key(config.seed)
    model = make_model(config, metadata, key)
    prediction_fn = make_prediction_fn()
    loss_fn = make_wrapped_mse_loss(
        input_key="context_angles",
        target_key="target_angles",
        prediction_fn=prediction_fn,
    )
    metric_fn = make_wrapped_mae_metric(
        input_key="context_angles",
        target_key="target_angles",
        prediction_fn=prediction_fn,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(config.learning_rate),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    train_loader = make_loader(config, "train")
    train_loader_next = jax.jit(train_loader.next)
    val_loader = make_loader(config, "val")
    val_loader_next = jax.jit(val_loader.next)

    key, train_key, val_key = jax.random.split(key, 3)
    train_state = train_loader.init_state(train_key)
    val_state = val_loader.init_state(val_key)

    @eqx.filter_jit
    def train_step(m, opt_state, batch, mask, step_key):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(m, batch, mask, step_key)
        updates, new_opt_state = optimizer.update(grads, opt_state, m)
        return eqx.apply_updates(m, updates), new_opt_state, loss

    eval_metric = eqx.filter_jit(metric_fn)

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
    context = jnp.asarray(arrays["context_angles"])
    target = jnp.asarray(arrays["target_angles"])

    sample_keys = jax.random.split(jax.random.key(config.seed + 99), context.shape[0])
    predict_fn = jax.vmap(lambda c, k: model(c, k))
    predicted = jax.device_get(predict_fn(context, sample_keys))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_PATH,
        predicted=np.asarray(predicted),
        actual=np.asarray(target),
        context_angles=np.asarray(context),
    )
    print(f"saved {OUT_PATH}", flush=True)
    print(f"n_test = {predicted.shape[0]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
