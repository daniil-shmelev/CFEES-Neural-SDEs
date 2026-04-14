"""Wrapped-angle losses and metrics for torus-valued targets."""

from __future__ import annotations

from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from experiment.losses import masked_mean
from models.torus import wrap_to_pi

PyTree = Any
LossFn = Callable[[eqx.Module, PyTree, jax.Array, jax.Array], jax.Array]


def wrapped_angle_diff(a: jax.Array, b: jax.Array) -> jax.Array:
    return wrap_to_pi(a - b)


def make_wrapped_mse_loss(
    *,
    input_key: str,
    target_key: str,
    prediction_fn: Callable[[eqx.Module, jax.Array, jax.Array], jax.Array],
) -> LossFn:
    def loss_fn(model, batch, mask, key):
        inputs = batch[input_key]
        targets = batch[target_key]
        predictions = prediction_fn(model, inputs, key)
        per_example = jnp.mean(wrapped_angle_diff(predictions, targets) ** 2, axis=-1)
        return masked_mean(per_example, mask)

    return loss_fn


def make_wrapped_mae_metric(
    *,
    input_key: str,
    target_key: str,
    prediction_fn: Callable[[eqx.Module, jax.Array, jax.Array], jax.Array],
) -> LossFn:
    def metric_fn(model, batch, mask, key):
        inputs = batch[input_key]
        targets = batch[target_key]
        predictions = prediction_fn(model, inputs, key)
        per_example = jnp.mean(jnp.abs(wrapped_angle_diff(predictions, targets)), axis=-1)
        return masked_mean(per_example, mask)

    return metric_fn
