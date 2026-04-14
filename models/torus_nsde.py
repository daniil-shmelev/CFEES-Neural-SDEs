"""Neural SDE on the flat n-torus T^d = SO(2)^d for RNA torsion forecasting."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from diffrax import (
    AbstractReversibleSolver,
    AbstractSolver,
    ControlTerm,
    DirectAdjoint,
    MultiTerm,
    ODETerm,
    ReversibleAdjoint,
    SaveAt,
    VirtualBrownianTree,
    diffeqsolve,
)
from georax import CFEES25, GeometricTerm

from models.torus import Torus, wrap_to_pi


def angle_features(theta: jax.Array) -> jax.Array:
    """Standard sin/cos encoding for inputs that live on a torus."""
    return jnp.concatenate([jnp.sin(theta), jnp.cos(theta)])


class TorusDriftField(eqx.Module):
    mlp: eqx.nn.MLP
    geometry: Torus

    def __init__(self, geometry, ctx_dim, hidden_dim, *, key):
        d = geometry.dimension
        mlp = eqx.nn.MLP(
            in_size=2 * d + ctx_dim,
            out_size=d,
            width_size=hidden_dim,
            depth=3,
            activation=jax.nn.silu,
            key=key,
        )
        last = mlp.layers[-1]
        last = eqx.tree_at(lambda l: l.bias, last, jnp.zeros_like(last.bias))
        self.mlp = eqx.tree_at(lambda m: m.layers[-1], mlp, last)
        self.geometry = geometry

    def __call__(self, t, theta, ctx):
        del t
        inp = jnp.concatenate([angle_features(theta), ctx])
        coeffs = self.mlp(inp)
        return self.geometry.from_frame(theta, coeffs)


class TorusDiffusionField(eqx.Module):
    mlp: eqx.nn.MLP
    geometry: Torus
    diffusion_scale: float

    def __init__(self, geometry, ctx_dim, hidden_dim, diffusion_scale, *, key):
        d = geometry.dimension
        mlp = eqx.nn.MLP(
            in_size=2 * d + ctx_dim,
            out_size=d,
            width_size=hidden_dim,
            depth=2,
            activation=jax.nn.silu,
            key=key,
        )
        last = mlp.layers[-1]
        last = eqx.tree_at(lambda l: l.weight, last, last.weight * 0.1)
        last = eqx.tree_at(lambda l: l.bias, last, jnp.full_like(last.bias, -2.0))
        self.mlp = eqx.tree_at(lambda m: m.layers[-1], mlp, last)
        self.geometry = geometry
        self.diffusion_scale = diffusion_scale

    def __call__(self, t, theta, ctx):
        del t
        inp = jnp.concatenate([angle_features(theta), ctx])
        scales = jax.nn.softplus(self.mlp(inp)) * self.diffusion_scale
        basis = self.geometry.frame(theta)  # (d, d) identity
        return basis * scales[None, :]


class GRUEncoder(eqx.Module):
    """GRU encoder over a sequence of sin/cos angle states."""

    cell: eqx.nn.GRUCell
    proj: eqx.nn.Linear
    hidden_dim: int = eqx.field(static=True)

    def __init__(self, input_dim, hidden_dim, ctx_dim, *, key):
        k1, k2 = jax.random.split(key)
        self.cell = eqx.nn.GRUCell(input_dim, hidden_dim, key=k1)
        self.proj = eqx.nn.Linear(hidden_dim, ctx_dim, key=k2)
        self.hidden_dim = hidden_dim

    def __call__(self, x_seq):
        def step(h, x):
            h = self.cell(x, h)
            return h, None

        h0 = jnp.zeros((self.hidden_dim,), dtype=x_seq.dtype)
        h_final, _ = jax.lax.scan(step, h0, x_seq)
        return self.proj(h_final)


class TorusNeuralSDE(eqx.Module):
    """Neural SDE on T^d for one-step torsion-angle forecasting."""

    encoder: GRUEncoder
    drift_field: TorusDriftField
    diffusion_field: TorusDiffusionField
    name: str = eqx.field(static=True)

    d: int = eqx.field(static=True)
    ctx_dim: int = eqx.field(static=True)
    n_steps: int = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    solver: AbstractSolver = eqx.field(static=True)

    def __init__(
        self,
        num_angles: int,
        hidden_dim: int = 128,
        ctx_dim: int = 64,
        n_steps: int = 5,
        dt: float = 0.05,
        solver: AbstractSolver = CFEES25(),
        diffusion_scale: float = 0.3,
        *,
        key,
    ):
        k1, k2, k3 = jax.random.split(key, 3)

        geometry = Torus(num_angles)
        d = geometry.dimension

        self.name = "torus_nsde"
        self.d = d
        self.ctx_dim = ctx_dim
        self.n_steps = n_steps
        self.dt = dt
        self.solver = solver

        self.encoder = GRUEncoder(2 * d, hidden_dim, ctx_dim, key=k1)
        self.drift_field = TorusDriftField(
            geometry=geometry, ctx_dim=ctx_dim, hidden_dim=hidden_dim, key=k2
        )
        self.diffusion_field = TorusDiffusionField(
            geometry=geometry,
            ctx_dim=ctx_dim,
            hidden_dim=hidden_dim,
            diffusion_scale=diffusion_scale,
            key=k3,
        )

    def __call__(self, context_angles, key):
        """context_angles: (T, d). Returns predicted (d,) angle state."""
        y0 = wrap_to_pi(context_angles[-1])

        context_features = jax.vmap(angle_features)(context_angles)
        ctx = self.encoder(context_features)

        t1 = self.n_steps * self.dt
        brownian_path = VirtualBrownianTree(
            t0=0.0,
            t1=t1,
            tol=self.dt / 4.0,
            shape=(self.d,),
            key=key,
        )

        term = GeometricTerm(
            inner=MultiTerm(
                ODETerm(self.drift_field),
                ControlTerm(self.diffusion_field, brownian_path),
            ),
            geometry=self.drift_field.geometry,
        )

        adjoint = (
            ReversibleAdjoint()
            if isinstance(self.solver, AbstractReversibleSolver)
            else DirectAdjoint()
        )

        sol = diffeqsolve(
            term,
            self.solver,
            t0=0.0,
            t1=t1,
            dt0=self.dt,
            y0=y0,
            args=ctx,
            saveat=SaveAt(t1=True),
            adjoint=adjoint,
            max_steps=self.n_steps + 8,
        )

        return wrap_to_pi(sol.ys[0])
