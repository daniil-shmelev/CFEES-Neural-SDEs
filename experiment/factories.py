from typing import Callable, Literal

import equinox as eqx
import jax
from cyreal.loader import DataLoader
from cyreal.transforms import BatchTransform
from diffrax import AbstractSolver
from georax import CFEES25, CFEES27, CG2, CG4, RKMK

from experiment.config import ExperimentConfig, Experiments, Solvers

_SOLVER_CLASSES: dict[Solvers, type[AbstractSolver]] = {
    Solvers.CFEES25: CFEES25,
    Solvers.CFEES27: CFEES27,
    Solvers.CG2: CG2,
    Solvers.CG4: CG4,
    Solvers.RKMK: RKMK,
}


def build_solver(name: Solvers) -> AbstractSolver:
    cls = _SOLVER_CLASSES[name]
    return cls()


def make_loader(
    config: ExperimentConfig,
    split: Literal["train", "val", "test"],
) -> DataLoader:
    match config.experiment:
        case Experiments.SPD:
            from datasets.spd.dataset import CovarianceDataset

            source = CovarianceDataset(split=split).make_array_source()
            return DataLoader(
                [
                    source,
                    BatchTransform(
                        batch_size=config.batch_size,
                        drop_last=split == "train",
                    ),
                ]
            )
        case Experiments.RNA:
            from datasets.rna.dataset import RNATorsionDataset

            source = RNATorsionDataset(
                split=split,
                context_length=config.context_length,
                residues_per_state=config.residues_per_state,
                max_chains=config.max_chains,
            ).make_array_source()
            return DataLoader(
                [
                    source,
                    BatchTransform(
                        batch_size=config.batch_size,
                        drop_last=split == "train",
                    ),
                ]
            )
        case _:
            raise ValueError(f"Unsupported experiment: {config.experiment}")


def make_model(
    config: ExperimentConfig,
    metadata: dict[str, int],
    key: jax.Array,
) -> eqx.Module:
    match config.experiment:
        case Experiments.SPD:
            from models.nsde import ManifoldNeuralSDE

            return ManifoldNeuralSDE(
                n_stocks=metadata["n_stocks"],
                hidden_dim=config.hidden_dim,
                ctx_dim=config.ctx_dim,
                n_steps=config.n_steps,
                dt=config.dt,
                solver=build_solver(config.solver),
                diffusion_scale=config.diffusion_scale,
                key=key,
            )
        case Experiments.RNA:
            from models.torus_nsde import TorusNeuralSDE

            return TorusNeuralSDE(
                num_angles=metadata["num_angles"],
                hidden_dim=config.hidden_dim,
                ctx_dim=config.ctx_dim,
                n_steps=config.n_steps,
                dt=config.dt,
                solver=build_solver(config.solver),
                diffusion_scale=config.diffusion_scale,
                residues_per_state=config.residues_per_state,
                key=key,
            )
        case _:
            raise ValueError(f"Unsupported experiment: {config.experiment}")


def make_prediction_fn() -> Callable[[eqx.Module, jax.Array, jax.Array], jax.Array]:
    def prediction_fn(
        model: eqx.Module,
        inputs: jax.Array,
        key: jax.Array,
    ) -> jax.Array:
        sample_keys = jax.random.split(key, inputs.shape[0])
        return jax.vmap(lambda context, sample_key: model(context, sample_key))(
            inputs, sample_keys
        )

    return prediction_fn
