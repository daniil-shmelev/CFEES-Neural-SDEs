"""Cyreal dataset for one-step RNA torsion-angle forecasting on T^d."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from cyreal.datasets.dataset_protocol import DatasetProtocol
from cyreal.datasets.utils import to_host_jax_array
from cyreal.sources import ArraySource

from datasets.rna.download import (
    CACHE_DIR,
    PDB_CACHE_DIR,
    ensure_pdb_files,
    fetch_bgsu_nr_list,
    parse_chain_token,
)
from datasets.rna.preprocessing import (
    NUM_ANGLES,
    compute_chain_torsions,
    load_chain_atoms,
    split_into_contiguous_runs,
)

logger = logging.getLogger(__name__)


@dataclass
class RNATorsionDataset(DatasetProtocol):
    """One-step-ahead forecasting dataset on the n-torus.

    Each sample is a dict with:
    - ``context_angles``: ``(context_length, 7 * residues_per_state)``
    - ``target_angles``:  ``(7 * residues_per_state,)``

    Splits are over chain identity (not residue index) so that no residue
    leaks between train/val/test.
    """

    split: Literal["train", "val", "test"] = "train"
    context_length: int = 20
    residues_per_state: int = 1
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    cutoff_angstrom: float = 3.0
    max_chains: int | None = 200
    cache_dir: Path = CACHE_DIR
    pdb_cache_dir: Path = PDB_CACHE_DIR
    min_chain_length: int = 30
    seed: int = 0
    ordering: Literal["sequential", "shuffle"] = field(init=False)

    def __post_init__(self) -> None:
        if self.context_length <= 0:
            raise ValueError("context_length must be positive.")
        if self.residues_per_state <= 0:
            raise ValueError("residues_per_state must be positive.")
        self.ordering = "shuffle" if self.split == "train" else "sequential"

        chains = _load_or_build_chain_angles(
            cutoff_angstrom=self.cutoff_angstrom,
            max_chains=self.max_chains,
            cache_dir=Path(self.cache_dir),
            pdb_cache_dir=Path(self.pdb_cache_dir),
            min_chain_length=self.min_chain_length,
        )
        chain_ids = sorted(chains.keys())

        train_ids, val_ids, test_ids = _chain_level_split(
            chain_ids,
            train_fraction=self.train_fraction,
            val_fraction=self.val_fraction,
            seed=self.seed,
        )
        if self.split == "train":
            split_ids = train_ids
        elif self.split == "val":
            split_ids = val_ids
        else:
            split_ids = test_ids

        contexts, targets = _build_pairs(
            {cid: chains[cid] for cid in split_ids},
            context_length=self.context_length,
            residues_per_state=self.residues_per_state,
        )

        self._contexts = to_host_jax_array(contexts)
        self._targets = to_host_jax_array(targets)
        self._num_angles = int(NUM_ANGLES * self.residues_per_state)
        self._n_chains = len(split_ids)

    def __len__(self) -> int:
        return int(self._contexts.shape[0])

    def __getitem__(self, index: int) -> dict[str, jax.Array]:
        return {
            "context_angles": self._contexts[index],
            "target_angles": self._targets[index],
        }

    def as_array_dict(self) -> dict[str, jax.Array]:
        return {
            "context_angles": self._contexts,
            "target_angles": self._targets,
        }

    def metadata(self) -> dict[str, int]:
        return {
            "num_angles": self._num_angles,
            "context_length": self.context_length,
            "residues_per_state": self.residues_per_state,
            "dataset_size": len(self),
            "n_chains": self._n_chains,
        }

    def make_array_source(self) -> ArraySource:
        return ArraySource(self.as_array_dict(), ordering=self.ordering)


def _chain_level_split(
    chain_ids: list[str],
    *,
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1).")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1).")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be < 1.")

    train, val, test = [], [], []
    for cid in chain_ids:
        h = hashlib.sha256(f"{seed}:{cid}".encode()).digest()
        bucket = int.from_bytes(h[:8], "big") / 2**64
        if bucket < train_fraction:
            train.append(cid)
        elif bucket < train_fraction + val_fraction:
            val.append(cid)
        else:
            test.append(cid)
    return train, val, test


def _build_pairs(
    chains: dict[str, list[np.ndarray]],
    *,
    context_length: int,
    residues_per_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (context, target) pairs from per-chain valid runs.

    Each chain contributes a list of contiguous-residue runs. Within a run we
    group every ``residues_per_state`` consecutive residues into one state of
    shape ``(7 * residues_per_state,)``. We then slide a window of length
    ``context_length`` over the resulting state sequence and emit one pair per
    window position whose target state is the immediately following state.
    """
    state_dim = NUM_ANGLES * residues_per_state
    contexts: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    for runs in chains.values():
        for run in runs:
            n_states = run.shape[0] // residues_per_state
            if n_states <= context_length:
                continue
            usable = n_states * residues_per_state
            grouped = run[:usable].reshape(n_states, state_dim)
            num_pairs = n_states - context_length
            row_idx = np.arange(num_pairs)[:, None] + np.arange(context_length)[None, :]
            contexts.append(grouped[row_idx])
            targets.append(grouped[context_length : context_length + num_pairs])

    if not contexts:
        return (
            np.zeros((0, context_length, state_dim), dtype=np.float32),
            np.zeros((0, state_dim), dtype=np.float32),
        )
    return (
        np.concatenate(contexts, axis=0).astype(np.float32),
        np.concatenate(targets, axis=0).astype(np.float32),
    )


def _cache_key(*, cutoff_angstrom: float, max_chains: int | None) -> str:
    payload = f"cutoff={cutoff_angstrom}|max_chains={max_chains}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _load_or_build_chain_angles(
    *,
    cutoff_angstrom: float,
    max_chains: int | None,
    cache_dir: Path,
    pdb_cache_dir: Path,
    min_chain_length: int,
) -> dict[str, list[np.ndarray]]:
    """Return ``{chain_token: [run_array, ...]}`` for valid contiguous runs."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"chain_angles_{_cache_key(cutoff_angstrom=cutoff_angstrom, max_chains=max_chains)}.npz"
    chains: dict[str, list[np.ndarray]] = {}

    if cache_path.exists():
        logger.info("Loading cached RNA torsion data from %s", cache_path)
        data = np.load(cache_path, allow_pickle=False)
        for key in data.files:
            if not key.startswith("angles__"):
                continue
            sanitized = key[len("angles__") :]
            token = sanitized.replace("__", "|")
            run_lengths = data[f"runs__{sanitized}"]
            angles = data[key]
            offset = 0
            runs: list[np.ndarray] = []
            for length in run_lengths:
                runs.append(angles[offset : offset + int(length)])
                offset += int(length)
            chains[token] = runs
        return chains

    logger.info(
        "Building RNA torsion cache (cutoff=%sA, max_chains=%s)",
        cutoff_angstrom,
        max_chains,
    )
    tokens = fetch_bgsu_nr_list(cutoff_angstrom=cutoff_angstrom)
    if max_chains is not None:
        tokens = tokens[:max_chains]
    pdb_ids = sorted({parse_chain_token(t)[0] for t in tokens})
    paths = ensure_pdb_files(pdb_ids, cache_dir=pdb_cache_dir)

    for token in tokens:
        try:
            pdb_id, model, chain_id = parse_chain_token(token)
        except ValueError as exc:
            logger.warning("Skipping malformed token %r: %s", token, exc)
            continue
        path = paths.get(pdb_id)
        if path is None:
            continue
        atoms = load_chain_atoms(path, chain_id, model=model)
        if atoms is None:
            continue
        angles, res_ids = compute_chain_torsions(atoms)
        if len(angles) < min_chain_length:
            continue
        runs = [r for r in split_into_contiguous_runs(angles, res_ids) if len(r) >= min_chain_length]
        if not runs:
            continue
        chains[token] = runs

    payload: dict[str, np.ndarray] = {}
    for token, runs in chains.items():
        sanitized = token.replace("|", "__")
        all_angles = np.concatenate(runs, axis=0).astype(np.float32)
        run_lengths = np.asarray([len(r) for r in runs], dtype=np.int64)
        payload[f"angles__{sanitized}"] = all_angles
        payload[f"runs__{sanitized}"] = run_lengths

    np.savez(cache_path, **payload)
    logger.info("Cached %d chains to %s", len(chains), cache_path)
    return chains
