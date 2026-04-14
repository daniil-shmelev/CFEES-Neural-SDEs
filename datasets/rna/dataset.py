"""Cyreal dataset for one-step RNA torsion-angle forecasting on T^d."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import jax
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
    NUM_BASES,
    compute_chain_torsions,
    load_chain_atoms,
    split_into_contiguous_runs,
)

logger = logging.getLogger(__name__)

# Bumped when the on-disk cache layout changes. Old caches are ignored.
CACHE_LAYOUT_VERSION = "v2"


@dataclass
class RNATorsionDataset(DatasetProtocol):
    """One-step-ahead forecasting dataset on the n-torus.

    Each sample is a dict with:
    - ``context_angles``: ``(context_length, 7 * residues_per_state)`` float32
    - ``context_bases``:  ``(context_length * residues_per_state,)`` int32 in 0..3
    - ``target_angles``:  ``(7 * residues_per_state,)`` float32
    - ``target_bases``:   ``(residues_per_state,)`` int32 in 0..3

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

        c_angles, c_bases, t_angles, t_bases = _build_pairs(
            {cid: chains[cid] for cid in split_ids},
            context_length=self.context_length,
            residues_per_state=self.residues_per_state,
        )

        self._context_angles = to_host_jax_array(c_angles)
        self._context_bases = to_host_jax_array(c_bases)
        self._target_angles = to_host_jax_array(t_angles)
        self._target_bases = to_host_jax_array(t_bases)
        self._num_angles = int(NUM_ANGLES * self.residues_per_state)
        self._n_chains = len(split_ids)

    def __len__(self) -> int:
        return int(self._context_angles.shape[0])

    def __getitem__(self, index: int) -> dict[str, jax.Array]:
        return {
            "context_angles": self._context_angles[index],
            "context_bases": self._context_bases[index],
            "target_angles": self._target_angles[index],
            "target_bases": self._target_bases[index],
        }

    def as_array_dict(self) -> dict[str, jax.Array]:
        return {
            "context_angles": self._context_angles,
            "context_bases": self._context_bases,
            "target_angles": self._target_angles,
            "target_bases": self._target_bases,
        }

    def metadata(self) -> dict[str, int]:
        return {
            "num_angles": self._num_angles,
            "num_bases": NUM_BASES,
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
    chains: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    *,
    context_length: int,
    residues_per_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build ``(context_angles, context_bases, target_angles, target_bases)``.

    Each chain contributes a list of contiguous-residue runs, where each run
    is a ``(angles, base_ids)`` pair. Within a run we group every
    ``residues_per_state`` consecutive residues into one state of shape
    ``(7 * residues_per_state,)``. We slide a window of length
    ``context_length`` over the resulting state sequence and emit one pair
    per window position whose target state is the immediately following
    state. Base identities flow through the same reshape / slide so each
    state has ``residues_per_state`` bases attached.
    """
    state_dim = NUM_ANGLES * residues_per_state
    bases_per_state = residues_per_state
    c_angles_chunks: list[np.ndarray] = []
    c_bases_chunks: list[np.ndarray] = []
    t_angles_chunks: list[np.ndarray] = []
    t_bases_chunks: list[np.ndarray] = []

    for runs in chains.values():
        for run_angles, run_bases in runs:
            n_states = run_angles.shape[0] // residues_per_state
            if n_states <= context_length:
                continue
            usable = n_states * residues_per_state
            grouped_angles = run_angles[:usable].reshape(n_states, state_dim)
            grouped_bases = run_bases[:usable].reshape(n_states, bases_per_state)
            num_pairs = n_states - context_length
            row_idx = np.arange(num_pairs)[:, None] + np.arange(context_length)[None, :]
            c_angles_chunks.append(grouped_angles[row_idx])
            c_bases_chunks.append(grouped_bases[row_idx])
            t_angles_chunks.append(
                grouped_angles[context_length : context_length + num_pairs]
            )
            t_bases_chunks.append(
                grouped_bases[context_length : context_length + num_pairs]
            )

    if not c_angles_chunks:
        return (
            np.zeros((0, context_length, state_dim), dtype=np.float32),
            np.zeros((0, context_length, bases_per_state), dtype=np.int32),
            np.zeros((0, state_dim), dtype=np.float32),
            np.zeros((0, bases_per_state), dtype=np.int32),
        )

    c_angles = np.concatenate(c_angles_chunks, axis=0).astype(np.float32)
    c_bases = np.concatenate(c_bases_chunks, axis=0).astype(np.int32)
    t_angles = np.concatenate(t_angles_chunks, axis=0).astype(np.float32)
    t_bases = np.concatenate(t_bases_chunks, axis=0).astype(np.int32)
    return c_angles, c_bases, t_angles, t_bases


def _cache_key(*, cutoff_angstrom: float, max_chains: int | None) -> str:
    payload = (
        f"{CACHE_LAYOUT_VERSION}|cutoff={cutoff_angstrom}|max_chains={max_chains}"
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _load_or_build_chain_angles(
    *,
    cutoff_angstrom: float,
    max_chains: int | None,
    cache_dir: Path,
    pdb_cache_dir: Path,
    min_chain_length: int,
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Return ``{chain_token: [(angles, base_ids), ...]}`` for valid runs."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"chain_angles_{_cache_key(cutoff_angstrom=cutoff_angstrom, max_chains=max_chains)}.npz"
    chains: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}

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
            bases = data[f"bases__{sanitized}"]
            offset = 0
            runs: list[tuple[np.ndarray, np.ndarray]] = []
            for length in run_lengths:
                length = int(length)
                runs.append(
                    (angles[offset : offset + length], bases[offset : offset + length])
                )
                offset += length
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
        angles, res_ids, base_ids = compute_chain_torsions(atoms)
        if len(angles) < min_chain_length:
            continue
        runs = [
            (a, b)
            for (a, b) in split_into_contiguous_runs(angles, res_ids, base_ids)
            if len(a) >= min_chain_length
        ]
        if not runs:
            continue
        chains[token] = runs

    payload: dict[str, np.ndarray] = {}
    for token, runs in chains.items():
        sanitized = token.replace("|", "__")
        run_angles = [a for (a, _) in runs]
        run_bases = [b for (_, b) in runs]
        all_angles = np.concatenate(run_angles, axis=0).astype(np.float32)
        all_bases = np.concatenate(run_bases, axis=0).astype(np.int8)
        run_lengths = np.asarray([len(a) for a in run_angles], dtype=np.int64)
        payload[f"angles__{sanitized}"] = all_angles
        payload[f"bases__{sanitized}"] = all_bases
        payload[f"runs__{sanitized}"] = run_lengths

    np.savez(cache_path, **payload)
    logger.info("Cached %d chains to %s", len(chains), cache_path)
    return chains
