"""Autonomously sweep one hyperparameter at a time and record the results.

Walks a fixed list of sweeps. For each sweep:

1. Runs ``scripts/train_and_save_predictions_demo.py`` once per candidate
   value, holding the current champion config fixed for every other knob.
2. Parses the per-epoch log to extract the best val_mae.
3. Picks the value with the lowest val_mae (or the minimum across runs that
   completed at all).
4. Appends a results block to ``report/rna_sweep_notes.md``.
5. If the new winner beats the current champion by ``IMPROVEMENT_MARGIN``
   rad, update the champion and use that value for every subsequent sweep.

The champion is also persisted to ``/tmp/rna_auto_sweep_state.json`` so
that the orchestrator is resumable -- rerunning after a crash skips any
sweeps that already have a ``[sweep_name]`` block in the notes file.

Run in the background and walk away::

    uv run python scripts/auto_sweep.py > /tmp/rna_auto_sweep.log 2>&1 &
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTES_PATH = PROJECT_ROOT / "report" / "rna_sweep_notes.md"
STATE_PATH = Path("/tmp/rna_auto_sweep_state.json")
LOG_DIR = Path("/tmp/rna_auto_sweep")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Accept a new value only if it beats the champion by this margin.
IMPROVEMENT_MARGIN = 0.003

# Initial champion config. Everything after the activation sweep -- which is
# still in flight when this script starts -- will be layered on top.
CHAMPION: dict[str, str] = {
    "RNA_EPOCHS": "8",
    "RNA_LR": "3e-4",
    "RNA_BATCH_SIZE": "64",
    "RNA_HIDDEN_DIM": "256",
    "RNA_CTX_DIM": "128",
    "RNA_N_STEPS": "20",
    "RNA_DT": "0.125",
    "RNA_DIFFUSION_SCALE": "0.1",
    "RNA_CONTEXT_LENGTH": "20",
    "RNA_FUTURE_BASES_WINDOW": "0",
    "RNA_ACTIVATION": "relu",
    "RNA_DRIFT_DEPTH": "3",
    "RNA_DIFFUSION_DEPTH": "2",
    "RNA_MAX_CHAINS": "1000",
    "RNA_WARMUP": "1000",
    "RNA_ENERGY_SAMPLES": "4",
    "RNA_END_LR_FRAC": "0.05",
}

# List of (sweep name, env var, candidate values). The current champion
# value is automatically included in each sweep so comparisons are
# head-to-head.
SWEEPS: list[tuple[str, str, list[str]]] = [
    ("drift_depth", "RNA_DRIFT_DEPTH", ["2", "3", "4", "5"]),
    ("diffusion_depth", "RNA_DIFFUSION_DEPTH", ["1", "2", "3"]),
    ("hidden_dim", "RNA_HIDDEN_DIM", ["128", "192", "256", "384"]),
    ("ctx_dim", "RNA_CTX_DIM", ["64", "128", "192"]),
    ("diffusion_scale", "RNA_DIFFUSION_SCALE", ["0.05", "0.1", "0.2", "0.3"]),
    ("warmup_steps", "RNA_WARMUP", ["500", "1000", "2000", "3000"]),
    ("n_steps", "RNA_N_STEPS", ["10", "20", "30"]),
    ("energy_samples", "RNA_ENERGY_SAMPLES", ["4", "8"]),
]


def run_one(env_overrides: dict[str, str], log_path: Path) -> tuple[float | None, str]:
    env = os.environ.copy()
    env.update(CHAMPION)
    env.update(env_overrides)
    t0 = time.time()
    try:
        with log_path.open("w") as f:
            subprocess.run(
                ["uv", "run", "python", "scripts/train_and_save_predictions_demo.py"],
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=PROJECT_ROOT,
                check=False,
            )
    except Exception as exc:  # noqa: BLE001
        return None, f"launch error: {exc}"
    elapsed = time.time() - t0
    text = log_path.read_text()
    matches = re.findall(r"best_val=([\d.]+)", text)
    if not matches:
        return None, f"no epoch output after {elapsed:.0f}s (likely OOM or crash)"
    best = min(float(m) for m in matches)
    if best > 1.3:  # clearly diverged — wrapped MAE should be ~< 0.7
        return best, f"diverged (best {best:.3f}) in {elapsed:.0f}s"
    return best, f"best={best:.4f} ({elapsed:.0f}s)"


def already_done(sweep_name: str) -> bool:
    if not NOTES_PATH.exists():
        return False
    text = NOTES_PATH.read_text()
    return f"### auto: {sweep_name}" in text


def append_block(sweep_name: str, env_var: str, results: list[tuple[str, float | None, str]], winner: str, champion_before: str, champion_after: str) -> None:
    lines = [
        f"\n### auto: {sweep_name}\n",
        f"Varying `{env_var}`; every other knob held at the current champion.\n",
        "\n",
        "| value | best val_mae | note |\n",
        "|---|---|---|\n",
    ]
    for value, best, note in results:
        marker = " **winner**" if value == winner else ""
        best_str = f"{best:.4f}" if best is not None else "—"
        lines.append(f"| {value}{marker} | {best_str} | {note} |\n")
    if champion_after != champion_before:
        lines.append(f"\n**Champion updated**: `{env_var}` {champion_before} → {champion_after}\n")
    else:
        lines.append(f"\n**Champion unchanged** (`{env_var}` stays at {champion_before}).\n")
    with NOTES_PATH.open("a") as f:
        f.writelines(lines)


def save_state() -> None:
    STATE_PATH.write_text(json.dumps(CHAMPION, indent=2, sort_keys=True) + "\n")


def main() -> int:
    print(f"auto_sweep: starting with champion = {CHAMPION}", flush=True)
    save_state()
    for sweep_name, env_var, values in SWEEPS:
        if already_done(sweep_name):
            print(f"auto_sweep: skipping {sweep_name} (already recorded)", flush=True)
            continue
        print(f"\n=== {sweep_name}: sweeping {env_var} over {values} ===", flush=True)
        # Always include current champion value for head-to-head comparison.
        champion_value = CHAMPION[env_var]
        to_try = list(dict.fromkeys([champion_value, *values]))  # dedupe, keep order
        results: list[tuple[str, float | None, str]] = []
        for value in to_try:
            tag = value.replace(".", "p").replace("-", "m")
            log_path = LOG_DIR / f"{sweep_name}_{tag}.log"
            print(f"  [{sweep_name}] {env_var}={value} ... ", flush=True)
            best, note = run_one({env_var: value}, log_path)
            results.append((value, best, note))
            print(f"    -> {note}", flush=True)
        # Pick the winner (lowest best val_mae among non-None results).
        ranked = sorted(
            [(v, b, n) for v, b, n in results if b is not None],
            key=lambda r: r[1],
        )
        if not ranked:
            print(f"  [{sweep_name}] no usable results; leaving champion alone", flush=True)
            append_block(sweep_name, env_var, results, winner=champion_value, champion_before=champion_value, champion_after=champion_value)
            save_state()
            continue
        winner_value, winner_best, _ = ranked[0]
        # Only accept if it beats current champion by IMPROVEMENT_MARGIN.
        champ_score = next((b for v, b, _ in results if v == champion_value and b is not None), None)
        if champ_score is None:
            champion_after = winner_value
        elif winner_best + IMPROVEMENT_MARGIN < champ_score:
            champion_after = winner_value
        else:
            champion_after = champion_value
        append_block(
            sweep_name,
            env_var,
            results,
            winner=winner_value,
            champion_before=champion_value,
            champion_after=champion_after,
        )
        CHAMPION[env_var] = champion_after
        save_state()
        print(f"  [{sweep_name}] champion {env_var}: {champion_value} -> {champion_after}", flush=True)
    # Final summary block.
    with NOTES_PATH.open("a") as f:
        f.write("\n### Final champion (auto_sweep)\n\n")
        for k, v in sorted(CHAMPION.items()):
            f.write(f"- `{k}` = `{v}`\n")
    print("\nauto_sweep: done", flush=True)
    print(json.dumps(CHAMPION, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
