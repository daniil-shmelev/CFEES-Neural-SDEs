"""Peak-GPU-memory probe using XLA's device memory stats.

``peak_bytes_in_use`` is monotonic over process lifetime, so honest per-cell
numbers require subprocess isolation. Each sweep cell runs in a fresh process.
"""

from __future__ import annotations

from typing import Callable

import jax


def read_peak_memory_bytes() -> int:
    """Peak GPU bytes allocated by XLA since process start (0 if unavailable)."""
    try:
        stats = jax.devices()[0].memory_stats()
    except Exception:  # noqa: BLE001
        return 0
    if stats is None:
        return 0
    return int(stats.get("peak_bytes_in_use", 0))


def read_bytes_limit() -> int:
    """Total device memory budget reported by XLA (0 if unavailable)."""
    try:
        stats = jax.devices()[0].memory_stats()
    except Exception:  # noqa: BLE001
        return 0
    if stats is None:
        return 0
    return int(stats.get("bytes_limit", 0))


def measure_step_peak(
    run_one_step: Callable[[], object],
    *,
    warmup: int = 3,
    repeat: int = 3,
) -> int:
    """Run warmup steps to JIT, then ``repeat`` more, return the peak bytes seen."""
    for _ in range(warmup):
        out = run_one_step()
        jax.block_until_ready(out)
    for _ in range(repeat):
        out = run_one_step()
        jax.block_until_ready(out)
    return read_peak_memory_bytes()
