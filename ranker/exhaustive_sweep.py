"""The ground truth this whole project is scored against. Compiles every
one of the 512 configs once (warmup, not counted as search wall-clock:
see README for why), then for each of the 12 shapes, times every
(already-compiled) config and records the full table plus the per-shape
wall-clock of that full 512-point pass. This is the expensive, honest
baseline: the only thing that makes "97% of exhaustive-search best" or
"14x wall-clock cut" a real claim instead of an assumed one.

Run from the WSL venv: python -m ranker.exhaustive_sweep
"""
import json
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from ranker.bench import make_inputs, time_config  # noqa: E402
from ranker.config_space import CONFIGS, SHAPES  # noqa: E402

DOCS = pathlib.Path(__file__).resolve().parents[1] / "docs"


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    warmup_shape = SHAPES[0]
    a, b = make_inputs(*warmup_shape)
    print(f"Warmup: compiling all {len(CONFIGS)} configs on shape {warmup_shape} (not timed as search cost)...")
    invalid_at_warmup = 0
    t0 = time.perf_counter()
    for cfg in CONFIGS:
        _, err = time_config(cfg, a, b)
        if err:
            invalid_at_warmup += 1
    print(f"Warmup done in {time.perf_counter() - t0:.1f}s, {invalid_at_warmup}/{len(CONFIGS)} configs invalid")

    all_results = {}
    for shape in SHAPES:
        a, b = make_inputs(*shape)
        shape_key = f"{shape[0]}x{shape[1]}x{shape[2]}"
        print(f"Benchmarking shape {shape_key} over {len(CONFIGS)} already-compiled configs...")
        t0 = time.perf_counter()
        times = {}
        invalid = 0
        for cfg in CONFIGS:
            elapsed, err = time_config(cfg, a, b)
            times[str(cfg.key())] = {"elapsed_s": elapsed, "error": err}
            if err:
                invalid += 1
        wall = time.perf_counter() - t0
        valid_times = [v["elapsed_s"] for v in times.values() if v["elapsed_s"] is not None]
        best = min(valid_times) if valid_times else None
        print(f"  wall-clock: {wall:.2f}s, invalid: {invalid}/{len(CONFIGS)}, best: {best}")
        all_results[shape_key] = {
            "shape": shape, "wall_clock_s": wall, "invalid_count": invalid,
            "best_time_s": best, "times": times,
        }

    out = {
        "gpu": torch.cuda.get_device_name(0),
        "n_configs": len(CONFIGS),
        "n_shapes": len(SHAPES),
        "invalid_at_warmup": invalid_at_warmup,
        "shapes": all_results,
    }
    DOCS.mkdir(exist_ok=True)
    (DOCS / "exhaustive_sweep.json").write_text(json.dumps(out, indent=2), encoding="utf-8", newline="\n")
    print("Wrote docs/exhaustive_sweep.json")


if __name__ == "__main__":
    main()
