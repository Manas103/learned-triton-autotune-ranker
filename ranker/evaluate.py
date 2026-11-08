"""Leave-one-shape-out evaluation. For each of the 12 shapes: train the
ranker on the other 11 shapes' exhaustive measurements (ground truth,
read from docs/exhaustive_sweep.json), then for the held-out shape,
actually benchmark on the GPU, for real, exactly TOTAL_BUDGET configs
(the 8 fixed probes plus the top TOTAL_BUDGET-8 the calibrated ranker
picks) -- this is the 6% of the space this project claims to benchmark,
and it is a real second GPU measurement, not a lookup into the
exhaustive table, so the wall-clock number below is not the arithmetic
identity 512/31 but whatever this process actually took.

The exhaustive table is still used for one thing the ranker is honestly
not allowed to avoid: the true best time, against which regret is
measured, which by definition requires having searched the whole space
at least once.

Run from the WSL venv: python -m ranker.evaluate
"""
import json
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from ranker.bench import make_inputs, time_config  # noqa: E402
from ranker.config_space import CONFIGS, PROBE_CONFIGS, SHAPES  # noqa: E402
from ranker import model as model_mod  # noqa: E402

DOCS = pathlib.Path(__file__).resolve().parents[1] / "docs"
TOTAL_BUDGET = 31  # 8 probes + 23 ranker picks; 31/512 = 6.05%


def _shape_key(shape):
    return f"{shape[0]}x{shape[1]}x{shape[2]}"


def load_exhaustive():
    data = json.loads((DOCS / "exhaustive_sweep.json").read_text())
    shapes_times = {}
    exhaustive_wall_clock = {}
    true_best = {}
    for shape in SHAPES:
        rec = data["shapes"][_shape_key(shape)]
        times = {k: v["elapsed_s"] for k, v in rec["times"].items()}
        shapes_times[shape] = times
        exhaustive_wall_clock[shape] = rec["wall_clock_s"]
        true_best[shape] = rec["best_time_s"]
    return shapes_times, exhaustive_wall_clock, true_best


def main():
    shapes_times, exhaustive_wall_clock, true_best = load_exhaustive()
    results = []

    for held_out in SHAPES:
        train_data = {s: t for s, t in shapes_times.items() if s != held_out}
        model = model_mod.train(train_data)

        a, b = make_inputs(*held_out)
        t0 = time.perf_counter()

        probe_times = {}
        for cfg in PROBE_CONFIGS:
            elapsed, err = time_config(cfg, a, b)
            probe_times[str(cfg.key())] = elapsed

        ca, cb = model_mod.calibrate(model, held_out, probe_times)
        ranked = model_mod.rank_all_configs(model, held_out, ca, cb)

        picks = [c for c in ranked if c not in PROBE_CONFIGS][: TOTAL_BUDGET - len(PROBE_CONFIGS)]
        measured = dict(probe_times)
        for cfg in picks:
            elapsed, err = time_config(cfg, a, b)
            measured[str(cfg.key())] = elapsed

        wall_clock_ours = time.perf_counter() - t0
        valid_measured = [v for v in measured.values() if v is not None]
        found_best = min(valid_measured) if valid_measured else None

        best = true_best[held_out]
        regret = found_best / best if (found_best and best) else None
        pct_of_best = 100.0 * best / found_best if (found_best and best) else None
        wall_clock_cut = exhaustive_wall_clock[held_out] / wall_clock_ours if wall_clock_ours > 0 else None

        n_measured = len(measured)
        rec = {
            "shape": held_out,
            "n_configs_measured": n_measured,
            "pct_of_space_measured": round(100.0 * n_measured / len(CONFIGS), 2),
            "true_best_time_s": best,
            "found_best_time_s": found_best,
            "regret_ratio": regret,
            "pct_of_exhaustive_best": pct_of_best,
            "exhaustive_wall_clock_s": exhaustive_wall_clock[held_out],
            "ours_wall_clock_s": wall_clock_ours,
            "wall_clock_cut_x": wall_clock_cut,
        }
        results.append(rec)
        print(f"{_shape_key(held_out)}: {pct_of_best:.1f}% of best, "
              f"{rec['pct_of_space_measured']}% of space, {wall_clock_cut:.1f}x wall-clock cut")

    regrets = [r["pct_of_exhaustive_best"] for r in results]
    cuts = [r["wall_clock_cut_x"] for r in results]
    summary = {
        "total_budget_per_shape": TOTAL_BUDGET,
        "pct_of_space_measured": round(100.0 * TOTAL_BUDGET / len(CONFIGS), 2),
        "mean_pct_of_exhaustive_best": sum(regrets) / len(regrets),
        "median_pct_of_exhaustive_best": sorted(regrets)[len(regrets) // 2],
        "min_pct_of_exhaustive_best": min(regrets),
        "max_pct_of_exhaustive_best": max(regrets),
        "regret_distribution_pct": sorted(round(100 - r, 2) for r in regrets),
        "mean_wall_clock_cut_x": sum(cuts) / len(cuts),
        "median_wall_clock_cut_x": sorted(cuts)[len(cuts) // 2],
        "per_shape": results,
    }
    (DOCS / "evaluation_result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_shape"}, indent=2))


if __name__ == "__main__":
    main()
