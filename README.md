# learned-triton-autotune-ranker

A learned ranker that prunes Triton GEMM autotuning: static tile features plus 8 real measured probes per shape are enough to pick, out of a fixed 512-configuration search space, a roughly 6%-sized subset that reaches most of the exhaustive-search best runtime, at a fraction of the wall-clock. Every number below was measured on this machine via leave-one-shape-out evaluation against a full exhaustive sweep, not targeted.

## Why this exists

Triton's own autotuner finds the best kernel configuration for a shape by actually running every candidate and keeping the fastest, which is correct but linear in the size of the search space. The question this project asks is narrower: given a brand-new GEMM shape the ranker has never seen, how much of that exhaustive search can be skipped by combining the configuration's static, shape-independent properties (tile dimensions, thread count, arithmetic intensity) with a handful of real measurements on the new shape, without ever letting the model grade its own correctness, since every number here comes from an actual GPU timing, not a model's opinion of one.

## What this is not

- **Not a general-purpose Triton autotuner.** The search space is one canonical GEMM kernel's 5 tunable parameters (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages), not an arbitrary kernel's configuration space.
- **Not a model that scores its own picks.** The ranker outputs a ranking; every number reported below (which configs were "measured", their times, the regret against the true best) comes from a real `torch.cuda.Event`-timed kernel launch on this GPU, looked up or re-run, never from the model's prediction.
- **The "true best" is itself only true for a 512-point grid.** It is the minimum over this project's fixed 512-configuration space, found once by exhaustively benchmarking all of it per shape; a continuous or larger search space could contain a faster configuration this project never considers.
- **Leave-one-shape-out, not a held-out production workload.** All 12 shapes are synthetic GEMM problems chosen to vary aspect ratio and size; the ranker is evaluated on each shape with that shape's own data excluded from training, which is the right discipline for "has this model seen this before" but is not the same claim as generalizing to an arbitrary real model's shape distribution.
- **Some of the 512 configs are not valid kernels at all** (illegal shared-memory or register requirements for a given BLOCK/num_stages combination); these fail to compile or launch and are recorded as invalid, not silently dropped or hidden from the reported counts.

## Machine and toolchain

WSL2 Ubuntu 22.04 on Windows 11, NVIDIA GeForce RTX 4070 SUPER (sm_89), CUDA 12.8 (PyTorch's bundled runtime), PyTorch 2.11.0+cu128, Triton 3.6.0, scikit-learn 1.7.2, Python 3.10.12.

## Architecture

```
ranker/
  config_space.py   the fixed 512-configuration grid, the 8 probe configs, and the 12 shapes
  gemm_kernel.py     one canonical Triton GEMM, parameterized by the 5 config fields
  bench.py           compiles, runs once, and times a (config, shape) pair; invalid configs
                      are caught and reported, not treated as a crash
  features.py        static tile features: block dims, tile/thread counts, arithmetic intensity
  model.py            RandomForest trained on 11 shapes' exhaustive data, plus the 8-probe
                      affine calibration step for the 12th
  exhaustive_sweep.py the ground truth: every config x every shape, timed, written to docs/
  evaluate.py          leave-one-shape-out: train, probe, select, re-benchmark for real, score
tests/
  test_ranker.py      config-space invariants, feature determinism, calibration recovery on
                      synthetic data, and the GEMM kernel against an fp64 oracle
docs/                  committed raw sweep and evaluation output
```

### Design deep-dives

**Why static features plus probes, not static features alone.** A model trained only on 11 shapes' static features has no way to know the 12th shape's absolute scale: GEMM runtime depends on M, N, K in a way the model can extrapolate roughly but not calibrate precisely for a shape it has never measured. The 8 probes are real, actual timings on the exact shape being ranked; fitting a 1-parameter affine correction (`actual_log_time = a * predicted_log_time + b`) against those 8 points corrects the model's shape-specific bias before it ranks the remaining 504 configs it was never allowed to measure.

**Why the probes are fixed, not chosen per shape.** Choosing probes after seeing a shape's data would be measuring the shape more than 8 times' worth and calling it 8; the 8 probe configs in `config_space.py` (four corners of the grid, the center, two small-tile and two large-tile points) are fixed once, before any shape-specific evaluation, so every shape pays exactly the same real measurement cost.

**Why leave-one-shape-out and not a train/test split within a shape.** Every config's true cost depends on the shape; splitting configs within one shape would let the model see that shape's own scale during training, which is exactly the information the 8 probes exist to supply honestly instead.

**Why wall-clock is measured twice, not computed as 512/31.** It would be circular to report "14x" as the arithmetic identity 512 configs over 31 configs; see Findings and Measured results for why this project instead times the actual 31-config selection-and-measurement process on the GPU and compares it to the actual previously-measured 512-config sweep's wall-clock, so the reported multiplier reflects real fixed overhead (model inference, config compilation state, Python dispatch) instead of assuming it away.

## Validation

1. **Config-space invariants** (`test_config_space_has_512_configs`, `test_probes_are_8_and_in_space`, `test_shapes_are_12_and_distinct`): the grid this whole project is scored against is checked to actually be what it claims to be.
2. **Feature determinism and correctness** (`test_feature_extraction_is_deterministic_and_right_length`, `test_feature_block_dims_match_config`): the static features a shape-unseen model relies on are checked against hand-computed values, not just "ran without an exception".
3. **Calibration self-check** (`test_calibration_recovers_linear_relationship`): a fake model with a known linear relationship to the "true" transfer function is calibrated against synthetic probe data, and the fitted (a, b) must recover the planted values exactly; this is the harness self-check in the same spirit as the sibling repo's injected-divergence test, and it is what would catch a bug that made calibration a no-op.
4. **Invalid-config penalty check** (`test_build_training_set_penalizes_invalid_configs`): confirms invalid configs are trained on with a worse label than every valid config, not dropped and not accidentally ranked well.
5. **GEMM correctness against an fp64 oracle** (`test_gemm_kernel_matches_fp64_oracle`, `test_gemm_kernel_handles_non_divisible_shapes`): the one kernel this whole search space tunes is diffed against a float64 PyTorch matmul using the same reduction-order tolerance budget derived in the sibling `gpu-kernel-equivalence-harness` repo, and checked on a shape that does not divide the chosen block size evenly.

Full output: `docs/test_output.txt`.

## Findings

**80 of 512 configs are invalid, on every shape, identically.** The exhaustive sweep's warmup pass (compiling all 512 configs once) found 80 configs that fail to compile or launch, and the per-shape benchmarking passes found the same 80/512 invalid on all 12 shapes without exception. Invalidity here is a property of the (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages) tuple alone (register and shared-memory limits a config either does or does not exceed), not of the shape being multiplied, which is exactly what `build_training_set`'s shape-relative penalty assumes and is now a checked fact rather than an assumption.

**Predicting absolute log(runtime) generalized leave-one-shape-out badly; normalizing by the shape's own probe median was a real, measured improvement.** Attempt 1 (the model predicting absolute log(runtime) from static features trained on 11 other shapes) reached a median of only 78.2% of each held-out shape's true best. The problem was visible in the regret numbers themselves: shapes whose absolute scale (total FLOPs) was far from the training shapes' scale regressed worst, which is exactly what "the model has never seen this shape's scale" predicts. Attempt 2 changed the training target to `log(elapsed_s / that shape's own 8-probe median)` (see `model.probe_median`), which removes most of the cross-shape scale variance before the model ever sees it, and raised the median to 88.3% of best in that evaluation run. Attempt 3 tried shallower trees (`max_depth=6, min_samples_leaf=8`) to fight a suspected overfit to the 11 training shapes' idiosyncrasies further; it made both the median regret and the wall-clock cut worse and was reverted. The code ships attempt 2's model configuration; `model.train`'s docstring carries all three numbers.

**The ranker is consistently worst on the most aspect-ratio-extreme shapes.** Across every evaluation run, the four worst-ranked held-out shapes are consistently `512x512x4096`, `4096x512x512`, `256x2048x2048` and `2048x2048x256`, the four shapes whose M:N:K ratio is most unlike the other 11's. With only 12 shapes total, 11 is not enough aspect-ratio diversity for leave-one-shape-out to generalize to the most unusual one; this is a real limitation of a 12-shape evaluation, not a bug, and is why Limitations says a deployment-scale evaluation would need far more shapes.

**Re-running the exact same shipped model configuration gives a different number each time, because the 8 probes are measured fresh on real GPU timing every run.** Two back-to-back evaluation runs of the identical attempt-2 configuration produced medians of 88.3% and 81.2% of best. This is not nondeterminism in the model (the RandomForest's `random_state` is fixed); it is the real GPU timing noise this project's own README elsewhere calls out, now visible in a number that matters. The committed `docs/evaluation_result.json` is the second of those two runs; both are real, and the honest statement is a range, not either single number alone.

## Measured results

**Machine:** same as above. Exhaustive sweep: 512 configs x 12 shapes, warmup (all-configs compile) 971.5s, per-shape benchmark wall-clock 0.76s-4.36s (already-compiled). 80/512 configs invalid on every shape.

| Claim | Measured | Meets claim |
|---|---|---|
| 512 configurations ranked per GEMM shape | **512** (80 of which are invalid on every shape, disclosed, not hidden) | Yes |
| ranking from static tile features plus 8 measured probes | **True**: see `model.py` | Yes |
| 97% of the exhaustive-search best runtime | **81.2% median** (mean 69.3%, min 18.6%, max 107.6%, across the 12 leave-one-shape-out folds); a second run of the identical configuration measured 88.3% median, see Findings | No |
| 6% of the configuration space benchmarked | **6.05%** (31 of 512: 8 probes + 23 ranker picks) | Yes |
| 14x autotune wall-clock cut across 12 shapes | **2.6x median, 3.1x mean**, measured as a real second GPU timing of the 31-config selection process against the real 512-config sweep's wall-clock, not computed as 512/31 | No |
| regret distribution reported rather than the mean | **True**: both are reported; the distribution (`docs/evaluation_result.json`'s `regret_distribution_pct`) ranges from -7.6% (the selected subset found something faster than the recorded "best", a real artifact of GPU timing noise) to 81.4% | Yes |

Per-shape percent of exhaustive best, worst to best (final committed run):

```
4096x512x512      18.6%   512x512x4096      25.1%   256x2048x2048     28.2%
2048x2048x256     61.5%   2048x512x512      66.7%   512x2048x512      69.7%
1024x4096x256     81.2%   256x4096x1024     83.7%   2048x2048x2048    93.2%
1024x1024x1024    94.5%   512x512x512      101.2%   256x256x256      107.6%
```

Full output: `docs/exhaustive_sweep_console.txt`, `docs/exhaustive_sweep.json` (the full 512x12 timing table, 805 KB, the ground truth every other number here is scored against), `docs/benchmark_output.txt`, `docs/evaluation_result.json`.

## Building and running

```
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python -m pytest tests/ -v

python -m ranker.exhaustive_sweep     # the ground truth; ~512 x 12 timed launches
python -m ranker.evaluate             # leave-one-shape-out, re-benchmarking 31/512 configs per shape
```

## Limitations

- The search space is one kernel's 5 parameters; a different kernel body (split-K, persistent kernels, different tiling altogether) is a different space this project does not search.
- 12 shapes is enough to run leave-one-shape-out, not enough to characterize a production model's actual shape distribution; a deployment-scale evaluation would need far more shapes.
- The affine calibration step is a single scale-and-offset correction, not a per-feature recalibration; a model whose error is non-uniform across the 512-point grid (worse on small tiles than large ones, say) would not be fully corrected by it.
- `features.py`'s arithmetic-intensity proxy assumes fp16 operands; it would need updating for a different dtype.
- **The percent-of-best and wall-clock-cut numbers vary meaningfully run to run** (88.3% vs 81.2% median regret on two back-to-back runs of the identical configuration), because the 8 probes are real GPU timings subject to this shared machine's normal measurement noise; a single run's number should not be read as more precise than that range.
