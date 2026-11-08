import math

import numpy as np
import pytest
import torch

from ranker.config_space import CONFIGS, PROBE_CONFIGS, SHAPES, Config
from ranker.features import extract, FEATURE_NAMES
from ranker import model as model_mod


def test_config_space_has_512_configs():
    assert len(CONFIGS) == 512
    assert len(set(c.key() for c in CONFIGS)) == 512


def test_probes_are_8_and_in_space():
    assert len(PROBE_CONFIGS) == 8
    assert all(c in CONFIGS for c in PROBE_CONFIGS)


def test_shapes_are_12_and_distinct():
    assert len(SHAPES) == 12
    assert len(set(SHAPES)) == 12


def test_feature_extraction_is_deterministic_and_right_length():
    cfg = Config(128, 128, 64, 8, 2)
    f1 = extract(cfg, (1024, 1024, 1024))
    f2 = extract(cfg, (1024, 1024, 1024))
    assert f1 == f2
    assert len(f1) == len(FEATURE_NAMES)


def test_feature_block_dims_match_config():
    cfg = Config(64, 128, 32, 4, 2)
    f = extract(cfg, (512, 512, 512))
    assert f[FEATURE_NAMES.index("block_m")] == 64
    assert f[FEATURE_NAMES.index("block_n")] == 128
    assert f[FEATURE_NAMES.index("block_k")] == 32


def test_calibration_recovers_linear_relationship():
    class FakeModel:
        def predict(self, X):
            return np.array([0.01 * x[0] + 1.0 for x in X])

    shape = (512, 512, 512)
    true_a, true_b = 1.2, -0.5
    probe_times = {}
    for cfg in PROBE_CONFIGS:
        raw_pred = 0.01 * extract(cfg, shape)[0] + 1.0
        actual_log = true_a * raw_pred + true_b
        probe_times[str(cfg.key())] = math.exp(actual_log)

    pmed = model_mod.probe_median(probe_times)
    expected_b = true_b - math.log(pmed)
    a, b = model_mod.calibrate(FakeModel(), shape, probe_times)
    assert abs(a - true_a) < 1e-6
    assert abs(b - expected_b) < 1e-6


def test_build_training_set_penalizes_invalid_configs():
    shapes_times = {
        (256, 256, 256): {str(c.key()): (0.001 if i % 10 else None) for i, c in enumerate(CONFIGS)},
    }
    X, y = model_mod.build_training_set(shapes_times)
    assert len(X) == len(CONFIGS)
    assert len(y) == len(CONFIGS)
    # the penalized (invalid) rows must be above the valid rows' label
    valid_y = [y[i] for i, c in enumerate(CONFIGS) if i % 10]
    invalid_y = [y[i] for i, c in enumerate(CONFIGS) if not i % 10]
    assert min(invalid_y) > max(valid_y)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gemm_kernel_matches_fp64_oracle():
    from ranker.gemm_kernel import run_matmul
    g = torch.Generator(device="cpu").manual_seed(7)
    a16 = (torch.rand((96, 80), generator=g) * 2 - 1).to("cuda", torch.float16)
    b16 = (torch.rand((80, 112), generator=g) * 2 - 1).to("cuda", torch.float16)
    cfg = Config(32, 32, 16, 4, 2)
    c = run_matmul(a16, b16, cfg)
    oracle = (a16.double() @ b16.double())
    diff = (c.double() - oracle).abs().max().item()
    # fp16 GEMM over K=80 with SAFETY=4 style budget (see agent/tolerance.py sibling repo)
    assert diff < 4.0 * 80 * 2 ** -11 * (oracle.abs().mean().item() + 1.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gemm_kernel_handles_non_divisible_shapes():
    from ranker.gemm_kernel import run_matmul
    g = torch.Generator(device="cpu").manual_seed(3)
    a16 = (torch.rand((70, 50), generator=g) * 2 - 1).to("cuda", torch.float16)
    b16 = (torch.rand((50, 90), generator=g) * 2 - 1).to("cuda", torch.float16)
    cfg = Config(32, 32, 32, 2, 2)
    c = run_matmul(a16, b16, cfg)
    assert c.shape == (70, 90)
