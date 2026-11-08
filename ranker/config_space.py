"""The 512-configuration Triton GEMM autotuning space this project ranks.
4 BLOCK_M x 4 BLOCK_N x 4 BLOCK_K x 4 num_warps x 2 num_stages = 512, fixed
and enumerated once so every shape is searched over the exact same space."""
from dataclasses import dataclass

BLOCK_M_CHOICES = (32, 64, 128, 256)
BLOCK_N_CHOICES = (32, 64, 128, 256)
BLOCK_K_CHOICES = (16, 32, 64, 128)
NUM_WARPS_CHOICES = (2, 4, 8, 16)
NUM_STAGES_CHOICES = (2, 4)


@dataclass(frozen=True)
class Config:
    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int

    def key(self):
        return (self.block_m, self.block_n, self.block_k, self.num_warps, self.num_stages)


def all_configs():
    configs = []
    for bm in BLOCK_M_CHOICES:
        for bn in BLOCK_N_CHOICES:
            for bk in BLOCK_K_CHOICES:
                for nw in NUM_WARPS_CHOICES:
                    for ns in NUM_STAGES_CHOICES:
                        configs.append(Config(bm, bn, bk, nw, ns))
    return configs


CONFIGS = all_configs()
assert len(CONFIGS) == 512, f"expected 512 configs, got {len(CONFIGS)}"

# 8 fixed probe configs: the 4 "corner" configs of the space plus its
# center, a small-tile config and a large-tile config, chosen once before
# any shape is seen (not cherry-picked per shape) so the probe set is the
# same real measurement cost on every shape.
PROBE_CONFIGS = [
    Config(32, 32, 16, 2, 2),
    Config(256, 256, 128, 16, 4),
    Config(32, 256, 128, 2, 4),
    Config(256, 32, 16, 16, 2),
    Config(128, 128, 64, 8, 2),
    Config(64, 64, 32, 4, 2),
    Config(128, 256, 32, 8, 4),
    Config(256, 128, 64, 4, 2),
]
assert len(PROBE_CONFIGS) == 8
assert all(c in CONFIGS for c in PROBE_CONFIGS)

SHAPES = [
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (512, 2048, 512),
    (2048, 512, 512),
    (1024, 4096, 256),
    (256, 4096, 1024),
    (2048, 2048, 256),
    (256, 2048, 2048),
    (512, 512, 4096),
    (4096, 512, 512),
]
assert len(SHAPES) == 12
