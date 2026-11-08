"""Static tile features: everything here is computed from a config and a
shape with arithmetic, no GPU and no measurement. This is deliberately
the part of the ranker's input that generalizes across shapes it has
never seen; the 8 measured probes (see model.py) are the part that does
not generalize and has to be paid for on every new shape."""
import math

FEATURE_NAMES = [
    "block_m", "block_n", "block_k", "num_warps", "num_stages",
    "log2_block_m", "log2_block_n", "log2_block_k",
    "threads_per_block", "tile_area", "m_tiles", "n_tiles", "k_tiles", "total_tiles",
    "shape_m", "shape_n", "shape_k", "m_over_n", "k_over_m", "k_over_n",
    "bytes_per_tile_iter", "arithmetic_intensity",
]


def extract(cfg, shape):
    M, K, N = shape
    threads = cfg.num_warps * 32
    tile_area = cfg.block_m * cfg.block_n
    m_tiles = math.ceil(M / cfg.block_m)
    n_tiles = math.ceil(N / cfg.block_n)
    k_tiles = math.ceil(K / cfg.block_k)
    bytes_per_iter = 2 * (cfg.block_m * cfg.block_k + cfg.block_k * cfg.block_n)  # fp16
    flops_per_iter = 2 * cfg.block_m * cfg.block_n * cfg.block_k
    ai = flops_per_iter / bytes_per_iter
    return [
        cfg.block_m, cfg.block_n, cfg.block_k, cfg.num_warps, cfg.num_stages,
        math.log2(cfg.block_m), math.log2(cfg.block_n), math.log2(cfg.block_k),
        threads, tile_area, m_tiles, n_tiles, k_tiles, m_tiles * n_tiles,
        M, N, K, M / N, K / M, K / N,
        bytes_per_iter, ai,
    ]
