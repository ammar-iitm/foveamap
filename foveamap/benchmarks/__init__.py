"""Benchmarking and baseline verification utilities for FoveaMap."""
from .baseline import compare_uniform_baseline, compute_grid_geometry
from .bench_mapping import run_mapping_benchmark

__all__ = ["compare_uniform_baseline", "compute_grid_geometry", "run_mapping_benchmark"]
