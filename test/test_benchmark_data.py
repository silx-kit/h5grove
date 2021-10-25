"""Benchmark data requests with server apps in example/ folder"""
import pathlib
from typing import Generator
from urllib.parse import urlencode
import h5py
import numpy as np
import pytest

# Benchmark conditions
BENCHMARK_FORMAT = "json", "npy", "bin"
BENCHMARKS = dict(
    (f"/{size}square_{dtype}", (size, dtype))
    for dtype in ("float32",)
    for size in (64, 128, 256, 512, 1024)
)


@pytest.fixture(scope="module")
def h5filepath(subprocess_server) -> Generator[pathlib.Path, None, None]:
    """Fixture providing HDF5 file for benchmark.

    Provides pathlib.Path
    """
    filepath = subprocess_server.served_directory / "test.h5"

    with h5py.File(filepath, mode="w") as h5file:
        for name, (size, dtype) in BENCHMARKS.items():
            h5file[name] = np.random.random((size, size)).astype(dtype)

    yield filepath


@pytest.mark.parametrize("format", BENCHMARK_FORMAT)
@pytest.mark.parametrize("h5path", tuple(BENCHMARKS.keys()))
def test_benchmark_data(h5filepath, subprocess_server, benchmark, h5path, format):
    """/data/ benchmark data access"""
    subprocess_server.get(
        f"/data/?{urlencode({'file': h5filepath.name, 'path': h5path, 'format': format})}",
        benchmark,
    )
