# Copyright (c) 2026 California Vision, Inc. - David Anderson
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Fibonacci KV Cache - High-performance cache optimization using Fibonacci hashing.

This package provides a drop-in replacement for standard KV cache implementations
in diffusion language models, using Fibonacci hashing (golden ratio) for improved
distribution and performance.
"""

from pathlib import Path

# Read version from VERSION file (located in opensource/ directory)
try:
    _version_file = Path(__file__).parent.parent / "VERSION"
    __version__ = _version_file.read_text(encoding="utf-8").strip()
except Exception:
    __version__ = "0.0.0"

from fibkvc.fibonacci_hash import (
    fibonacci_hash,
    is_power_of_two,
    string_to_int,
    GOLDEN_RATIO,
    GOLDEN_RATIO_64,
)

from fibkvc.fibonacci_cache import FibonacciCacheOptimizer

from fibkvc.fibonacci_config import (
    FibonacciHashingConfig,
    get_global_config,
    set_global_config,
)

__author__ = "David Anderson"
__license__ = "MIT"

__all__ = [
    # Core hashing
    "fibonacci_hash",
    "is_power_of_two",
    "string_to_int",
    "GOLDEN_RATIO",
    "GOLDEN_RATIO_64",
    
    # Cache optimizer
    "FibonacciCacheOptimizer",
    
    # Configuration
    "FibonacciHashingConfig",
    "get_global_config",
    "set_global_config",
]
