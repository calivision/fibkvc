# Copyright (c) 2026 California Vision, Inc. - David Anderson
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
3D lattice extension for fibkvc.

This module provides 3D spatial hashing capabilities using the plastic constant
(3D generalization of the golden ratio) for efficient spatial caching in diffusion
models and other 3D applications.

Features:
- 3D Fibonacci hashing with spatial locality preservation
- FibonacciLatticeCache for 3D coordinate-based caching
- Octree spatial indexing for efficient range queries
- Optional multi-resolution hierarchical caching

Usage:
    from fibkvc.lattice import FibonacciLatticeCache, fibonacci_hash_3d
    
    # Create a 3D cache
    cache = FibonacciLatticeCache(dimensions=(100, 100, 100))
    
    # Store and retrieve values
    cache.set((10, 20, 30), "value")
    value = cache.get((10, 20, 30))
    
    # Range queries
    results = cache.get_range(center=(50, 50, 50), radius=10.0)
"""

# Note: Imports will be added as components are implemented
from .fibonacci_hash_3d import fibonacci_hash_3d, PHI_3, PHI_1, PHI_2
from .lattice_cache import FibonacciLatticeCache
from .octree import Octree, OctreeNode

__all__ = [
    # Will be populated as components are implemented
    "fibonacci_hash_3d",
    "PHI_3",
    "PHI_1",
    "PHI_2",
    "FibonacciLatticeCache",
    "Octree",
    "OctreeNode",
]

# Optional hierarchy support
try:
    from .hierarchy import HierarchicalLatticeCache
    __all__.append("HierarchicalLatticeCache")
except ImportError:
    pass
