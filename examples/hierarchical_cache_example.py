#!/usr/bin/env python3
# Copyright (c) 2026 California Vision, Inc. - David Anderson
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Example demonstrating HierarchicalLatticeCache for multi-resolution caching.

This example shows how to use the hierarchical cache for coarse-to-fine
algorithms, commonly used in diffusion models and other multi-scale applications.
"""

from fibkvc.lattice import HierarchicalLatticeCache


def main():
    print("=== HierarchicalLatticeCache Example ===\n")
    
    # Create a hierarchical cache with 3 resolution levels
    # Level 0: 128x128x128 (full resolution)
    # Level 1: 64x64x64 (half resolution)
    # Level 2: 32x32x32 (quarter resolution)
    cache = HierarchicalLatticeCache(
        dimensions=(128, 128, 128),
        num_levels=3,
        use_octree=False  # Disable octree for simplicity
    )
    
    print(f"Created hierarchical cache with {cache.num_levels} levels")
    print(f"Base dimensions: {cache.base_dimensions}\n")
    
    # Display dimensions for each level
    print("Level dimensions:")
    for level in range(cache.num_levels):
        dims = cache.get_level_dimensions(level)
        print(f"  Level {level}: {dims}")
    print()
    
    # Store values at different resolution levels
    print("Storing values at different levels:")
    
    # Store at full resolution (level 0)
    cache.set((64, 64, 64), "fine_detail", level=0)
    print(f"  Level 0 (full): stored 'fine_detail' at (64, 64, 64)")
    
    # Store at half resolution (level 1)
    # Coordinates (64, 64, 64) map to (32, 32, 32) at level 1
    cache.set((64, 64, 64), "medium_detail", level=1)
    print(f"  Level 1 (half): stored 'medium_detail' at (64, 64, 64)")
    
    # Store at quarter resolution (level 2)
    # Coordinates (64, 64, 64) map to (16, 16, 16) at level 2
    cache.set((64, 64, 64), "coarse_detail", level=2)
    print(f"  Level 2 (quarter): stored 'coarse_detail' at (64, 64, 64)")
    print()
    
    # Retrieve values from different levels
    print("Retrieving values from different levels:")
    for level in range(cache.num_levels):
        value = cache.get_hierarchical((64, 64, 64), level=level)
        print(f"  Level {level}: {value}")
    print()
    
    # Demonstrate coordinate scaling
    print("Coordinate scaling demonstration:")
    print("  At level 1 (half resolution), coordinates are scaled by 2")
    print("  So (64, 64, 64) and (65, 65, 65) both map to (32, 32, 32)")
    
    # Store at (64, 64, 64) level 1
    cache.set((64, 64, 64), "value_a", level=1)
    print(f"  Stored 'value_a' at (64, 64, 64), level 1")
    
    # Retrieve at (65, 65, 65) level 1 - should get same value
    value = cache.get_hierarchical((65, 65, 65), level=1)
    print(f"  Retrieved at (65, 65, 65), level 1: {value}")
    print()
    
    # Coarse-to-fine workflow example
    print("Coarse-to-fine workflow example:")
    print("  1. Process at coarse level (level 2)")
    cache.set((64, 64, 64), "coarse_result", level=2)
    print(f"     Stored coarse result at level 2")
    
    print("  2. Refine at medium level (level 1)")
    coarse = cache.get_hierarchical((64, 64, 64), level=2)
    refined = f"refined_{coarse}"
    cache.set((64, 64, 64), refined, level=1)
    print(f"     Refined to '{refined}' at level 1")
    
    print("  3. Final refinement at full resolution (level 0)")
    medium = cache.get_hierarchical((64, 64, 64), level=1)
    final = f"final_{medium}"
    cache.set((64, 64, 64), final, level=0)
    print(f"     Final result '{final}' at level 0")
    print()
    
    # Display statistics for each level
    print("Cache statistics per level:")
    for level in range(cache.num_levels):
        stats = cache.get_level_stats(level)
        print(f"  Level {level}:")
        print(f"    Load factor: {stats['load_factor']:.6f}")
        print(f"    Total collisions: {stats['total_collisions']}")
    print()
    
    print("=== Example Complete ===")


if __name__ == "__main__":
    main()
