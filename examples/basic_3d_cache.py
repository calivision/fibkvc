#!/usr/bin/env python3
# Copyright (c) 2026 California Vision, Inc. - David Anderson
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Basic 3D cache usage example for Fibonacci Lattice Cache.

This example demonstrates the core functionality of the FibonacciLatticeCache
for storing and retrieving values indexed by 3D coordinates.
"""

from fibkvc.lattice import FibonacciLatticeCache, fibonacci_hash_3d


def main():
    print("=" * 60)
    print("Fibonacci Lattice Cache - Basic 3D Usage Example")
    print("=" * 60)
    
    # 1. Create a 3D lattice cache
    print("\n1. Creating FibonacciLatticeCache...")
    cache = FibonacciLatticeCache(
        dimensions=(100, 100, 100),  # 100x100x100 lattice
        initial_size=256,
        use_octree=True  # Enable spatial indexing
    )
    print(f"   ✓ Cache created with dimensions: {cache.dimensions}")
    print(f"   ✓ Initial table size: {cache.table_size}")
    print(f"   ✓ Octree enabled: {cache._octree is not None}")
    
    # 2. Store values at 3D coordinates
    print("\n2. Storing values at 3D coordinates...")
    test_coords = [
        ((10, 20, 30), "point_A"),
        ((50, 50, 50), "center"),
        ((90, 90, 90), "corner"),
        ((25, 25, 25), "cluster_1"),
        ((26, 26, 26), "cluster_2"),
        ((27, 27, 27), "cluster_3"),
    ]
    
    for coords, value in test_coords:
        cache.set(coords, value)
        print(f"   ✓ Stored '{value}' at {coords}")
    
    # 3. Retrieve values
    print("\n3. Retrieving values from cache...")
    for coords, expected_value in test_coords:
        retrieved = cache.get(coords)
        status = "✓" if retrieved == expected_value else "✗"
        print(f"   {status} Get {coords} → '{retrieved}'")
    
    # 4. Test non-existent coordinates
    print("\n4. Testing non-existent coordinates...")
    missing_coords = [(99, 99, 99), (0, 0, 0), (15, 15, 15)]
    for coords in missing_coords:
        result = cache.get(coords)
        print(f"   Get {coords} → {result} (expected None)")
    
    # 5. Update existing values
    print("\n5. Updating existing values...")
    cache.set((50, 50, 50), "updated_center")
    updated = cache.get((50, 50, 50))
    print(f"   ✓ Updated (50, 50, 50) → '{updated}'")
    
    # 6. Demonstrate hash function
    print("\n6. Demonstrating 3D Fibonacci hash function...")
    sample_coords = [(0, 0, 0), (10, 20, 30), (50, 50, 50), (99, 99, 99)]
    table_size = 256
    
    print(f"   Hash values for table_size={table_size}:")
    for coords in sample_coords:
        hash_val = fibonacci_hash_3d(*coords, table_size)
        print(f"   {coords} → hash index {hash_val}")
    
    # 7. Show cache statistics
    print("\n7. Cache statistics:")
    stats = cache.get_collision_stats()
    print(f"   Total entries:      {cache._size}")
    print(f"   Table size:         {cache.table_size}")
    print(f"   Load factor:        {stats['load_factor']:.2%}")
    print(f"   Total collisions:   {stats['total_collisions']}")
    print(f"   Max chain length:   {stats['max_chain_length']}")
    print(f"   Avg chain length:   {stats['avg_chain_length']:.2f}")
    
    # 8. Test coordinate validation
    print("\n8. Testing coordinate validation...")
    try:
        cache.set((150, 150, 150), "out_of_bounds")
        print("   ✗ Should have raised ValueError")
    except ValueError as e:
        print(f"   ✓ Correctly rejected out-of-bounds: {e}")
    
    try:
        cache.set((-1, 0, 0), "negative")
        print("   ✗ Should have raised ValueError")
    except ValueError as e:
        print(f"   ✓ Correctly rejected negative coords: {e}")
    
    # 9. Demonstrate spatial locality
    print("\n9. Demonstrating spatial locality...")
    print("   Nearby points tend to have nearby hash indices:")
    
    base_coord = (50, 50, 50)
    nearby_coords = [
        (50, 50, 50),
        (51, 50, 50),
        (50, 51, 50),
        (50, 50, 51),
        (51, 51, 51),
    ]
    
    base_hash = fibonacci_hash_3d(*base_coord, table_size)
    print(f"   Base {base_coord} → hash {base_hash}")
    
    for coords in nearby_coords[1:]:
        hash_val = fibonacci_hash_3d(*coords, table_size)
        distance = abs(hash_val - base_hash)
        print(f"   {coords} → hash {hash_val} (distance: {distance})")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  - Try range_queries_example.py for spatial queries")
    print("  - Try hierarchical_cache_example.py for multi-resolution")


if __name__ == "__main__":
    main()
