#!/usr/bin/env python3
# Copyright (c) 2026 California Vision, Inc. - David Anderson
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Range query example for Fibonacci Lattice Cache.

This example demonstrates spatial range queries using the octree-based
spatial indexing for efficient neighborhood lookups.
"""

import random
from fibkvc.lattice import FibonacciLatticeCache


def euclidean_distance(p1, p2):
    """Calculate Euclidean distance between two 3D points."""
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2) ** 0.5


def main():
    print("=" * 60)
    print("Fibonacci Lattice Cache - Range Query Example")
    print("=" * 60)
    
    # 1. Create cache with octree enabled
    print("\n1. Creating cache with octree spatial indexing...")
    dimensions = (200, 200, 200)
    cache = FibonacciLatticeCache(
        dimensions=dimensions,
        initial_size=512,
        use_octree=True  # Required for range queries
    )
    print(f"   ✓ Cache created: {dimensions}")
    print(f"   ✓ Octree enabled for spatial queries")
    
    # 2. Populate cache with random data
    print("\n2. Populating cache with random 3D points...")
    num_points = 100
    random.seed(42)  # For reproducibility
    
    points = []
    for i in range(num_points):
        x = random.randint(0, dimensions[0] - 1)
        y = random.randint(0, dimensions[1] - 1)
        z = random.randint(0, dimensions[2] - 1)
        coords = (x, y, z)
        value = f"point_{i}"
        cache.set(coords, value)
        points.append((coords, value))
    
    print(f"   ✓ Stored {num_points} random points")
    
    # 3. Perform range query with small radius
    print("\n3. Range query with small radius...")
    center = (100, 100, 100)
    radius = 20.0
    
    results = cache.get_range(center, radius)
    print(f"   Query center: {center}")
    print(f"   Query radius: {radius}")
    print(f"   Found {len(results)} points within range")
    
    # Verify results
    print("\n   Verifying results:")
    for coords, value in results[:5]:  # Show first 5
        dist = euclidean_distance(center, coords)
        print(f"   - {coords}: '{value}' (distance: {dist:.2f})")
    
    if len(results) > 5:
        print(f"   ... and {len(results) - 5} more points")
    
    # 4. Perform range query with larger radius
    print("\n4. Range query with larger radius...")
    radius = 50.0
    results = cache.get_range(center, radius)
    print(f"   Query center: {center}")
    print(f"   Query radius: {radius}")
    print(f"   Found {len(results)} points within range")
    
    # 5. Query at different locations
    print("\n5. Queries at different locations...")
    query_points = [
        ((50, 50, 50), 25.0),
        ((150, 150, 150), 30.0),
        ((10, 10, 10), 15.0),
    ]
    
    for center, radius in query_points:
        results = cache.get_range(center, radius)
        print(f"   Center {center}, radius {radius:.1f} → {len(results)} points")
    
    # 6. Empty region query
    print("\n6. Query in empty region...")
    empty_center = (5, 5, 5)
    empty_radius = 3.0
    results = cache.get_range(empty_center, empty_radius)
    print(f"   Center {empty_center}, radius {empty_radius}")
    print(f"   Found {len(results)} points (expected few or none)")
    
    # 7. Create a cluster and query it
    print("\n7. Creating a dense cluster and querying...")
    cluster_center = (180, 180, 180)
    cluster_size = 10
    
    print(f"   Creating cluster around {cluster_center}...")
    for i in range(cluster_size):
        offset_x = random.randint(-5, 5)
        offset_y = random.randint(-5, 5)
        offset_z = random.randint(-5, 5)
        coords = (
            cluster_center[0] + offset_x,
            cluster_center[1] + offset_y,
            cluster_center[2] + offset_z
        )
        # Ensure within bounds
        coords = (
            max(0, min(dimensions[0] - 1, coords[0])),
            max(0, min(dimensions[1] - 1, coords[1])),
            max(0, min(dimensions[2] - 1, coords[2]))
        )
        cache.set(coords, f"cluster_point_{i}")
    
    print(f"   ✓ Added {cluster_size} points to cluster")
    
    # Query the cluster
    cluster_results = cache.get_range(cluster_center, 10.0)
    print(f"\n   Querying cluster (radius 10.0):")
    print(f"   Found {len(cluster_results)} points")
    
    for coords, value in cluster_results[:5]:
        dist = euclidean_distance(cluster_center, coords)
        print(f"   - {coords}: '{value}' (distance: {dist:.2f})")
    
    # 8. Performance comparison
    print("\n8. Performance comparison: Octree vs Linear Scan...")
    
    # Octree query (already done above)
    test_center = (100, 100, 100)
    test_radius = 30.0
    octree_results = cache.get_range(test_center, test_radius)
    
    # Manual linear scan for comparison
    linear_results = []
    for coords, value in points:
        if euclidean_distance(test_center, coords) <= test_radius:
            linear_results.append((coords, value))
    
    print(f"   Octree found: {len(octree_results)} points")
    print(f"   Linear scan found: {len(linear_results)} points")
    
    # Verify they match
    octree_coords = set(coords for coords, _ in octree_results)
    linear_coords = set(coords for coords, _ in linear_results)
    
    if octree_coords == linear_coords:
        print("   ✓ Results match - octree is correct!")
    else:
        print("   ✗ Results differ - investigating...")
        missing = linear_coords - octree_coords
        extra = octree_coords - linear_coords
        if missing:
            print(f"   Missing from octree: {len(missing)}")
        if extra:
            print(f"   Extra in octree: {len(extra)}")
    
    # 9. Demonstrate use case: Find nearest neighbors
    print("\n9. Use case: Finding nearest neighbors...")
    target = (100, 100, 100)
    
    # Start with small radius and expand if needed
    for radius in [10.0, 20.0, 30.0, 40.0]:
        neighbors = cache.get_range(target, radius)
        if len(neighbors) >= 5:
            print(f"   Found {len(neighbors)} neighbors within radius {radius}")
            print(f"   First 5 nearest:")
            
            # Sort by distance
            neighbors_with_dist = [
                (coords, value, euclidean_distance(target, coords))
                for coords, value in neighbors
            ]
            neighbors_with_dist.sort(key=lambda x: x[2])
            
            for coords, value, dist in neighbors_with_dist[:5]:
                print(f"   - {coords}: '{value}' (distance: {dist:.2f})")
            break
    
    # 10. Cache statistics
    print("\n10. Final cache statistics:")
    stats = cache.get_collision_stats()
    print(f"   Total entries:      {cache._size}")
    print(f"   Table size:         {cache.table_size}")
    print(f"   Load factor:        {stats['load_factor']:.2%}")
    print(f"   Total collisions:   {stats['total_collisions']}")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  - Range queries use octree for O(log n + k) performance")
    print("  - Octree automatically maintained during set() operations")
    print("  - Useful for spatial applications: diffusion models, 3D graphics, etc.")


if __name__ == "__main__":
    main()
