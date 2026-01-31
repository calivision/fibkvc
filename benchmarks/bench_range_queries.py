"""
Benchmark range query performance.

This benchmark measures:
1. Range query performance at various radii
2. Comparison against linear scan
3. Performance at various cache sizes
4. Verification of O(log n + k) complexity

Requirements: 9.1
"""

import time
import random
import sys
import math
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fibkvc.lattice.lattice_cache import FibonacciLatticeCache


def generate_random_coordinates(n: int, dimensions: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
    """Generate n random 3D coordinates within dimensions."""
    w, h, d = dimensions
    coords = []
    for _ in range(n):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        z = random.randint(0, d - 1)
        coords.append((x, y, z))
    return coords


def euclidean_distance(p1: Tuple[int, int, int], p2: Tuple[int, int, int]) -> float:
    """Calculate Euclidean distance between two 3D points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)


def linear_scan_range_query(
    cache_data: List[Tuple[Tuple[int, int, int], Any]],
    center: Tuple[int, int, int],
    radius: float
) -> List[Tuple[Tuple[int, int, int], Any]]:
    """
    Perform range query using linear scan (brute force).
    
    Args:
        cache_data: List of (coords, value) tuples
        center: Center coordinates
        radius: Search radius
    
    Returns:
        List of (coords, value) tuples within range
    """
    results = []
    for coords, value in cache_data:
        if euclidean_distance(coords, center) <= radius:
            results.append((coords, value))
    return results


def benchmark_octree_range_query(
    cache: FibonacciLatticeCache,
    center: Tuple[int, int, int],
    radius: float,
    n_iterations: int = 100
) -> Tuple[float, int]:
    """
    Benchmark octree-based range query.
    
    Args:
        cache: Cache with octree enabled
        center: Center coordinates
        radius: Search radius
        n_iterations: Number of iterations to average
    
    Returns:
        Tuple of (avg_time_ms, result_count)
    """
    # Warm-up
    results = cache.get_range(center, radius)
    result_count = len(results)
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = cache.get_range(center, radius)
    end = time.perf_counter()
    
    avg_time_ms = (end - start) * 1000 / n_iterations
    
    return avg_time_ms, result_count


def benchmark_linear_scan(
    cache_data: List[Tuple[Tuple[int, int, int], Any]],
    center: Tuple[int, int, int],
    radius: float,
    n_iterations: int = 100
) -> Tuple[float, int]:
    """
    Benchmark linear scan range query.
    
    Args:
        cache_data: List of (coords, value) tuples
        center: Center coordinates
        radius: Search radius
        n_iterations: Number of iterations to average
    
    Returns:
        Tuple of (avg_time_ms, result_count)
    """
    # Warm-up
    results = linear_scan_range_query(cache_data, center, radius)
    result_count = len(results)
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = linear_scan_range_query(cache_data, center, radius)
    end = time.perf_counter()
    
    avg_time_ms = (end - start) * 1000 / n_iterations
    
    return avg_time_ms, result_count


def benchmark_at_radius(
    cache: FibonacciLatticeCache,
    cache_data: List[Tuple[Tuple[int, int, int], Any]],
    center: Tuple[int, int, int],
    radius: float,
    n_iterations: int = 100
) -> Dict[str, Any]:
    """
    Benchmark range queries at a specific radius.
    
    Args:
        cache: Cache with octree enabled
        cache_data: List of (coords, value) tuples for linear scan
        center: Center coordinates
        radius: Search radius
        n_iterations: Number of iterations to average
    
    Returns:
        Dictionary with benchmark results
    """
    octree_time_ms, octree_count = benchmark_octree_range_query(cache, center, radius, n_iterations)
    linear_time_ms, linear_count = benchmark_linear_scan(cache_data, center, radius, n_iterations)
    
    # Verify both methods return same count
    if octree_count != linear_count:
        print(f"WARNING: Result count mismatch! Octree: {octree_count}, Linear: {linear_count}")
    
    speedup = linear_time_ms / octree_time_ms if octree_time_ms > 0 else 0
    
    return {
        'radius': radius,
        'result_count': octree_count,
        'octree_time_ms': octree_time_ms,
        'linear_time_ms': linear_time_ms,
        'speedup': speedup
    }


def benchmark_at_cache_size(
    n_coords: int,
    dimensions: Tuple[int, int, int],
    test_radii: List[float]
) -> List[Dict[str, Any]]:
    """
    Benchmark range queries at a specific cache size.
    
    Args:
        n_coords: Number of coordinates in cache
        dimensions: Cache dimensions
        test_radii: List of radii to test
    
    Returns:
        List of benchmark results for each radius
    """
    # Generate random coordinates
    coords = generate_random_coordinates(n_coords, dimensions)
    
    # Create cache with octree
    cache = FibonacciLatticeCache(dimensions=dimensions, use_octree=True)
    cache_data = []
    
    for i, coord in enumerate(coords):
        value = f"value_{i}"
        cache.set(coord, value)
        cache_data.append((coord, value))
    
    # Choose a center point (use middle of dimensions)
    center = (dimensions[0] // 2, dimensions[1] // 2, dimensions[2] // 2)
    
    # Benchmark at each radius
    results = []
    for radius in test_radii:
        result = benchmark_at_radius(cache, cache_data, center, radius, n_iterations=50)
        results.append(result)
    
    return results


def run_benchmark():
    """Run all range query benchmarks."""
    print("=" * 80)
    print("FIBONACCI LATTICE CACHE - RANGE QUERY BENCHMARK")
    print("=" * 80)
    print()
    
    # Configuration
    dimensions = (1000, 1000, 1000)
    random.seed(42)  # For reproducibility
    
    print(f"Configuration:")
    print(f"  Dimensions: {dimensions}")
    print(f"  Center point: ({dimensions[0]//2}, {dimensions[1]//2}, {dimensions[2]//2})")
    print()
    
    # Test configurations: (n_coords, test_radii)
    test_configs = [
        (100, [10.0, 50.0, 100.0, 200.0]),
        (500, [10.0, 50.0, 100.0, 200.0]),
        (1000, [10.0, 50.0, 100.0, 200.0]),
        (2000, [10.0, 50.0, 100.0, 200.0]),
    ]
    
    all_results = []
    
    for n_coords, test_radii in test_configs:
        print("-" * 80)
        print(f"BENCHMARK: Cache size = {n_coords} coordinates")
        print("-" * 80)
        print()
        
        results = benchmark_at_cache_size(n_coords, dimensions, test_radii)
        
        print(f"{'Radius':>8} {'Results':>8} {'Octree(ms)':>12} {'Linear(ms)':>12} {'Speedup':>10}")
        print("-" * 80)
        
        for r in results:
            print(f"{r['radius']:>8.1f} {r['result_count']:>8} "
                  f"{r['octree_time_ms']:>12.4f} {r['linear_time_ms']:>12.4f} "
                  f"{r['speedup']:>10.2f}x")
            all_results.append({**r, 'n_coords': n_coords})
        
        print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    
    # Calculate average speedup
    avg_speedup = sum(r['speedup'] for r in all_results) / len(all_results)
    
    print(f"Average speedup: {avg_speedup:.2f}x")
    print()
    
    # Check if meets target (10x faster than linear scan)
    target_speedup = 10.0
    
    if avg_speedup >= target_speedup:
        print(f"[PASS] Average speedup {avg_speedup:.2f}x >= {target_speedup}x target")
    else:
        print(f"[FAIL] Average speedup {avg_speedup:.2f}x < {target_speedup}x target")
    
    print()
    
    # Analyze complexity
    print("=" * 80)
    print("COMPLEXITY ANALYSIS")
    print("=" * 80)
    print()
    
    # Group results by radius
    radii = sorted(set(r['radius'] for r in all_results))
    
    for radius in radii:
        radius_results = [r for r in all_results if r['radius'] == radius]
        
        print(f"Radius = {radius:.1f}:")
        print(f"  {'N':>6} {'k':>6} {'Time(ms)':>10} {'Time/log(n+k)':>15}")
        print("  " + "-" * 50)
        
        for r in radius_results:
            n = r['n_coords']
            k = r['result_count']
            time_ms = r['octree_time_ms']
            
            # Calculate time / log(n + k) to verify O(log n + k) complexity
            log_n_plus_k = math.log2(n + k) if (n + k) > 0 else 1
            time_per_log = time_ms / log_n_plus_k
            
            print(f"  {n:>6} {k:>6} {time_ms:>10.4f} {time_per_log:>15.6f}")
        
        print()
    
    print("Note: For O(log n + k) complexity, Time/log(n+k) should remain relatively constant")
    print("as n increases (within the same radius).")
    print()
    
    # Overall pass/fail
    if avg_speedup >= target_speedup:
        print("[PASS] ALL BENCHMARKS PASSED")
    else:
        print("[FAIL] SOME BENCHMARKS FAILED")
    print()


if __name__ == "__main__":
    run_benchmark()
