"""
Benchmark hash distribution uniformity and collision rates.

This benchmark measures:
1. Hash computation time
2. Collision rate for 10,000 random coordinates
3. Distribution uniformity using chi-square test
4. Comparison against Python's built-in hash()

Requirements: 9.1, 9.2
"""

import time
import random
import sys
from pathlib import Path
from typing import List, Tuple
from collections import Counter

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fibkvc.lattice.fibonacci_hash_3d import fibonacci_hash_3d


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


def benchmark_hash_computation_time(coords: List[Tuple[int, int, int]], table_size: int) -> float:
    """
    Benchmark hash computation time.
    
    Args:
        coords: List of coordinates to hash
        table_size: Hash table size
    
    Returns:
        Average time per hash in nanoseconds
    """
    start = time.perf_counter()
    for x, y, z in coords:
        fibonacci_hash_3d(x, y, z, table_size)
    end = time.perf_counter()
    
    total_time_ns = (end - start) * 1e9
    avg_time_ns = total_time_ns / len(coords)
    return avg_time_ns


def benchmark_python_hash_time(coords: List[Tuple[int, int, int]], table_size: int) -> float:
    """
    Benchmark Python's built-in hash() for comparison.
    
    Args:
        coords: List of coordinates to hash
        table_size: Hash table size
    
    Returns:
        Average time per hash in nanoseconds
    """
    start = time.perf_counter()
    for coord in coords:
        hash(coord) % table_size
    end = time.perf_counter()
    
    total_time_ns = (end - start) * 1e9
    avg_time_ns = total_time_ns / len(coords)
    return avg_time_ns


def measure_collision_rate(coords: List[Tuple[int, int, int]], table_size: int) -> float:
    """
    Measure collision rate for given coordinates.
    
    Args:
        coords: List of coordinates to hash
        table_size: Hash table size
    
    Returns:
        Collision rate as percentage (0-100)
    """
    hash_values = []
    for x, y, z in coords:
        h = fibonacci_hash_3d(x, y, z, table_size)
        hash_values.append(h)
    
    unique_hashes = len(set(hash_values))
    total_hashes = len(hash_values)
    collisions = total_hashes - unique_hashes
    collision_rate = (collisions / total_hashes) * 100
    
    return collision_rate


def measure_python_hash_collision_rate(coords: List[Tuple[int, int, int]], table_size: int) -> float:
    """
    Measure collision rate for Python's built-in hash().
    
    Args:
        coords: List of coordinates to hash
        table_size: Hash table size
    
    Returns:
        Collision rate as percentage (0-100)
    """
    hash_values = []
    for coord in coords:
        h = hash(coord) % table_size
        hash_values.append(h)
    
    unique_hashes = len(set(hash_values))
    total_hashes = len(hash_values)
    collisions = total_hashes - unique_hashes
    collision_rate = (collisions / total_hashes) * 100
    
    return collision_rate


def chi_square_test(hash_values: List[int], table_size: int) -> Tuple[float, float]:
    """
    Perform chi-square test for uniformity.
    
    Args:
        hash_values: List of hash values
        table_size: Hash table size
    
    Returns:
        Tuple of (chi_square_statistic, p_value)
        p_value > 0.05 indicates uniform distribution
    """
    # Count frequency of each hash value
    counts = Counter(hash_values)
    
    # Expected frequency for uniform distribution
    expected = len(hash_values) / table_size
    
    # Compute chi-square statistic
    chi_square = 0.0
    for i in range(table_size):
        observed = counts.get(i, 0)
        chi_square += ((observed - expected) ** 2) / expected
    
    # Degrees of freedom
    df = table_size - 1
    
    # Approximate p-value using chi-square distribution
    # For large df, chi-square ~ Normal(df, 2*df)
    # This is a rough approximation
    mean = df
    std = (2 * df) ** 0.5
    z_score = (chi_square - mean) / std
    
    # Approximate p-value (two-tailed)
    # Using complementary error function approximation
    import math
    if z_score > 0:
        p_value = 2 * (1 - 0.5 * (1 + math.erf(z_score / math.sqrt(2))))
    else:
        p_value = 2 * 0.5 * (1 + math.erf(abs(z_score) / math.sqrt(2)))
    
    return chi_square, p_value


def measure_distribution_uniformity(coords: List[Tuple[int, int, int]], table_size: int) -> Tuple[float, float]:
    """
    Measure distribution uniformity using chi-square test.
    
    Args:
        coords: List of coordinates to hash
        table_size: Hash table size
    
    Returns:
        Tuple of (chi_square_statistic, p_value)
    """
    hash_values = []
    for x, y, z in coords:
        h = fibonacci_hash_3d(x, y, z, table_size)
        hash_values.append(h)
    
    return chi_square_test(hash_values, table_size)


def run_benchmark():
    """Run all hash distribution benchmarks."""
    print("=" * 80)
    print("FIBONACCI HASH 3D - DISTRIBUTION BENCHMARK")
    print("=" * 80)
    print()
    
    # Configuration
    n_coords = 10000
    dimensions = (1000, 1000, 1000)
    table_size = 1024
    
    print(f"Configuration:")
    print(f"  Number of coordinates: {n_coords:,}")
    print(f"  Dimensions: {dimensions}")
    print(f"  Table size: {table_size}")
    print()
    
    # Generate random coordinates
    print("Generating random coordinates...")
    random.seed(42)  # For reproducibility
    coords = generate_random_coordinates(n_coords, dimensions)
    print(f"Generated {len(coords):,} coordinates")
    print()
    
    # Benchmark 1: Hash computation time
    print("-" * 80)
    print("BENCHMARK 1: Hash Computation Time")
    print("-" * 80)
    
    fib_time = benchmark_hash_computation_time(coords, table_size)
    py_time = benchmark_python_hash_time(coords, table_size)
    
    print(f"Fibonacci hash 3D: {fib_time:.2f} ns/hash")
    print(f"Python hash():     {py_time:.2f} ns/hash")
    print(f"Speedup:           {py_time / fib_time:.2f}x")
    
    # Check if meets target (<100ns)
    target_time = 100.0
    if fib_time < target_time:
        print(f"[PASS] Hash time {fib_time:.2f}ns < {target_time}ns target")
    else:
        print(f"[FAIL] Hash time {fib_time:.2f}ns >= {target_time}ns target")
    print()
    
    # Benchmark 2: Collision rate
    print("-" * 80)
    print("BENCHMARK 2: Collision Rate")
    print("-" * 80)
    
    fib_collision_rate = measure_collision_rate(coords, table_size)
    py_collision_rate = measure_python_hash_collision_rate(coords, table_size)
    
    print(f"Fibonacci hash 3D collision rate: {fib_collision_rate:.2f}%")
    print(f"Python hash() collision rate:     {py_collision_rate:.2f}%")
    
    # Check if meets target (<5%)
    target_collision_rate = 5.0
    if fib_collision_rate < target_collision_rate:
        print(f"[PASS] Collision rate {fib_collision_rate:.2f}% < {target_collision_rate}% target")
    else:
        print(f"[FAIL] Collision rate {fib_collision_rate:.2f}% >= {target_collision_rate}% target")
    print()
    
    # Benchmark 3: Distribution uniformity
    print("-" * 80)
    print("BENCHMARK 3: Distribution Uniformity (Chi-Square Test)")
    print("-" * 80)
    
    chi_square, p_value = measure_distribution_uniformity(coords, table_size)
    
    print(f"Chi-square statistic: {chi_square:.2f}")
    print(f"P-value:              {p_value:.4f}")
    
    # Check if meets target (p-value > 0.05)
    target_p_value = 0.05
    if p_value > target_p_value:
        print(f"[PASS] P-value {p_value:.4f} > {target_p_value} (uniform distribution)")
    else:
        print(f"[FAIL] P-value {p_value:.4f} <= {target_p_value} (non-uniform distribution)")
    print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Hash computation time:  {fib_time:.2f} ns/hash")
    print(f"Collision rate:         {fib_collision_rate:.2f}%")
    print(f"Distribution p-value:   {p_value:.4f}")
    print()
    
    # Overall pass/fail
    all_pass = (
        fib_time < target_time and
        fib_collision_rate < target_collision_rate and
        p_value > target_p_value
    )
    
    if all_pass:
        print("[PASS] ALL BENCHMARKS PASSED")
    else:
        print("[FAIL] SOME BENCHMARKS FAILED")
    print()


if __name__ == "__main__":
    run_benchmark()
