# FibKVC Benchmarks

This directory contains performance benchmarks for the fibkvc 3D lattice extension.

## Available Benchmarks

### 1. Hash Distribution Benchmark (`bench_hash_distribution.py`)

Measures the performance and quality of the 3D Fibonacci hash function.

**Metrics:**
- Hash computation time (target: <100ns per hash)
- Collision rate for 10,000 random coordinates (target: <5%)
- Distribution uniformity using chi-square test (target: p-value > 0.05)
- Comparison against Python's built-in `hash()`

**Usage:**
```bash
python benchmarks/bench_hash_distribution.py
```

**Requirements:** 9.1, 9.2

### 2. Cache Performance Benchmark (`bench_cache_performance.py`)

Measures the performance of FibonacciLatticeCache insert and get operations.

**Metrics:**
- Insert operation performance at various load factors
- Get operation performance at various load factors
- Comparison against Python dict
- Collision statistics (rate, chain lengths)
- Verification of 15-20% improvement target

**Usage:**
```bash
python benchmarks/bench_cache_performance.py
```

**Requirements:** 9.1

### 3. Range Query Benchmark (`bench_range_queries.py`)

Measures the performance of octree-based range queries.

**Metrics:**
- Range query performance at various radii
- Comparison against linear scan
- Performance at various cache sizes (100, 500, 1000, 2000 coordinates)
- Complexity analysis to verify O(log n + k) behavior
- Verification of 10x speedup target over linear scan

**Usage:**
```bash
python benchmarks/bench_range_queries.py
```

**Requirements:** 9.1

## Running All Benchmarks

To run all benchmarks sequentially:

```bash
python benchmarks/bench_hash_distribution.py
python benchmarks/bench_cache_performance.py
python benchmarks/bench_range_queries.py
```

Or create a simple script:

```bash
# run_all_benchmarks.sh
#!/bin/bash
echo "Running Hash Distribution Benchmark..."
python benchmarks/bench_hash_distribution.py
echo ""
echo "Running Cache Performance Benchmark..."
python benchmarks/bench_cache_performance.py
echo ""
echo "Running Range Query Benchmark..."
python benchmarks/bench_range_queries.py
```

## Interpreting Results

### Hash Distribution

- **Hash computation time**: Should be <100ns per hash for O(1) performance
- **Collision rate**: Should be <5% for uniform distribution (note: high collision rates are expected when hashing many coordinates into small tables)
- **P-value**: Should be >0.05 indicating uniform distribution

### Cache Performance

- **Insert/Get times**: Measured in microseconds (μs) and nanoseconds (ns)
- **Speedup**: Ratio of Python dict time to FibonacciLatticeCache time
  - Values >1.0 indicate FibonacciLatticeCache is faster
  - Values <1.0 indicate Python dict is faster (expected for highly optimized C implementation)
- **Collision statistics**: Track hash table health
  - Higher load factors → more collisions
  - Longer chain lengths → slower lookups

### Range Queries

- **Speedup**: Ratio of linear scan time to octree time
  - Target: 10x faster than linear scan for large datasets
  - Speedup increases with dataset size (n)
- **Complexity**: Time/log(n+k) should remain relatively constant
  - Verifies O(log n + k) complexity
  - k = number of results returned

## Performance Targets

Based on requirements 9.1 and 9.2:

1. **Hash function**: <100ns per hash, <5% collision rate, uniform distribution
2. **Cache operations**: 15-20% improvement over Python dict (1.15-1.20x speedup)
3. **Range queries**: 10x faster than linear scan for large datasets

## Notes

- All benchmarks use `random.seed(42)` for reproducibility
- Benchmarks measure average performance over multiple iterations
- Python's built-in dict is highly optimized (C implementation), so direct comparisons may show dict as faster
- The value of FibonacciLatticeCache is in spatial locality preservation and range query support, not raw speed
- Performance varies by system, Python version, and hardware

## Dependencies

- Python 3.8+
- fibkvc package (with 3D lattice extension)
- No external dependencies (uses only Python standard library)
