# Fibonacci KV Cache Examples

This directory contains example scripts demonstrating the usage of the Fibonacci KV Cache library.

## 1D Cache Examples

### basic_usage.py
Demonstrates the core functionality of the 1D FibonacciCacheOptimizer:
- Creating an optimizer
- Serializing and deserializing cache states
- Computing hash indices
- Saving and loading cache files
- Monitoring statistics

**Run:**
```bash
python examples/basic_usage.py
```

### monitoring.py
Shows advanced monitoring and callback features:
- Configuring collision callbacks
- Tracking cache hit rates
- Monitoring lookup latency
- Custom event handlers

**Run:**
```bash
python examples/monitoring.py
```

## 3D Lattice Cache Examples

### basic_3d_cache.py
Introduces the 3D spatial cache functionality:
- Creating a FibonacciLatticeCache
- Storing and retrieving values at 3D coordinates
- Understanding the 3D Fibonacci hash function
- Coordinate validation
- Spatial locality demonstration

**Run:**
```bash
python examples/basic_3d_cache.py
```

**Key concepts:**
- 3D coordinates: `(x, y, z)` tuples
- Plastic constant (φ₃ ≈ 1.4656) for 3D hashing
- Spatial locality: nearby points hash to nearby indices

### range_queries_example.py
Demonstrates spatial range queries using octree indexing:
- Populating cache with random 3D points
- Performing range queries with different radii
- Finding nearest neighbors
- Comparing octree vs linear scan performance
- Creating and querying dense clusters

**Run:**
```bash
python examples/range_queries_example.py
```

**Key concepts:**
- Range queries: `get_range(center, radius)`
- O(log n + k) complexity with octree
- Automatic octree maintenance
- Spatial applications: diffusion models, 3D graphics

### hierarchical_cache_example.py
Shows multi-resolution caching for coarse-to-fine algorithms:
- Creating hierarchical cache with multiple levels
- Storing values at different resolutions
- Coordinate scaling between levels
- Coarse-to-fine workflow
- Per-level statistics

**Run:**
```bash
python examples/hierarchical_cache_example.py
```

**Key concepts:**
- Level 0: Full resolution
- Level 1: Half resolution (dimensions / 2)
- Level 2: Quarter resolution (dimensions / 4)
- Automatic coordinate scaling
- Use cases: diffusion models, multi-scale processing

## Installation

Before running the examples, install the package:

```bash
# For 1D cache only
pip install fibkvc

# For 3D lattice extension
pip install fibkvc[3d]

# Or install from source
pip install -e .
```

## Example Output

All examples include:
- ✓ Success indicators for operations
- Detailed step-by-step output
- Statistics and performance metrics
- Error handling demonstrations

## Next Steps

After running the examples:

1. **Explore the tests**: See `tests/` for comprehensive test coverage
2. **Check benchmarks**: See `benchmarks/` for performance analysis
3. **Read the docs**: See `README.md` for full API reference
4. **Try integration**: Integrate with your own projects

## Support

- 📚 [Full Documentation](https://fibkvc.readthedocs.io)
- 🐛 [Report Issues](https://github.com/calivision/fibkvc/issues)
- 💬 [Discussions](https://github.com/calivision/fibkvc/discussions)
