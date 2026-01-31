# Fibonacci KV Cache - Quick Reference

## Installation

```bash
# 1D cache only
pip install fibkvc

# With 3D lattice extension
pip install fibkvc[3d]
```

## Basic Usage (1D)

```python
from fibkvc import FibonacciCacheOptimizer

# Create optimizer
optimizer = FibonacciCacheOptimizer()

# Serialize cache
json_str = optimizer.serialize_cache_state(cache_state)

# Deserialize cache
restored = optimizer.deserialize_cache_state(json_str)

# Get hash index
hash_idx = optimizer.get_hash_index(token_position=42)
```

## 3D Lattice Cache

```python
from fibkvc.lattice import FibonacciLatticeCache

# Create 3D cache
cache = FibonacciLatticeCache(
    dimensions=(100, 100, 100),
    use_octree=True
)

# Store and retrieve
cache.set((x, y, z), value)
value = cache.get((x, y, z))

# Range queries
results = cache.get_range(center=(50, 50, 50), radius=20.0)
```

## Core Functions

### 1D Cache

#### `fibonacci_hash(key, table_size)`
Compute Fibonacci hash for a key.

```python
from fibkvc import fibonacci_hash

hash_idx = fibonacci_hash(key=12345, table_size=256)
```

#### `FibonacciCacheOptimizer`
Main cache optimizer class.

```python
optimizer = FibonacciCacheOptimizer(
    use_fibonacci=True,        # Enable Fibonacci hashing
    initial_table_size=256,    # Must be power of 2
    config=None                # Optional config
)
```

**Methods:**
- `serialize_cache_state(cache_state)` → JSON string
- `deserialize_cache_state(json_str)` → Dict
- `get_hash_index(token_position)` → int
- `get_statistics()` → Dict
- `save_to_file(cache_state, filepath)`
- `load_from_file(filepath)` → Dict

#### `FibonacciHashingConfig`
Configuration for monitoring and callbacks.

```python
from fibkvc import FibonacciHashingConfig

config = FibonacciHashingConfig(
    monitor_cache_hit_rate=True,
    monitor_lookup_latency=True,
    log_collisions=True,
    on_collision_callback=lambda e: print(e)
)
```

### 3D Lattice Cache

#### `fibonacci_hash_3d(x, y, z, table_size)`
Compute 3D Fibonacci hash using plastic constant.

```python
from fibkvc.lattice import fibonacci_hash_3d

hash_idx = fibonacci_hash_3d(x=10, y=20, z=30, table_size=256)
```

#### `FibonacciLatticeCache`
3D spatial cache with octree indexing.

```python
from fibkvc.lattice import FibonacciLatticeCache

cache = FibonacciLatticeCache(
    dimensions=(width, height, depth),
    initial_size=1024,
    max_load_factor=0.75,
    use_octree=True
)
```

**Methods:**
- `set(coords, value)` - Store value at 3D coordinates
- `get(coords)` → value or None
- `get_range(center, radius)` → List[(coords, value)]
- `get_statistics()` → Dict

#### `HierarchicalLatticeCache`
Multi-resolution cache for coarse-to-fine algorithms.

```python
from fibkvc.lattice import HierarchicalLatticeCache

cache = HierarchicalLatticeCache(
    dimensions=(128, 128, 128),
    num_levels=3
)

# Store at different levels
cache.set((x, y, z), value, level=0)  # Full resolution
cache.set((x, y, z), value, level=1)  # Half resolution

# Retrieve from specific level
value = cache.get_hierarchical((x, y, z), level=0)
```

## Monitoring

```python
# Get statistics
stats = optimizer.get_statistics()

print(f"Cache hit rate: {stats['monitoring_stats']['cache_hit_rate']:.1f}%")
print(f"Collisions: {stats['collision_count']}")
print(f"Load factor: {stats['load_factor']:.2%}")
```

## Common Patterns

### Save/Load Cache

```python
# Save
optimizer.save_to_file(cache_state, "cache.json")

# Load
cache_state = optimizer.load_from_file("cache.json")
```

### Custom Callbacks

```python
def on_collision(event):
    print(f"Collision at {event['token_position']}")

config = FibonacciHashingConfig(
    on_collision_callback=on_collision
)
```

### Batch Lookups

```python
positions = [0, 1, 42, 100, 255]
hash_indices = [optimizer.get_hash_index(pos) for pos in positions]
```

## Performance Tips

1. **Table Size**: Use power of 2 (256, 512, 1024)
2. **Load Factor**: Auto-resizes at 0.75 (default)
3. **Monitoring**: Enable only in development/testing
4. **Callbacks**: Keep them lightweight

## Troubleshooting

**Q: "table_size must be power of 2"**  
A: Use 256, 512, 1024, etc. Not 300, 500, etc.

**Q: Slow performance?**  
A: Check load factor. If > 0.75, table will auto-resize.

**Q: High collision rate?**  
A: Increase initial_table_size or check key distribution.

## Links

- 📚 [Full Documentation](https://fibkvc.readthedocs.io)
- 📄 [Research Paper](https://arxiv.org/abs/2501.xxxxx)
- 🐛 [Report Issues](https://github.com/calivision/fibkvc)
- 💬 [Discussions](https://github.com/calivision/fibkvc/discussions)

## Enterprise (Q2 2026)

Need production support? → [https://fibkvc.california.vision](https://fibkvc.california.vision)
