# Copyright (c) 2026 California Vision, Inc. - David Anderson
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Basic usage example for Fibonacci KV Cache.

This example demonstrates the core functionality of the FibonacciCacheOptimizer
including serialization, deserialization, and hash index lookups.
"""

from fibkvc import FibonacciCacheOptimizer, FibonacciHashingConfig


def main():
    """Demonstrate basic usage of Fibonacci KV Cache."""
    
    print("=" * 60)
    print("Fibonacci KV Cache - Basic Usage Example")
    print("=" * 60)
    
    # 1. Create optimizer with default settings
    print("\n1. Creating FibonacciCacheOptimizer...")
    optimizer = FibonacciCacheOptimizer(
        use_fibonacci=True,
        initial_table_size=256
    )
    print(f"   ✓ Optimizer created with table size: {optimizer.table_size}")
    
    # 2. Create sample cache state
    print("\n2. Creating sample cache state...")
    cache_state = {
        "model_name": "dream-7b",
        "entries": {
            0: {"key": "k0", "value": "v0"},
            1: {"key": "k1", "value": "v1"},
            42: {"key": "k42", "value": "v42"},
            100: {"key": "k100", "value": "v100"},
            255: {"key": "k255", "value": "v255"},
        }
    }
    print(f"   ✓ Created cache with {len(cache_state['entries'])} entries")
    
    # 3. Serialize cache state
    print("\n3. Serializing cache state with Fibonacci optimization...")
    json_str = optimizer.serialize_cache_state(cache_state)
    print(f"   ✓ Serialized to {len(json_str)} bytes")
    print(f"   First 100 chars: {json_str[:100]}...")
    
    # 4. Deserialize cache state
    print("\n4. Deserializing cache state...")
    restored_state = optimizer.deserialize_cache_state(json_str)
    print(f"   ✓ Restored {len(restored_state.get('entries', {}))} entries")
    
    # 5. Verify round-trip
    print("\n5. Verifying round-trip integrity...")
    original_entries = cache_state["entries"]
    restored_entries = restored_state["entries"]
    
    all_match = True
    for key in original_entries:
        if str(key) not in restored_entries:
            print(f"   ✗ Missing key: {key}")
            all_match = False
        elif original_entries[key] != restored_entries[str(key)]:
            print(f"   ✗ Mismatch for key {key}")
            all_match = False
    
    if all_match:
        print("   ✓ All entries match - round-trip successful!")
    
    # 6. Get hash indices for token positions
    print("\n6. Computing Fibonacci hash indices...")
    token_positions = [0, 1, 42, 100, 255, 1000, 10000]
    
    for pos in token_positions:
        hash_idx = optimizer.get_hash_index(pos)
        print(f"   Token position {pos:5d} → Hash index {hash_idx:3d}")
    
    # 7. Show statistics
    print("\n7. Optimizer statistics:")
    stats = optimizer.get_statistics()
    print(f"   Total serializations:   {stats['total_serializations']}")
    print(f"   Total deserializations: {stats['total_deserializations']}")
    print(f"   Collision count:        {stats['collision_count']}")
    print(f"   Load factor:            {stats['load_factor']:.2%}")
    print(f"   Table size:             {stats['table_size']}")
    print(f"   Entries:                {stats['num_entries']}")
    
    # 8. Save to file
    print("\n8. Saving cache state to file...")
    optimizer.save_to_file(cache_state, "cache_snapshot.json")
    print("   ✓ Saved to cache_snapshot.json")
    
    # 9. Load from file
    print("\n9. Loading cache state from file...")
    loaded_state = optimizer.load_from_file("cache_snapshot.json")
    print(f"   ✓ Loaded {len(loaded_state.get('entries', {}))} entries")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
