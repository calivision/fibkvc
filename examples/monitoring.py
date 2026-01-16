# Copyright (c) 2026 California Vision, Inc. - David Anderson
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Advanced monitoring example for Fibonacci KV Cache.

This example demonstrates how to use monitoring features including:
- Cache hit rate tracking
- Lookup latency monitoring
- Collision event callbacks
- Resize event callbacks
- Statistics reporting
"""

from fibkvc import FibonacciCacheOptimizer, FibonacciHashingConfig
import random


def collision_handler(event):
    """Handle collision events."""
    print(f"   [COLLISION] Position {event['token_position']} → "
          f"Chain length: {event['chain_length']}")


def resize_handler(event):
    """Handle resize events."""
    print(f"   [RESIZE] {event['old_size']} → {event['new_size']} "
          f"({event['resize_time_ms']:.2f}ms)")


def main():
    """Demonstrate monitoring features."""
    
    print("=" * 60)
    print("Fibonacci KV Cache - Monitoring Example")
    print("=" * 60)
    
    # 1. Create config with monitoring enabled
    print("\n1. Creating configuration with monitoring...")
    config = FibonacciHashingConfig(
        monitor_cache_hit_rate=True,
        monitor_lookup_latency=True,
        log_collisions=True,
        log_resizes=True,
        log_statistics=True,
        on_collision_callback=collision_handler,
        on_resize_callback=resize_handler
    )
    print("   ✓ Monitoring enabled")
    
    # 2. Create optimizer with small table to trigger resizes
    print("\n2. Creating optimizer with small initial table...")
    optimizer = FibonacciCacheOptimizer(
        use_fibonacci=True,
        initial_table_size=32,  # Small to trigger resizes
        config=config
    )
    print(f"   ✓ Initial table size: {optimizer.table_size}")
    
    # 3. Perform lookups to generate cache hits/misses
    print("\n3. Performing 100 random lookups...")
    positions = []
    for i in range(100):
        # Mix of new and repeated positions
        if i < 50 or random.random() < 0.3:
            pos = random.randint(0, 1000)
            positions.append(pos)
        else:
            pos = random.choice(positions)
        
        optimizer.get_hash_index(pos)
    
    print(f"   ✓ Completed 100 lookups")
    
    # 4. Show monitoring statistics
    print("\n4. Monitoring statistics:")
    stats = optimizer.get_statistics()
    monitoring = stats['monitoring_stats']
    
    print(f"   Cache hits:          {monitoring['cache_hit_count']}")
    print(f"   Cache misses:        {monitoring['cache_miss_count']}")
    print(f"   Cache hit rate:      {monitoring['cache_hit_rate']:.1f}%")
    print(f"   Avg lookup time:     {monitoring['average_lookup_time_ms']:.3f}ms")
    print(f"   Total lookups:       {monitoring['lookup_count']}")
    
    # 5. Show hash table statistics
    print("\n5. Hash table statistics:")
    print(f"   Table size:          {stats['table_size']}")
    print(f"   Entries:             {stats['num_entries']}")
    print(f"   Load factor:         {stats['load_factor']:.2%}")
    print(f"   Collisions:          {stats['collision_count']}")
    print(f"   Max chain length:    {stats['max_collision_chain_length']}")
    print(f"   Resize count:        {stats['resize_count']}")
    
    # 6. Trigger more operations to show resize
    print("\n6. Adding more entries to trigger resize...")
    for i in range(1000, 1100):
        optimizer.get_hash_index(i)
    
    # 7. Final statistics
    print("\n7. Final statistics:")
    final_stats = optimizer.get_statistics()
    print(f"   Table size:          {final_stats['table_size']}")
    print(f"   Entries:             {final_stats['num_entries']}")
    print(f"   Load factor:         {final_stats['load_factor']:.2%}")
    print(f"   Total resizes:       {final_stats['resize_count']}")
    
    print("\n" + "=" * 60)
    print("Monitoring example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
