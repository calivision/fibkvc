# Copyright (c) 2026 California Vision, Inc. - David Anderson
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Property-based tests for FibonacciLatticeCache.

These tests use Hypothesis to verify universal properties that should hold
for all valid inputs, providing stronger correctness guarantees than
example-based unit tests.

Property 3: Set-Get Round Trip
Validates: Requirements 2.3, 2.4
"""

import pytest
from hypothesis import given, strategies as st, settings
from fibkvc.lattice import FibonacciLatticeCache
import math


# Strategy for generating valid 3D coordinates within dimensions
@st.composite
def coordinates_3d(draw, dimensions=(100, 100, 100)):
    """
    Generate valid 3D coordinates within specified dimensions.
    
    Args:
        draw: Hypothesis draw function
        dimensions: (width, height, depth) bounds
    
    Returns:
        Tuple of (x, y, z) coordinates
    """
    w, h, d = dimensions
    x = draw(st.integers(min_value=0, max_value=w-1))
    y = draw(st.integers(min_value=0, max_value=h-1))
    z = draw(st.integers(min_value=0, max_value=d-1))
    return (x, y, z)


# Strategy for generating coordinate-value pairs
@st.composite
def coord_value_pairs(draw, dimensions=(100, 100, 100), num_pairs=None):
    """
    Generate a list of (coords, value) pairs.
    
    Args:
        draw: Hypothesis draw function
        dimensions: (width, height, depth) bounds
        num_pairs: Number of pairs to generate (None for random)
    
    Returns:
        List of (coords, value) tuples
    """
    if num_pairs is None:
        num_pairs = draw(st.integers(min_value=1, max_value=20))
    
    pairs = []
    for _ in range(num_pairs):
        coords = draw(coordinates_3d(dimensions))
        value = draw(st.integers())
        pairs.append((coords, value))
    
    return pairs


class TestSetGetRoundTrip:
    """
    Property 3: Set-Get Round Trip
    
    For any valid 3D coordinates and any value, after calling cache.set(coords, value),
    calling cache.get(coords) should return the same value.
    
    Validates: Requirements 2.3, 2.4
    Feature: fibkvc-3d-lattice-extension, Property 3
    """
    
    @settings(max_examples=100)
    @given(
        coords=coordinates_3d(dimensions=(100, 100, 100)),
        value=st.integers()
    )
    def test_single_entry_round_trip(self, coords, value):
        """
        Test that a single set-get round trip preserves the value.
        
        Property: For any coords and value, cache.get(coords) == value after cache.set(coords, value)
        """
        cache = FibonacciLatticeCache(dimensions=(100, 100, 100))
        
        # Set value
        cache.set(coords, value)
        
        # Get should return same value
        retrieved = cache.get(coords)
        assert retrieved == value, \
            f"Round trip failed: set {value} at {coords}, got {retrieved}"
    
    @settings(max_examples=100)
    @given(
        pairs=coord_value_pairs(dimensions=(100, 100, 100))
    )
    def test_multiple_entries_round_trip(self, pairs):
        """
        Test that multiple set-get round trips preserve all values.
        
        Property: For any list of (coords, value) pairs, all values should be
        retrievable after insertion.
        """
        cache = FibonacciLatticeCache(dimensions=(100, 100, 100))
        
        # Insert all pairs
        for coords, value in pairs:
            cache.set(coords, value)
        
        # Verify all pairs are retrievable
        # Note: If same coords appear multiple times, last value wins
        coords_to_value = {}
        for coords, value in pairs:
            coords_to_value[coords] = value
        
        for coords, expected_value in coords_to_value.items():
            retrieved = cache.get(coords)
            assert retrieved == expected_value, \
                f"Round trip failed for {coords}: expected {expected_value}, got {retrieved}"
    
    @settings(max_examples=100)
    @given(
        coords=coordinates_3d(dimensions=(100, 100, 100)),
        value1=st.integers(),
        value2=st.integers()
    )
    def test_update_round_trip(self, coords, value1, value2):
        """
        Test that updating a value preserves the new value.
        
        Property: After setting coords to value1, then to value2,
        get(coords) should return value2.
        """
        cache = FibonacciLatticeCache(dimensions=(100, 100, 100))
        
        # Set initial value
        cache.set(coords, value1)
        
        # Update to new value
        cache.set(coords, value2)
        
        # Get should return updated value
        retrieved = cache.get(coords)
        assert retrieved == value2, \
            f"Update failed: set {value1} then {value2} at {coords}, got {retrieved}"
    
    @settings(max_examples=100)
    @given(
        pairs=coord_value_pairs(dimensions=(50, 50, 50), num_pairs=30)
    )
    def test_round_trip_with_collisions(self, pairs):
        """
        Test round trip with small cache to force collisions.
        
        Property: Even with hash collisions, all values should be retrievable.
        Uses small initial size to increase collision probability.
        """
        # Small initial size to force collisions
        cache = FibonacciLatticeCache(
            dimensions=(50, 50, 50),
            initial_size=16
        )
        
        # Insert all pairs
        for coords, value in pairs:
            cache.set(coords, value)
        
        # Build expected state (last value for each coord)
        coords_to_value = {}
        for coords, value in pairs:
            coords_to_value[coords] = value
        
        # Verify all values are retrievable
        for coords, expected_value in coords_to_value.items():
            retrieved = cache.get(coords)
            assert retrieved == expected_value, \
                f"Collision handling failed for {coords}: expected {expected_value}, got {retrieved}"
    
    @settings(max_examples=100)
    @given(
        pairs=coord_value_pairs(dimensions=(100, 100, 100), num_pairs=50)
    )
    def test_round_trip_with_resize(self, pairs):
        """
        Test round trip when cache resizes during insertions.
        
        Property: Values should remain retrievable after automatic resize.
        Uses small initial size and low load factor to trigger resize.
        """
        # Small size and low load factor to trigger resize
        cache = FibonacciLatticeCache(
            dimensions=(100, 100, 100),
            initial_size=8,
            max_load_factor=0.5
        )
        
        # Insert all pairs (should trigger resize)
        for coords, value in pairs:
            cache.set(coords, value)
        
        # Build expected state
        coords_to_value = {}
        for coords, value in pairs:
            coords_to_value[coords] = value
        
        # Verify all values are retrievable after resize
        for coords, expected_value in coords_to_value.items():
            retrieved = cache.get(coords)
            assert retrieved == expected_value, \
                f"Resize broke round trip for {coords}: expected {expected_value}, got {retrieved}"
    
    @settings(max_examples=100)
    @given(
        coords=coordinates_3d(dimensions=(100, 100, 100)),
        value=st.one_of(
            st.integers(),
            st.text(),
            st.lists(st.integers()),
            st.dictionaries(st.text(), st.integers()),
            st.none()
        )
    )
    def test_round_trip_various_types(self, coords, value):
        """
        Test round trip with various value types.
        
        Property: Round trip should work for any value type (int, str, list, dict, None).
        """
        cache = FibonacciLatticeCache(dimensions=(100, 100, 100))
        
        # Set value
        cache.set(coords, value)
        
        # Get should return same value
        retrieved = cache.get(coords)
        assert retrieved == value, \
            f"Round trip failed for type {type(value)}: set {value}, got {retrieved}"



class TestSpatialCollisionResolution:
    """
    Property 4: Spatial Collision Resolution
    
    For any sequence of insertions that cause collisions, when retrieving values,
    all values should be retrievable and the probe sequence should prioritize
    spatially nearby indices.
    
    Validates: Requirements 3.1, 3.2
    Feature: fibkvc-3d-lattice-extension, Property 4
    """
    
    @settings(max_examples=100)
    @given(
        pairs=coord_value_pairs(dimensions=(50, 50, 50), num_pairs=20)
    )
    def test_all_values_retrievable_with_collisions(self, pairs):
        """
        Test that all values are retrievable even with forced collisions.
        
        Property: For any set of insertions with collisions (small table size),
        all inserted values should be retrievable via get().
        """
        # Use very small table size to force many collisions
        cache = FibonacciLatticeCache(
            dimensions=(50, 50, 50),
            initial_size=8,
            max_load_factor=0.9  # High load factor to avoid resize
        )
        
        # Insert all pairs
        for coords, value in pairs:
            cache.set(coords, value)
        
        # Build expected state (last value for each coord)
        coords_to_value = {}
        for coords, value in pairs:
            coords_to_value[coords] = value
        
        # Verify all values are retrievable
        for coords, expected_value in coords_to_value.items():
            retrieved = cache.get(coords)
            assert retrieved == expected_value, \
                f"Failed to retrieve value after collision: coords={coords}, expected={expected_value}, got={retrieved}"
    
    @settings(max_examples=100)
    @given(
        pairs=coord_value_pairs(dimensions=(30, 30, 30), num_pairs=15)
    )
    def test_spatial_probe_sequence_prioritizes_nearby(self, pairs):
        """
        Test that spatial probing prioritizes nearby indices.
        
        Property: When collisions occur, the probe sequence should result in
        spatially nearby coordinates being stored at nearby table indices more
        often than random probing would.
        
        This is verified by checking that collision chain lengths are reasonable
        and that the cache maintains good performance characteristics.
        """
        # Small table to force collisions
        cache = FibonacciLatticeCache(
            dimensions=(30, 30, 30),
            initial_size=16,
            max_load_factor=0.9
        )
        
        # Insert all pairs
        for coords, value in pairs:
            cache.set(coords, value)
        
        # Get collision statistics
        stats = cache.get_collision_stats()
        
        # Verify reasonable collision behavior
        # With spatial probing, average chain length should be low
        assert stats['avg_chain_length'] < 5.0, \
            f"Average chain length too high: {stats['avg_chain_length']}"
        
        # Maximum chain length should be reasonable
        assert stats['max_chain_length'] < 10, \
            f"Maximum chain length too high: {stats['max_chain_length']}"
        
        # All values should still be retrievable
        coords_to_value = {}
        for coords, value in pairs:
            coords_to_value[coords] = value
        
        for coords, expected_value in coords_to_value.items():
            retrieved = cache.get(coords)
            assert retrieved == expected_value, \
                f"Spatial probing broke retrieval: coords={coords}"
    
    @settings(max_examples=100)
    @given(
        seed=st.integers(min_value=0, max_value=1000)
    )
    def test_collision_handling_with_same_hash(self, seed):
        """
        Test collision handling when multiple coordinates hash to same index.
        
        Property: When multiple coordinates hash to the same index, all should
        be stored and retrievable using spatial probing.
        """
        import random
        random.seed(seed)
        
        # Very small table to maximize collision probability
        cache = FibonacciLatticeCache(
            dimensions=(20, 20, 20),
            initial_size=4,
            max_load_factor=0.95
        )
        
        # Generate random coordinates and values
        pairs = []
        for i in range(10):
            coords = (
                random.randint(0, 19),
                random.randint(0, 19),
                random.randint(0, 19)
            )
            value = random.randint(0, 1000)
            pairs.append((coords, value))
        
        # Insert all pairs
        for coords, value in pairs:
            cache.set(coords, value)
        
        # Build expected state
        coords_to_value = {}
        for coords, value in pairs:
            coords_to_value[coords] = value
        
        # Verify all are retrievable
        for coords, expected_value in coords_to_value.items():
            retrieved = cache.get(coords)
            assert retrieved == expected_value, \
                f"Collision handling failed: coords={coords}, expected={expected_value}, got={retrieved}"
    
    @settings(max_examples=100)
    @given(
        pairs=coord_value_pairs(dimensions=(40, 40, 40), num_pairs=25)
    )
    def test_collision_stats_tracking(self, pairs):
        """
        Test that collision statistics are properly tracked.
        
        Property: The cache should track collision chain lengths for monitoring,
        and these statistics should be consistent with the actual cache state.
        """
        cache = FibonacciLatticeCache(
            dimensions=(40, 40, 40),
            initial_size=16,
            max_load_factor=0.9
        )
        
        # Insert all pairs
        for coords, value in pairs:
            cache.set(coords, value)
        
        # Get statistics
        stats = cache.get_collision_stats()
        
        # Verify statistics are reasonable
        assert stats['load_factor'] >= 0.0, "Load factor should be non-negative"
        assert stats['load_factor'] <= 1.0, "Load factor should not exceed 1.0"
        assert stats['avg_chain_length'] >= 0.0, "Average chain length should be non-negative"
        assert stats['max_chain_length'] >= 0, "Max chain length should be non-negative"
        assert stats['total_collisions'] >= 0, "Total collisions should be non-negative"
        
        # If there are collisions, max should be >= avg
        if stats['total_collisions'] > 0:
            assert stats['max_chain_length'] >= stats['avg_chain_length'], \
                "Max chain length should be >= average chain length"
    
    @settings(max_examples=100)
    @given(
        pairs1=coord_value_pairs(dimensions=(30, 30, 30), num_pairs=10),
        pairs2=coord_value_pairs(dimensions=(30, 30, 30), num_pairs=10)
    )
    def test_collision_resolution_with_updates(self, pairs1, pairs2):
        """
        Test that collision resolution works correctly with updates.
        
        Property: When updating values in a cache with collisions, the updates
        should not break the collision chains and all values should remain retrievable.
        """
        cache = FibonacciLatticeCache(
            dimensions=(30, 30, 30),
            initial_size=8,
            max_load_factor=0.9
        )
        
        # Insert first batch
        for coords, value in pairs1:
            cache.set(coords, value)
        
        # Insert second batch (may include updates)
        for coords, value in pairs2:
            cache.set(coords, value)
        
        # Build expected final state
        coords_to_value = {}
        for coords, value in pairs1:
            coords_to_value[coords] = value
        for coords, value in pairs2:
            coords_to_value[coords] = value
        
        # Verify all values are retrievable
        for coords, expected_value in coords_to_value.items():
            retrieved = cache.get(coords)
            assert retrieved == expected_value, \
                f"Update with collisions failed: coords={coords}, expected={expected_value}, got={retrieved}"



class TestOctreeCacheConsistency:
    """
    Property 6: Octree-Cache Consistency
    
    For any sequence of insert operations on the cache, range queries should
    always return exactly the set of values that are currently in the cache
    within the specified range (no stale values, no missing values).
    
    Validates: Requirements 5.2, 5.3
    Feature: fibkvc-3d-lattice-extension, Property 6
    """
    
    @staticmethod
    def _euclidean_distance(p1, p2):
        """Compute Euclidean distance between two 3D points."""
        return math.sqrt(
            (p1[0] - p2[0])**2 + 
            (p1[1] - p2[1])**2 + 
            (p1[2] - p2[2])**2
        )
    
    @settings(max_examples=100)
    @given(
        pairs=coord_value_pairs(dimensions=(50, 50, 50), num_pairs=20),
        center=coordinates_3d(dimensions=(50, 50, 50)),
        radius=st.floats(min_value=1.0, max_value=30.0)
    )
    def test_range_query_returns_all_within_radius(self, pairs, center, radius):
        """
        Test that range queries return all cached values within radius.
        
        Property: For any set of insertions and any range query, all cached
        coordinates within the radius should be returned.
        """
        # Create cache with octree enabled
        cache = FibonacciLatticeCache(
            dimensions=(50, 50, 50),
            use_octree=True
        )
        
        # Insert all pairs
        for coords, value in pairs:
            cache.set(coords, value)
        
        # Build expected state (last value for each coord)
        coords_to_value = {}
        for coords, value in pairs:
            coords_to_value[coords] = value
        
        # Perform range query
        results = cache.get_range(center, radius)
        result_coords = {coords for coords, value in results}
        
        # Verify all coordinates within radius are in results
        for coords, value in coords_to_value.items():
            distance = self._euclidean_distance(coords, center)
            if distance <= radius:
                assert coords in result_coords, \
                    f"Missing coordinate {coords} at distance {distance} from center {center} (radius={radius})"
    
    @settings(max_examples=100)
    @given(
        pairs=coord_value_pairs(dimensions=(50, 50, 50), num_pairs=20),
        center=coordinates_3d(dimensions=(50, 50, 50)),
        radius=st.floats(min_value=1.0, max_value=30.0)
    )
    def test_range_query_excludes_all_outside_radius(self, pairs, center, radius):
        """
        Test that range queries exclude all cached values outside radius.
        
        Property: For any set of insertions and any range query, no cached
        coordinates outside the radius should be returned.
        """
        # Create cache with octree enabled
        cache = FibonacciLatticeCache(
            dimensions=(50, 50, 50),
            use_octree=True
        )
        
        # Insert all pairs
        for coords, value in pairs:
            cache.set(coords, value)
        
        # Perform range query
        results = cache.get_range(center, radius)
        
        # Verify all returned coordinates are within radius
        for coords, value in results:
            distance = self._euclidean_distance(coords, center)
            assert distance <= radius, \
                f"Coordinate {coords} at distance {distance} should not be in results (radius={radius})"
    
    @settings(max_examples=100)
    @given(
        pairs=coord_value_pairs(dimensions=(50, 50, 50), num_pairs=20),
        center=coordinates_3d(dimensions=(50, 50, 50)),
        radius=st.floats(min_value=1.0, max_value=30.0)
    )
    def test_range_query_returns_correct_values(self, pairs, center, radius):
        """
        Test that range queries return correct values (not stale).
        
        Property: For any set of insertions and any range query, the returned
        values should match the current cache state (no stale values).
        """
        # Create cache with octree enabled
        cache = FibonacciLatticeCache(
            dimensions=(50, 50, 50),
            use_octree=True
        )
        
        # Insert all pairs
        for coords, value in pairs:
            cache.set(coords, value)
        
        # Build expected state (last value for each coord)
        coords_to_value = {}
        for coords, value in pairs:
            coords_to_value[coords] = value
        
        # Perform range query
        results = cache.get_range(center, radius)
        
        # Verify all returned values match cache state
        for coords, value in results:
            expected_value = coords_to_value[coords]
            assert value == expected_value, \
                f"Stale value for {coords}: expected {expected_value}, got {value}"
    
    @settings(max_examples=100)
    @given(
        pairs1=coord_value_pairs(dimensions=(50, 50, 50), num_pairs=10),
        pairs2=coord_value_pairs(dimensions=(50, 50, 50), num_pairs=10),
        center=coordinates_3d(dimensions=(50, 50, 50)),
        radius=st.floats(min_value=1.0, max_value=30.0)
    )
    def test_range_query_consistency_after_updates(self, pairs1, pairs2, center, radius):
        """
        Test that range queries remain consistent after updates.
        
        Property: After a sequence of insertions and updates, range queries
        should return the current state (no stale values from octree).
        """
        # Create cache with octree enabled
        cache = FibonacciLatticeCache(
            dimensions=(50, 50, 50),
            use_octree=True
        )
        
        # Insert first batch
        for coords, value in pairs1:
            cache.set(coords, value)
        
        # Insert second batch (may include updates)
        for coords, value in pairs2:
            cache.set(coords, value)
        
        # Build expected final state
        coords_to_value = {}
        for coords, value in pairs1:
            coords_to_value[coords] = value
        for coords, value in pairs2:
            coords_to_value[coords] = value
        
        # Perform range query
        results = cache.get_range(center, radius)
        result_dict = {coords: value for coords, value in results}
        
        # Verify completeness: all within radius are returned
        for coords, expected_value in coords_to_value.items():
            distance = self._euclidean_distance(coords, center)
            if distance <= radius:
                assert coords in result_dict, \
                    f"Missing coordinate {coords} after updates"
                assert result_dict[coords] == expected_value, \
                    f"Stale value for {coords} after updates: expected {expected_value}, got {result_dict[coords]}"
        
        # Verify correctness: all returned are within radius
        for coords, value in results:
            distance = self._euclidean_distance(coords, center)
            assert distance <= radius, \
                f"Coordinate {coords} outside radius after updates"
    
    @settings(max_examples=100)
    @given(
        pairs=coord_value_pairs(dimensions=(50, 50, 50), num_pairs=30),
        center=coordinates_3d(dimensions=(50, 50, 50)),
        radius=st.floats(min_value=1.0, max_value=30.0)
    )
    def test_range_query_consistency_with_resize(self, pairs, center, radius):
        """
        Test that range queries remain consistent after cache resize.
        
        Property: After cache resizes (which rebuilds octree), range queries
        should still return correct results.
        """
        # Create cache with small size to trigger resize
        cache = FibonacciLatticeCache(
            dimensions=(50, 50, 50),
            initial_size=8,
            max_load_factor=0.5,
            use_octree=True
        )
        
        # Insert all pairs (should trigger resize)
        for coords, value in pairs:
            cache.set(coords, value)
        
        # Build expected state
        coords_to_value = {}
        for coords, value in pairs:
            coords_to_value[coords] = value
        
        # Perform range query after resize
        results = cache.get_range(center, radius)
        result_dict = {coords: value for coords, value in results}
        
        # Verify completeness
        for coords, expected_value in coords_to_value.items():
            distance = self._euclidean_distance(coords, center)
            if distance <= radius:
                assert coords in result_dict, \
                    f"Missing coordinate {coords} after resize"
                assert result_dict[coords] == expected_value, \
                    f"Wrong value for {coords} after resize"
        
        # Verify correctness
        for coords, value in results:
            distance = self._euclidean_distance(coords, center)
            assert distance <= radius, \
                f"Coordinate {coords} outside radius after resize"
    
    @settings(max_examples=100)
    @given(
        pairs=coord_value_pairs(dimensions=(50, 50, 50), num_pairs=15),
        center=coordinates_3d(dimensions=(50, 50, 50))
    )
    def test_range_query_with_zero_radius(self, pairs, center):
        """
        Test range query with zero radius (edge case).
        
        Property: A range query with radius=0 should only return the exact
        center coordinate if it exists in the cache.
        """
        # Create cache with octree enabled
        cache = FibonacciLatticeCache(
            dimensions=(50, 50, 50),
            use_octree=True
        )
        
        # Insert all pairs
        for coords, value in pairs:
            cache.set(coords, value)
        
        # Build expected state
        coords_to_value = {}
        for coords, value in pairs:
            coords_to_value[coords] = value
        
        # Perform range query with radius=0
        results = cache.get_range(center, radius=0.0)
        
        # Should only return center if it exists
        if center in coords_to_value:
            assert len(results) == 1, \
                f"Zero radius should return only center coordinate"
            assert results[0][0] == center, \
                f"Zero radius should return center coordinate"
            assert results[0][1] == coords_to_value[center], \
                f"Zero radius should return correct value for center"
        else:
            assert len(results) == 0, \
                f"Zero radius should return empty if center not in cache"
    
    @settings(max_examples=100)
    @given(
        pairs=coord_value_pairs(dimensions=(50, 50, 50), num_pairs=20),
        center=coordinates_3d(dimensions=(50, 50, 50)),
        radius=st.floats(min_value=50.0, max_value=100.0)
    )
    def test_range_query_with_large_radius(self, pairs, center, radius):
        """
        Test range query with large radius (should return all or most entries).
        
        Property: A range query with very large radius should return all
        cached entries within the lattice dimensions.
        """
        # Create cache with octree enabled
        cache = FibonacciLatticeCache(
            dimensions=(50, 50, 50),
            use_octree=True
        )
        
        # Insert all pairs
        for coords, value in pairs:
            cache.set(coords, value)
        
        # Build expected state
        coords_to_value = {}
        for coords, value in pairs:
            coords_to_value[coords] = value
        
        # Perform range query with large radius
        results = cache.get_range(center, radius)
        result_dict = {coords: value for coords, value in results}
        
        # Verify all within radius are returned
        for coords, expected_value in coords_to_value.items():
            distance = self._euclidean_distance(coords, center)
            if distance <= radius:
                assert coords in result_dict, \
                    f"Missing coordinate {coords} with large radius"
                assert result_dict[coords] == expected_value, \
                    f"Wrong value for {coords} with large radius"
    
    @settings(max_examples=100)
    @given(
        seed=st.integers(min_value=0, max_value=1000)
    )
    def test_range_query_incremental_consistency(self, seed):
        """
        Test that range queries remain consistent after each insertion.
        
        Property: After each insertion, range queries should reflect the
        current cache state (incremental consistency).
        """
        import random
        random.seed(seed)
        
        # Create cache with octree enabled
        cache = FibonacciLatticeCache(
            dimensions=(30, 30, 30),
            use_octree=True
        )
        
        # Fixed center and radius for testing
        center = (15, 15, 15)
        radius = 10.0
        
        # Track expected state
        coords_to_value = {}
        
        # Insert entries one by one and verify after each
        for i in range(15):
            coords = (
                random.randint(0, 29),
                random.randint(0, 29),
                random.randint(0, 29)
            )
            value = random.randint(0, 1000)
            
            # Insert
            cache.set(coords, value)
            coords_to_value[coords] = value
            
            # Verify range query is consistent
            results = cache.get_range(center, radius)
            result_dict = {c: v for c, v in results}
            
            # Check completeness
            for c, v in coords_to_value.items():
                distance = self._euclidean_distance(c, center)
                if distance <= radius:
                    assert c in result_dict, \
                        f"Missing {c} after insertion {i+1}"
                    assert result_dict[c] == v, \
                        f"Wrong value for {c} after insertion {i+1}"
            
            # Check correctness
            for c, v in results:
                distance = self._euclidean_distance(c, center)
                assert distance <= radius, \
                    f"Coordinate {c} outside radius after insertion {i+1}"
