# Copyright (c) 2026 California Vision, Inc. - David Anderson
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for FibonacciLatticeCache.

Tests cover:
- Initialization with various dimensions
- Set and get operations with single and multiple entries
- Update existing entries
- Get non-existent entries
- Out-of-bounds coordinate validation
- Automatic resizing when load factor exceeded
"""

import pytest
from fibkvc.lattice import FibonacciLatticeCache


class TestFibonacciLatticeCacheInit:
    """Test cache initialization."""
    
    def test_init_basic(self):
        """Test basic initialization with valid dimensions."""
        cache = FibonacciLatticeCache(dimensions=(100, 100, 100))
        assert cache.dimensions == (100, 100, 100)
        assert cache.table_size >= 1024  # Should be power of 2
        assert cache.max_load_factor == 0.75
        assert cache._size == 0
    
    def test_init_custom_size(self):
        """Test initialization with custom initial size."""
        cache = FibonacciLatticeCache(
            dimensions=(50, 50, 50),
            initial_size=512
        )
        assert cache.table_size == 512
    
    def test_init_custom_load_factor(self):
        """Test initialization with custom load factor."""
        cache = FibonacciLatticeCache(
            dimensions=(100, 100, 100),
            max_load_factor=0.5
        )
        assert cache.max_load_factor == 0.5
    
    def test_init_rounds_to_power_of_2(self):
        """Test that initial size is rounded to power of 2."""
        cache = FibonacciLatticeCache(
            dimensions=(100, 100, 100),
            initial_size=100
        )
        assert cache.table_size == 128  # Next power of 2
    
    def test_init_invalid_dimensions_not_tuple(self):
        """Test that invalid dimensions raise ValueError."""
        with pytest.raises(ValueError, match="must be a tuple"):
            FibonacciLatticeCache(dimensions=[100, 100, 100])
    
    def test_init_invalid_dimensions_wrong_length(self):
        """Test that wrong number of dimensions raises ValueError."""
        with pytest.raises(ValueError, match="must be a tuple of 3"):
            FibonacciLatticeCache(dimensions=(100, 100))
    
    def test_init_invalid_dimensions_negative(self):
        """Test that negative dimensions raise ValueError."""
        with pytest.raises(ValueError, match="positive integers"):
            FibonacciLatticeCache(dimensions=(100, -50, 100))
    
    def test_init_invalid_dimensions_zero(self):
        """Test that zero dimensions raise ValueError."""
        with pytest.raises(ValueError, match="positive integers"):
            FibonacciLatticeCache(dimensions=(0, 100, 100))
    
    def test_init_invalid_load_factor_too_low(self):
        """Test that load factor <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            FibonacciLatticeCache(
                dimensions=(100, 100, 100),
                max_load_factor=0.0
            )
    
    def test_init_invalid_load_factor_too_high(self):
        """Test that load factor > 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            FibonacciLatticeCache(
                dimensions=(100, 100, 100),
                max_load_factor=1.5
            )


class TestFibonacciLatticeCacheSetGet:
    """Test set and get operations."""
    
    def test_set_and_get_single_entry(self):
        """Test setting and getting a single entry."""
        cache = FibonacciLatticeCache(dimensions=(100, 100, 100))
        cache.set((10, 20, 30), "value1")
        assert cache.get((10, 20, 30)) == "value1"
    
    def test_set_and_get_multiple_entries(self):
        """Test setting and getting multiple entries."""
        cache = FibonacciLatticeCache(dimensions=(100, 100, 100))
        
        # Insert multiple entries
        cache.set((10, 20, 30), "value1")
        cache.set((40, 50, 60), "value2")
        cache.set((70, 80, 90), "value3")
        
        # Verify all can be retrieved
        assert cache.get((10, 20, 30)) == "value1"
        assert cache.get((40, 50, 60)) == "value2"
        assert cache.get((70, 80, 90)) == "value3"
    
    def test_update_existing_entry(self):
        """Test updating an existing entry."""
        cache = FibonacciLatticeCache(dimensions=(100, 100, 100))
        
        # Insert initial value
        cache.set((10, 20, 30), "value1")
        assert cache.get((10, 20, 30)) == "value1"
        
        # Update with new value
        cache.set((10, 20, 30), "value2")
        assert cache.get((10, 20, 30)) == "value2"
    
    def test_get_nonexistent_entry(self):
        """Test that getting non-existent entry returns None."""
        cache = FibonacciLatticeCache(dimensions=(100, 100, 100))
        assert cache.get((10, 20, 30)) is None
    
    def test_get_after_other_insertions(self):
        """Test that non-existent entry returns None even after other insertions."""
        cache = FibonacciLatticeCache(dimensions=(100, 100, 100))
        cache.set((10, 20, 30), "value1")
        cache.set((40, 50, 60), "value2")
        
        # Query non-existent coordinate
        assert cache.get((99, 99, 99)) is None
    
    def test_set_with_various_value_types(self):
        """Test setting values of different types."""
        cache = FibonacciLatticeCache(dimensions=(100, 100, 100))
        
        # String
        cache.set((10, 20, 30), "string_value")
        assert cache.get((10, 20, 30)) == "string_value"
        
        # Integer
        cache.set((20, 30, 40), 42)
        assert cache.get((20, 30, 40)) == 42
        
        # List
        cache.set((30, 40, 50), [1, 2, 3])
        assert cache.get((30, 40, 50)) == [1, 2, 3]
        
        # Dict
        cache.set((40, 50, 60), {"key": "value"})
        assert cache.get((40, 50, 60)) == {"key": "value"}
        
        # None
        cache.set((50, 60, 70), None)
        assert cache.get((50, 60, 70)) is None


class TestFibonacciLatticeCacheValidation:
    """Test coordinate validation."""
    
    def test_set_out_of_bounds_x_negative(self):
        """Test that negative x coordinate raises ValueError."""
        cache = FibonacciLatticeCache(dimensions=(100, 100, 100))
        with pytest.raises(ValueError, match="out of bounds"):
            cache.set((-1, 20, 30), "value")
    
    def test_set_out_of_bounds_x_too_large(self):
        """Test that x >= width raises ValueError."""
        cache = FibonacciLatticeCache(dimensions=(100, 100, 100))
        with pytest.raises(ValueError, match="out of bounds"):
            cache.set((100, 20, 30), "value")
    
    def test_set_out_of_bounds_y(self):
        """Test that out of bounds y raises ValueError."""
        cache = FibonacciLatticeCache(dimensions=(100, 100, 100))
        with pytest.raises(ValueError, match="out of bounds"):
            cache.set((10, 100, 30), "value")
    
    def test_set_out_of_bounds_z(self):
        """Test that out of bounds z raises ValueError."""
        cache = FibonacciLatticeCache(dimensions=(100, 100, 100))
        with pytest.raises(ValueError, match="out of bounds"):
            cache.set((10, 20, 100), "value")
    
    def test_get_out_of_bounds(self):
        """Test that get with out of bounds coordinates raises ValueError."""
        cache = FibonacciLatticeCache(dimensions=(100, 100, 100))
        with pytest.raises(ValueError, match="out of bounds"):
            cache.get((100, 100, 100))
    
    def test_set_invalid_coords_not_tuple(self):
        """Test that non-tuple coordinates raise TypeError."""
        cache = FibonacciLatticeCache(dimensions=(100, 100, 100))
        with pytest.raises(TypeError, match="must be a tuple"):
            cache.set([10, 20, 30], "value")
    
    def test_set_invalid_coords_wrong_length(self):
        """Test that wrong number of coordinates raises TypeError."""
        cache = FibonacciLatticeCache(dimensions=(100, 100, 100))
        with pytest.raises(TypeError, match="must be a tuple of 3"):
            cache.set((10, 20), "value")
    
    def test_set_invalid_coords_non_integer(self):
        """Test that non-integer coordinates raise TypeError."""
        cache = FibonacciLatticeCache(dimensions=(100, 100, 100))
        with pytest.raises(TypeError, match="must be integers"):
            cache.set((10.5, 20, 30), "value")
    
    def test_boundary_coordinates(self):
        """Test that boundary coordinates work correctly."""
        cache = FibonacciLatticeCache(dimensions=(100, 100, 100))
        
        # Test all corners
        cache.set((0, 0, 0), "origin")
        cache.set((99, 0, 0), "x_max")
        cache.set((0, 99, 0), "y_max")
        cache.set((0, 0, 99), "z_max")
        cache.set((99, 99, 99), "corner")
        
        assert cache.get((0, 0, 0)) == "origin"
        assert cache.get((99, 0, 0)) == "x_max"
        assert cache.get((0, 99, 0)) == "y_max"
        assert cache.get((0, 0, 99)) == "z_max"
        assert cache.get((99, 99, 99)) == "corner"


class TestFibonacciLatticeCacheResize:
    """Test automatic resizing."""
    
    def test_resize_when_load_factor_exceeded(self):
        """Test that cache resizes when load factor is exceeded."""
        # Use small initial size and low load factor to trigger resize
        cache = FibonacciLatticeCache(
            dimensions=(100, 100, 100),
            initial_size=8,
            max_load_factor=0.5
        )
        
        initial_size = cache.table_size
        assert initial_size == 8
        
        # Insert enough entries to exceed load factor (0.5 * 8 = 4)
        for i in range(5):
            cache.set((i, i, i), f"value{i}")
        
        # Table should have resized
        assert cache.table_size > initial_size
        
        # All values should still be retrievable
        for i in range(5):
            assert cache.get((i, i, i)) == f"value{i}"
    
    def test_resize_preserves_all_entries(self):
        """Test that resize preserves all existing entries."""
        cache = FibonacciLatticeCache(
            dimensions=(100, 100, 100),
            initial_size=16,
            max_load_factor=0.5
        )
        
        # Insert many entries to trigger multiple resizes
        entries = []
        for i in range(20):
            coords = (i, i * 2, i * 3)
            value = f"value{i}"
            entries.append((coords, value))
            cache.set(coords, value)
        
        # Verify all entries are still retrievable
        for coords, value in entries:
            assert cache.get(coords) == value
    
    def test_resize_doubles_table_size(self):
        """Test that resize doubles the table size."""
        cache = FibonacciLatticeCache(
            dimensions=(100, 100, 100),
            initial_size=8,
            max_load_factor=0.5
        )
        
        initial_size = cache.table_size
        
        # Trigger resize
        for i in range(5):
            cache.set((i, i, i), f"value{i}")
        
        # Table size should be doubled
        assert cache.table_size == initial_size * 2


class TestFibonacciLatticeCacheHelpers:
    """Test helper methods."""
    
    def test_next_power_of_2(self):
        """Test _next_power_of_2 helper."""
        assert FibonacciLatticeCache._next_power_of_2(1) == 1
        assert FibonacciLatticeCache._next_power_of_2(2) == 2
        assert FibonacciLatticeCache._next_power_of_2(3) == 4
        assert FibonacciLatticeCache._next_power_of_2(100) == 128
        assert FibonacciLatticeCache._next_power_of_2(1024) == 1024
        assert FibonacciLatticeCache._next_power_of_2(1025) == 2048
    
    def test_next_power_of_2_edge_cases(self):
        """Test _next_power_of_2 with edge cases."""
        assert FibonacciLatticeCache._next_power_of_2(0) == 1
        assert FibonacciLatticeCache._next_power_of_2(-5) == 1
