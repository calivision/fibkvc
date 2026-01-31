# Copyright (c) 2026 California Vision, Inc. - David Anderson
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for HierarchicalLatticeCache.

Tests cover:
- Initialization with multiple levels
- Set and get at different resolution levels
- Coordinate scaling between levels
- Invalid level handling
- Level dimension queries
- Level statistics
"""

import pytest
from fibkvc.lattice.hierarchy import HierarchicalLatticeCache


class TestHierarchicalLatticeCacheInit:
    """Test hierarchical cache initialization."""
    
    def test_init_basic(self):
        """Test basic initialization with valid dimensions."""
        cache = HierarchicalLatticeCache(dimensions=(128, 128, 128), num_levels=3)
        assert cache.num_levels == 3
        assert cache.base_dimensions == (128, 128, 128)
        assert len(cache.levels) == 3
    
    def test_init_single_level(self):
        """Test initialization with single level."""
        cache = HierarchicalLatticeCache(dimensions=(64, 64, 64), num_levels=1)
        assert cache.num_levels == 1
        assert len(cache.levels) == 1
    
    def test_init_many_levels(self):
        """Test initialization with many levels."""
        cache = HierarchicalLatticeCache(dimensions=(256, 256, 256), num_levels=5)
        assert cache.num_levels == 5
        assert len(cache.levels) == 5
    
    def test_init_level_dimensions_scaled(self):
        """Test that level dimensions are correctly scaled."""
        cache = HierarchicalLatticeCache(dimensions=(128, 128, 128), num_levels=4)
        
        # Level 0: full resolution
        assert cache.levels[0].dimensions == (128, 128, 128)
        # Level 1: half resolution
        assert cache.levels[1].dimensions == (64, 64, 64)
        # Level 2: quarter resolution
        assert cache.levels[2].dimensions == (32, 32, 32)
        # Level 3: eighth resolution
        assert cache.levels[3].dimensions == (16, 16, 16)
    
    def test_init_with_cache_kwargs(self):
        """Test initialization with custom cache parameters."""
        cache = HierarchicalLatticeCache(
            dimensions=(128, 128, 128),
            num_levels=2,
            initial_size=512,
            max_load_factor=0.8,
            use_octree=True
        )
        
        # Check that kwargs were passed to level caches
        assert cache.levels[0].table_size == 512
        assert cache.levels[0].max_load_factor == 0.8
        assert cache.levels[0]._octree is not None
    
    def test_init_invalid_dimensions_not_tuple(self):
        """Test that invalid dimensions raise ValueError."""
        with pytest.raises(ValueError, match="must be a tuple"):
            HierarchicalLatticeCache(dimensions=[128, 128, 128], num_levels=3)
    
    def test_init_invalid_dimensions_wrong_length(self):
        """Test that wrong number of dimensions raises ValueError."""
        with pytest.raises(ValueError, match="must be a tuple of 3"):
            HierarchicalLatticeCache(dimensions=(128, 128), num_levels=3)
    
    def test_init_invalid_dimensions_negative(self):
        """Test that negative dimensions raise ValueError."""
        with pytest.raises(ValueError, match="positive integers"):
            HierarchicalLatticeCache(dimensions=(128, -64, 128), num_levels=3)
    
    def test_init_invalid_num_levels_zero(self):
        """Test that num_levels=0 raises ValueError."""
        with pytest.raises(ValueError, match="positive integer"):
            HierarchicalLatticeCache(dimensions=(128, 128, 128), num_levels=0)
    
    def test_init_invalid_num_levels_negative(self):
        """Test that negative num_levels raises ValueError."""
        with pytest.raises(ValueError, match="positive integer"):
            HierarchicalLatticeCache(dimensions=(128, 128, 128), num_levels=-1)
    
    def test_init_too_many_levels_for_dimensions(self):
        """Test that too many levels for small dimensions raises ValueError."""
        # 8x8x8 can only support 3 levels (8, 4, 2) before dimensions become < 1
        with pytest.raises(ValueError, match="invalid dimensions"):
            HierarchicalLatticeCache(dimensions=(8, 8, 8), num_levels=5)


class TestHierarchicalLatticeCacheSetGet:
    """Test set and get operations at different levels."""
    
    def test_set_and_get_level_0(self):
        """Test setting and getting at full resolution (level 0)."""
        cache = HierarchicalLatticeCache(dimensions=(128, 128, 128), num_levels=3)
        cache.set((64, 64, 64), "value_level_0", level=0)
        
        result = cache.get_hierarchical((64, 64, 64), level=0)
        assert result == "value_level_0"
    
    def test_set_and_get_level_1(self):
        """Test setting and getting at half resolution (level 1)."""
        cache = HierarchicalLatticeCache(dimensions=(128, 128, 128), num_levels=3)
        cache.set((64, 64, 64), "value_level_1", level=1)
        
        result = cache.get_hierarchical((64, 64, 64), level=1)
        assert result == "value_level_1"
    
    def test_set_and_get_level_2(self):
        """Test setting and getting at quarter resolution (level 2)."""
        cache = HierarchicalLatticeCache(dimensions=(128, 128, 128), num_levels=3)
        cache.set((64, 64, 64), "value_level_2", level=2)
        
        result = cache.get_hierarchical((64, 64, 64), level=2)
        assert result == "value_level_2"
    
    def test_set_different_values_at_different_levels(self):
        """Test that different levels store independent values."""
        cache = HierarchicalLatticeCache(dimensions=(128, 128, 128), num_levels=3)
        
        # Store different values at same coordinates but different levels
        cache.set((64, 64, 64), "fine", level=0)
        cache.set((64, 64, 64), "medium", level=1)
        cache.set((64, 64, 64), "coarse", level=2)
        
        # Verify each level has its own value
        assert cache.get_hierarchical((64, 64, 64), level=0) == "fine"
        assert cache.get_hierarchical((64, 64, 64), level=1) == "medium"
        assert cache.get_hierarchical((64, 64, 64), level=2) == "coarse"
    
    def test_get_nonexistent_returns_none(self):
        """Test that getting non-existent entry returns None."""
        cache = HierarchicalLatticeCache(dimensions=(128, 128, 128), num_levels=3)
        
        result = cache.get_hierarchical((64, 64, 64), level=0)
        assert result is None
    
    def test_set_updates_existing_value(self):
        """Test that setting same coordinates updates the value."""
        cache = HierarchicalLatticeCache(dimensions=(128, 128, 128), num_levels=3)
        
        cache.set((64, 64, 64), "old_value", level=0)
        cache.set((64, 64, 64), "new_value", level=0)
        
        result = cache.get_hierarchical((64, 64, 64), level=0)
        assert result == "new_value"
    
    def test_set_multiple_entries_same_level(self):
        """Test setting multiple entries at the same level."""
        cache = HierarchicalLatticeCache(dimensions=(128, 128, 128), num_levels=3)
        
        cache.set((10, 20, 30), "value1", level=0)
        cache.set((40, 50, 60), "value2", level=0)
        cache.set((70, 80, 90), "value3", level=0)
        
        assert cache.get_hierarchical((10, 20, 30), level=0) == "value1"
        assert cache.get_hierarchical((40, 50, 60), level=0) == "value2"
        assert cache.get_hierarchical((70, 80, 90), level=0) == "value3"


class TestHierarchicalLatticeCacheCoordinateScaling:
    """Test coordinate scaling between levels."""
    
    def test_coordinate_scaling_level_1(self):
        """Test that coordinates are correctly scaled at level 1."""
        cache = HierarchicalLatticeCache(dimensions=(128, 128, 128), num_levels=3)
        
        # Coordinates (64, 64, 64) at level 1 should map to (32, 32, 32)
        cache.set((64, 64, 64), "value", level=1)
        
        # Coordinates (65, 65, 65) at level 1 should also map to (32, 32, 32)
        # due to integer division
        cache.set((65, 65, 65), "updated_value", level=1)
        
        # Both should retrieve the updated value
        assert cache.get_hierarchical((64, 64, 64), level=1) == "updated_value"
        assert cache.get_hierarchical((65, 65, 65), level=1) == "updated_value"
    
    def test_coordinate_scaling_level_2(self):
        """Test that coordinates are correctly scaled at level 2."""
        cache = HierarchicalLatticeCache(dimensions=(128, 128, 128), num_levels=3)
        
        # Coordinates (64, 64, 64) at level 2 should map to (16, 16, 16)
        cache.set((64, 64, 64), "value", level=2)
        
        # Coordinates in range [64-67, 64-67, 64-67] should all map to (16, 16, 16)
        assert cache.get_hierarchical((64, 64, 64), level=2) == "value"
        assert cache.get_hierarchical((65, 65, 65), level=2) == "value"
        assert cache.get_hierarchical((66, 66, 66), level=2) == "value"
        assert cache.get_hierarchical((67, 67, 67), level=2) == "value"
    
    def test_coordinate_scaling_preserves_independence(self):
        """Test that coordinate scaling doesn't affect other levels."""
        cache = HierarchicalLatticeCache(dimensions=(128, 128, 128), num_levels=3)
        
        # Set values at different levels with coordinates that scale differently
        cache.set((64, 64, 64), "level0_value", level=0)
        cache.set((64, 64, 64), "level1_value", level=1)
        
        # Level 0 should have its own value
        assert cache.get_hierarchical((64, 64, 64), level=0) == "level0_value"
        # Level 1 should have its own value
        assert cache.get_hierarchical((64, 64, 64), level=1) == "level1_value"
        
        # Coordinates that map to same scaled coords at level 1
        # should not affect level 0
        cache.set((65, 65, 65), "level0_different", level=0)
        assert cache.get_hierarchical((65, 65, 65), level=0) == "level0_different"
        assert cache.get_hierarchical((64, 64, 64), level=0) == "level0_value"


class TestHierarchicalLatticeCacheInvalidLevel:
    """Test handling of invalid level parameters."""
    
    def test_set_invalid_level_negative(self):
        """Test that negative level raises ValueError."""
        cache = HierarchicalLatticeCache(dimensions=(128, 128, 128), num_levels=3)
        
        with pytest.raises(ValueError, match="Invalid level"):
            cache.set((64, 64, 64), "value", level=-1)
    
    def test_set_invalid_level_too_high(self):
        """Test that level >= num_levels raises ValueError."""
        cache = HierarchicalLatticeCache(dimensions=(128, 128, 128), num_levels=3)
        
        with pytest.raises(ValueError, match="Invalid level"):
            cache.set((64, 64, 64), "value", level=3)
    
    def test_get_invalid_level_negative(self):
        """Test that negative level raises ValueError."""
        cache = HierarchicalLatticeCache(dimensions=(128, 128, 128), num_levels=3)
        
        with pytest.raises(ValueError, match="Invalid level"):
            cache.get_hierarchical((64, 64, 64), level=-1)
    
    def test_get_invalid_level_too_high(self):
        """Test that level >= num_levels raises ValueError."""
        cache = HierarchicalLatticeCache(dimensions=(128, 128, 128), num_levels=3)
        
        with pytest.raises(ValueError, match="Invalid level"):
            cache.get_hierarchical((64, 64, 64), level=3)
    
    def test_get_level_dimensions_invalid(self):
        """Test that invalid level for get_level_dimensions raises ValueError."""
        cache = HierarchicalLatticeCache(dimensions=(128, 128, 128), num_levels=3)
        
        with pytest.raises(ValueError, match="Invalid level"):
            cache.get_level_dimensions(5)
    
    def test_get_level_stats_invalid(self):
        """Test that invalid level for get_level_stats raises ValueError."""
        cache = HierarchicalLatticeCache(dimensions=(128, 128, 128), num_levels=3)
        
        with pytest.raises(ValueError, match="Invalid level"):
            cache.get_level_stats(-1)


class TestHierarchicalLatticeCacheHelperMethods:
    """Test helper methods for querying level information."""
    
    def test_get_level_dimensions(self):
        """Test getting dimensions for each level."""
        cache = HierarchicalLatticeCache(dimensions=(128, 128, 128), num_levels=4)
        
        assert cache.get_level_dimensions(0) == (128, 128, 128)
        assert cache.get_level_dimensions(1) == (64, 64, 64)
        assert cache.get_level_dimensions(2) == (32, 32, 32)
        assert cache.get_level_dimensions(3) == (16, 16, 16)
    
    def test_get_level_stats(self):
        """Test getting statistics for a level."""
        cache = HierarchicalLatticeCache(dimensions=(128, 128, 128), num_levels=3)
        
        # Add some entries
        cache.set((64, 64, 64), "value1", level=0)
        cache.set((32, 32, 32), "value2", level=0)
        
        stats = cache.get_level_stats(0)
        
        # Check that stats dictionary has expected keys
        assert 'load_factor' in stats
        assert 'total_collisions' in stats
        assert 'avg_chain_length' in stats
        assert 'max_chain_length' in stats
        
        # Load factor should be > 0 since we added entries
        assert stats['load_factor'] > 0
    
    def test_get_level_stats_empty_cache(self):
        """Test getting statistics for empty level."""
        cache = HierarchicalLatticeCache(dimensions=(128, 128, 128), num_levels=3)
        
        stats = cache.get_level_stats(0)
        
        # Empty cache should have zero load factor
        assert stats['load_factor'] == 0.0
        assert stats['total_collisions'] == 0


class TestHierarchicalLatticeCacheEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_coordinates_at_boundary(self):
        """Test coordinates at dimension boundaries."""
        cache = HierarchicalLatticeCache(dimensions=(128, 128, 128), num_levels=3)
        
        # Test at origin
        cache.set((0, 0, 0), "origin", level=0)
        assert cache.get_hierarchical((0, 0, 0), level=0) == "origin"
        
        # Test at max coordinates
        cache.set((127, 127, 127), "max", level=0)
        assert cache.get_hierarchical((127, 127, 127), level=0) == "max"
    
    def test_non_power_of_2_dimensions(self):
        """Test with dimensions that are not powers of 2."""
        cache = HierarchicalLatticeCache(dimensions=(100, 100, 100), num_levels=3)
        
        # Level 0: 100x100x100
        assert cache.get_level_dimensions(0) == (100, 100, 100)
        # Level 1: 50x50x50
        assert cache.get_level_dimensions(1) == (50, 50, 50)
        # Level 2: 25x25x25
        assert cache.get_level_dimensions(2) == (25, 25, 25)
        
        # Should be able to set and get values
        cache.set((50, 50, 50), "value", level=0)
        assert cache.get_hierarchical((50, 50, 50), level=0) == "value"
    
    def test_asymmetric_dimensions(self):
        """Test with asymmetric dimensions."""
        cache = HierarchicalLatticeCache(dimensions=(128, 64, 32), num_levels=3)
        
        assert cache.get_level_dimensions(0) == (128, 64, 32)
        assert cache.get_level_dimensions(1) == (64, 32, 16)
        assert cache.get_level_dimensions(2) == (32, 16, 8)
        
        cache.set((64, 32, 16), "value", level=0)
        assert cache.get_hierarchical((64, 32, 16), level=0) == "value"
