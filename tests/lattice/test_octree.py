# Copyright (c) 2026 California Vision, Inc. - David Anderson
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for Octree spatial index.

Tests cover:
- OctreeNode creation and splitting
- Octree insertion
- Range queries with various radii
- Edge cases (empty octree, no results)
"""

import pytest
from fibkvc.lattice.octree import Octree, OctreeNode


class TestOctreeNode:
    """Test OctreeNode functionality."""
    
    def test_node_creation(self):
        """Test OctreeNode can be created with bounds."""
        node = OctreeNode(
            bounds=((0, 0, 0), (100, 100, 100)),
            entries=[],
            children=None
        )
        assert node.bounds == ((0, 0, 0), (100, 100, 100))
        assert node.entries == []
        assert node.children is None
        assert node.is_leaf()
    
    def test_is_leaf(self):
        """Test is_leaf() returns correct value."""
        node = OctreeNode(
            bounds=((0, 0, 0), (100, 100, 100)),
            entries=[],
            children=None
        )
        assert node.is_leaf()
        
        # After splitting, should not be leaf
        node.split()
        assert not node.is_leaf()
    
    def test_contains_within_bounds(self):
        """Test contains() returns True for coordinates within bounds."""
        node = OctreeNode(
            bounds=((0, 0, 0), (100, 100, 100)),
            entries=[],
            children=None
        )
        assert node.contains((0, 0, 0))
        assert node.contains((50, 50, 50))
        assert node.contains((99, 99, 99))
    
    def test_contains_outside_bounds(self):
        """Test contains() returns False for coordinates outside bounds."""
        node = OctreeNode(
            bounds=((0, 0, 0), (100, 100, 100)),
            entries=[],
            children=None
        )
        assert not node.contains((-1, 0, 0))
        assert not node.contains((0, -1, 0))
        assert not node.contains((0, 0, -1))
        assert not node.contains((100, 0, 0))
        assert not node.contains((0, 100, 0))
        assert not node.contains((0, 0, 100))
    
    def test_split_creates_8_children(self):
        """Test split() creates 8 octant children."""
        node = OctreeNode(
            bounds=((0, 0, 0), (100, 100, 100)),
            entries=[],
            children=None
        )
        node.split()
        
        assert not node.is_leaf()
        assert len(node.children) == 8
        
        # Verify each child has correct bounds
        expected_bounds = [
            ((0, 0, 0), (50, 50, 50)),      # low, low, low
            ((50, 0, 0), (100, 50, 50)),    # high, low, low
            ((0, 50, 0), (50, 100, 50)),    # low, high, low
            ((50, 50, 0), (100, 100, 50)),  # high, high, low
            ((0, 0, 50), (50, 50, 100)),    # low, low, high
            ((50, 0, 50), (100, 50, 100)),  # high, low, high
            ((0, 50, 50), (50, 100, 100)),  # low, high, high
            ((50, 50, 50), (100, 100, 100)) # high, high, high
        ]
        
        for i, child in enumerate(node.children):
            assert child.bounds == expected_bounds[i]
    
    def test_split_redistributes_entries(self):
        """Test split() redistributes entries to appropriate children."""
        node = OctreeNode(
            bounds=((0, 0, 0), (100, 100, 100)),
            entries=[
                ((10, 10, 10), "value1"),
                ((60, 60, 60), "value2"),
                ((10, 60, 10), "value3"),
            ],
            children=None
        )
        node.split()
        
        # Parent should have no entries after split
        assert len(node.entries) == 0
        
        # Check entries are in correct children
        # (10, 10, 10) should be in child 0 (low, low, low)
        assert ((10, 10, 10), "value1") in node.children[0].entries
        
        # (60, 60, 60) should be in child 7 (high, high, high)
        assert ((60, 60, 60), "value2") in node.children[7].entries
        
        # (10, 60, 10) should be in child 2 (low, high, low)
        assert ((10, 60, 10), "value3") in node.children[2].entries
    
    def test_split_idempotent(self):
        """Test split() does nothing if already split."""
        node = OctreeNode(
            bounds=((0, 0, 0), (100, 100, 100)),
            entries=[],
            children=None
        )
        node.split()
        children_before = node.children
        
        # Split again should do nothing
        node.split()
        assert node.children is children_before


class TestOctree:
    """Test Octree functionality."""
    
    def test_octree_creation(self):
        """Test Octree can be created with dimensions."""
        octree = Octree(dimensions=(100, 100, 100))
        assert octree.root.bounds == ((0, 0, 0), (100, 100, 100))
        assert octree.root.is_leaf()
    
    def test_insert_single_entry(self):
        """Test inserting a single entry."""
        octree = Octree(dimensions=(100, 100, 100))
        octree.insert((50, 50, 50), "value")
        
        assert len(octree.root.entries) == 1
        assert octree.root.entries[0] == ((50, 50, 50), "value")
    
    def test_insert_multiple_entries(self):
        """Test inserting multiple entries."""
        octree = Octree(dimensions=(100, 100, 100))
        octree.insert((10, 10, 10), "value1")
        octree.insert((20, 20, 20), "value2")
        octree.insert((30, 30, 30), "value3")
        
        # Should still be in root (not enough to trigger split)
        assert len(octree.root.entries) == 3
    
    def test_insert_triggers_split(self):
        """Test inserting many entries triggers node split."""
        octree = Octree(dimensions=(100, 100, 100))
        
        # Insert more than max_entries (default 8)
        for i in range(10):
            octree.insert((i * 10, i * 10, i * 10), f"value{i}")
        
        # Root should have split
        assert not octree.root.is_leaf()
        assert len(octree.root.children) == 8
    
    def test_insert_out_of_bounds_raises_error(self):
        """Test inserting out of bounds coordinates raises ValueError."""
        octree = Octree(dimensions=(100, 100, 100))
        
        with pytest.raises(ValueError, match="out of bounds"):
            octree.insert((100, 50, 50), "value")
        
        with pytest.raises(ValueError, match="out of bounds"):
            octree.insert((-1, 50, 50), "value")
    
    def test_range_query_empty_octree(self):
        """Test range query on empty octree returns empty list."""
        octree = Octree(dimensions=(100, 100, 100))
        results = octree.range_query(center=(50, 50, 50), radius=10.0)
        assert results == []
    
    def test_range_query_single_entry_within_radius(self):
        """Test range query returns entry within radius."""
        octree = Octree(dimensions=(100, 100, 100))
        octree.insert((50, 50, 50), "value")
        
        results = octree.range_query(center=(50, 50, 50), radius=10.0)
        assert len(results) == 1
        assert results[0] == ((50, 50, 50), "value")
    
    def test_range_query_single_entry_outside_radius(self):
        """Test range query excludes entry outside radius."""
        octree = Octree(dimensions=(100, 100, 100))
        octree.insert((50, 50, 50), "value")
        
        results = octree.range_query(center=(80, 80, 80), radius=10.0)
        assert results == []
    
    def test_range_query_multiple_entries(self):
        """Test range query with multiple entries."""
        octree = Octree(dimensions=(100, 100, 100))
        
        # Insert entries at various distances from center (50, 50, 50)
        octree.insert((50, 50, 50), "center")      # distance 0
        octree.insert((55, 50, 50), "near1")       # distance 5
        octree.insert((50, 55, 50), "near2")       # distance 5
        octree.insert((50, 50, 55), "near3")       # distance 5
        octree.insert((70, 70, 70), "far")         # distance ~34.6
        
        # Query with radius 10 should get center and near entries
        results = octree.range_query(center=(50, 50, 50), radius=10.0)
        assert len(results) == 4
        
        coords = [coord for coord, _ in results]
        assert (50, 50, 50) in coords
        assert (55, 50, 50) in coords
        assert (50, 55, 50) in coords
        assert (50, 50, 55) in coords
        assert (70, 70, 70) not in coords
    
    def test_range_query_various_radii(self):
        """Test range query with various radii."""
        octree = Octree(dimensions=(100, 100, 100))
        
        octree.insert((50, 50, 50), "center")
        octree.insert((55, 50, 50), "near")
        octree.insert((70, 70, 70), "far")
        
        # Small radius
        results = octree.range_query(center=(50, 50, 50), radius=3.0)
        assert len(results) == 1
        
        # Medium radius
        results = octree.range_query(center=(50, 50, 50), radius=10.0)
        assert len(results) == 2
        
        # Large radius
        results = octree.range_query(center=(50, 50, 50), radius=50.0)
        assert len(results) == 3
    
    def test_range_query_no_results(self):
        """Test range query with no results in range."""
        octree = Octree(dimensions=(100, 100, 100))
        
        octree.insert((10, 10, 10), "value1")
        octree.insert((20, 20, 20), "value2")
        
        # Query far from entries
        results = octree.range_query(center=(90, 90, 90), radius=5.0)
        assert results == []
    
    def test_distance_calculation(self):
        """Test _distance() static method."""
        assert Octree._distance((0, 0, 0), (0, 0, 0)) == 0.0
        assert Octree._distance((0, 0, 0), (3, 4, 0)) == 5.0
        assert Octree._distance((0, 0, 0), (1, 1, 1)) == pytest.approx(1.732, rel=0.01)
    
    def test_bounds_intersect_sphere(self):
        """Test _bounds_intersect_sphere() static method."""
        bounds = ((0, 0, 0), (10, 10, 10))
        
        # Center inside box
        assert Octree._bounds_intersect_sphere(bounds, (5, 5, 5), 1.0)
        
        # Center outside but sphere intersects
        assert Octree._bounds_intersect_sphere(bounds, (12, 5, 5), 3.0)
        
        # Center outside and sphere doesn't intersect
        assert not Octree._bounds_intersect_sphere(bounds, (20, 20, 20), 5.0)
        
        # Sphere contains entire box
        assert Octree._bounds_intersect_sphere(bounds, (5, 5, 5), 100.0)

    def test_remove_single_entry(self):
        """Test removing a single entry from octree."""
        octree = Octree(dimensions=(100, 100, 100))
        octree.insert((50, 50, 50), "value")
        
        # Remove the entry
        result = octree.remove((50, 50, 50))
        assert result is True
        
        # Verify it's gone
        results = octree.range_query(center=(50, 50, 50), radius=10.0)
        assert len(results) == 0
    
    def test_remove_nonexistent_entry(self):
        """Test removing an entry that doesn't exist."""
        octree = Octree(dimensions=(100, 100, 100))
        octree.insert((50, 50, 50), "value")
        
        # Try to remove non-existent entry
        result = octree.remove((60, 60, 60))
        assert result is False
        
        # Original entry should still be there
        results = octree.range_query(center=(50, 50, 50), radius=10.0)
        assert len(results) == 1
    
    def test_remove_from_multiple_entries(self):
        """Test removing one entry when multiple exist."""
        octree = Octree(dimensions=(100, 100, 100))
        octree.insert((50, 50, 50), "value1")
        octree.insert((55, 50, 50), "value2")
        octree.insert((50, 55, 50), "value3")
        
        # Remove one entry
        result = octree.remove((55, 50, 50))
        assert result is True
        
        # Verify only that one is gone
        results = octree.range_query(center=(50, 50, 50), radius=10.0)
        assert len(results) == 2
        coords = [coord for coord, _ in results]
        assert (50, 50, 50) in coords
        assert (50, 55, 50) in coords
        assert (55, 50, 50) not in coords
    
    def test_remove_duplicate_coordinates(self):
        """Test removing when duplicate coordinates exist (updates)."""
        octree = Octree(dimensions=(100, 100, 100))
        octree.insert((50, 50, 50), "value1")
        octree.insert((50, 50, 50), "value2")  # Duplicate coords
        
        # Remove should remove all entries with those coords
        result = octree.remove((50, 50, 50))
        assert result is True
        
        # Verify all are gone
        results = octree.range_query(center=(50, 50, 50), radius=10.0)
        assert len(results) == 0
    
    def test_remove_out_of_bounds(self):
        """Test removing coordinates outside octree bounds."""
        octree = Octree(dimensions=(100, 100, 100))
        octree.insert((50, 50, 50), "value")
        
        # Try to remove out of bounds
        result = octree.remove((150, 150, 150))
        assert result is False
        
        # Original entry should still be there
        results = octree.range_query(center=(50, 50, 50), radius=10.0)
        assert len(results) == 1
