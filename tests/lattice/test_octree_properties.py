# Copyright (c) 2026 California Vision, Inc. - David Anderson
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Property-based tests for Octree spatial index.

Tests universal properties that should hold for all inputs:
- Property 5: Range Query Completeness and Correctness

Uses Hypothesis for property-based testing with minimum 100 iterations.
"""

import pytest
from hypothesis import given, strategies as st, settings
from fibkvc.lattice.octree import Octree


# Test generators
@st.composite
def coordinates_3d(draw, dimensions=(100, 100, 100)):
    """Generate valid 3D coordinates within dimensions."""
    x = draw(st.integers(min_value=0, max_value=dimensions[0]-1))
    y = draw(st.integers(min_value=0, max_value=dimensions[1]-1))
    z = draw(st.integers(min_value=0, max_value=dimensions[2]-1))
    return (x, y, z)


@st.composite
def octree_with_entries(draw, dimensions=(100, 100, 100), min_entries=5, max_entries=50):
    """Generate an octree with random entries."""
    octree = Octree(dimensions=dimensions)
    num_entries = draw(st.integers(min_value=min_entries, max_value=max_entries))
    
    entries = []
    for i in range(num_entries):
        coords = draw(coordinates_3d(dimensions))
        value = draw(st.integers())
        octree.insert(coords, value)
        entries.append((coords, value))
    
    return octree, entries


def distance(p1, p2):
    """Compute Euclidean distance between two points."""
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2) ** 0.5


class TestOctreeProperties:
    """Property-based tests for Octree."""
    
    @settings(max_examples=100)
    @given(
        octree_data=octree_with_entries(dimensions=(100, 100, 100)),
        center=coordinates_3d(dimensions=(100, 100, 100)),
        radius=st.floats(min_value=1.0, max_value=50.0)
    )
    def test_range_query_completeness_and_correctness(self, octree_data, center, radius):
        """
        Property 5: Range Query Completeness and Correctness
        Feature: fibkvc-3d-lattice-extension, Property 5
        Validates: Requirements 4.2
        
        For any set of cached values and any range query with center and radius,
        the returned results should satisfy:
        1. All returned coordinates are within the specified radius of the center
        2. All cached coordinates within the radius are included in the results
        3. No coordinates outside the radius are included
        """
        octree, entries = octree_data
        
        # Perform range query
        results = octree.range_query(center=center, radius=radius)
        result_coords = {coords for coords, _ in results}
        
        # Compute expected results (ground truth)
        expected_coords = {coords for coords, _ in entries if distance(coords, center) <= radius}
        
        # Property 1: All returned coordinates are within radius
        for coords, _ in results:
            dist = distance(coords, center)
            assert dist <= radius, f"Returned coord {coords} is outside radius {radius} (distance: {dist})"
        
        # Property 2: All cached coordinates within radius are included
        for coords in expected_coords:
            assert coords in result_coords, f"Coord {coords} within radius {radius} was not returned"
        
        # Property 3: No coordinates outside radius are included
        for coords in result_coords:
            dist = distance(coords, center)
            assert dist <= radius, f"Returned coord {coords} is outside radius {radius} (distance: {dist})"
        
        # Verify completeness: result set equals expected set
        assert result_coords == expected_coords, \
            f"Result set mismatch. Expected {len(expected_coords)} coords, got {len(result_coords)}"
    
    @settings(max_examples=100)
    @given(
        dimensions=st.tuples(
            st.integers(min_value=50, max_value=200),
            st.integers(min_value=50, max_value=200),
            st.integers(min_value=50, max_value=200)
        )
    )
    def test_range_query_empty_octree_returns_empty(self, dimensions):
        """
        Property: Range query on empty octree always returns empty list.
        
        For any dimensions and any query parameters, an empty octree
        should return no results.
        """
        octree = Octree(dimensions=dimensions)
        center = (dimensions[0] // 2, dimensions[1] // 2, dimensions[2] // 2)
        radius = 10.0
        
        results = octree.range_query(center=center, radius=radius)
        assert results == []
    
    @settings(max_examples=100)
    @given(
        octree_data=octree_with_entries(dimensions=(100, 100, 100)),
        center=coordinates_3d(dimensions=(100, 100, 100))
    )
    def test_range_query_zero_radius_returns_exact_match_only(self, octree_data, center):
        """
        Property: Range query with radius 0 returns only exact coordinate match.
        
        For any octree and any center, a query with radius 0 should return
        only entries at exactly that coordinate (may be multiple if duplicates exist).
        """
        octree, entries = octree_data
        
        results = octree.range_query(center=center, radius=0.0)
        
        # All results must be at exactly the center
        for coords, _ in results:
            assert coords == center, f"Result {coords} is not at center {center}"
        
        # Count how many entries are at the center
        expected_count = sum(1 for coords, _ in entries if coords == center)
        assert len(results) == expected_count, \
            f"Expected {expected_count} results at center, got {len(results)}"
    
    @settings(max_examples=100)
    @given(
        octree_data=octree_with_entries(dimensions=(100, 100, 100)),
        center=coordinates_3d(dimensions=(100, 100, 100))
    )
    def test_range_query_increasing_radius_monotonic(self, octree_data, center):
        """
        Property: Range query results are monotonic with increasing radius.
        
        For any octree and center, as radius increases, the number of results
        should never decrease (monotonic non-decreasing).
        """
        octree, _ = octree_data
        
        radii = [5.0, 10.0, 20.0, 50.0]
        prev_count = 0
        
        for radius in radii:
            results = octree.range_query(center=center, radius=radius)
            current_count = len(results)
            
            assert current_count >= prev_count, \
                f"Result count decreased from {prev_count} to {current_count} as radius increased"
            
            prev_count = current_count
    
    @settings(max_examples=100)
    @given(
        octree_data=octree_with_entries(dimensions=(100, 100, 100), min_entries=1, max_entries=20)
    )
    def test_range_query_large_radius_returns_all(self, octree_data):
        """
        Property: Range query with very large radius returns all entries.
        
        For any octree, a query with radius larger than the space diagonal
        should return all entries.
        """
        octree, entries = octree_data
        
        # Use radius larger than space diagonal
        radius = 200.0  # Larger than sqrt(100^2 + 100^2 + 100^2) ≈ 173
        center = (50, 50, 50)
        
        results = octree.range_query(center=center, radius=radius)
        
        # Should return all entries
        assert len(results) == len(entries)
