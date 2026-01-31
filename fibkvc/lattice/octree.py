# Copyright (c) 2026 California Vision, Inc. - David Anderson
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Octree spatial index for efficient 3D range queries.

This module provides an octree data structure for spatial indexing of 3D coordinates,
enabling efficient range queries with O(log n + k) complexity where k is the result size.
"""

from typing import List, Tuple, Optional, Any
from dataclasses import dataclass, field

Coords3D = Tuple[int, int, int]


@dataclass
class OctreeNode:
    """
    Octree node for spatial indexing.
    
    An octree recursively subdivides 3D space into 8 octants (children).
    Each node stores entries within its bounds and can split when it exceeds
    the maximum entry threshold.
    
    Attributes:
        bounds: ((min_x, min_y, min_z), (max_x, max_y, max_z))
        entries: List of (coords, value) stored in this node
        children: 8 child nodes (octants) or None if leaf
        max_entries: Split threshold (default 8)
    """
    bounds: Tuple[Coords3D, Coords3D]
    entries: List[Tuple[Coords3D, Any]] = field(default_factory=list)
    children: Optional[List['OctreeNode']] = None
    max_entries: int = 8
    
    def is_leaf(self) -> bool:
        """
        Check if this node is a leaf (has no children).
        
        Returns:
            True if node has no children, False otherwise
        """
        return self.children is None
    
    def contains(self, coords: Coords3D) -> bool:
        """
        Check if coordinates are within node bounds.
        
        Args:
            coords: (x, y, z) coordinates to check
            
        Returns:
            True if coordinates are within bounds, False otherwise
        """
        (min_x, min_y, min_z), (max_x, max_y, max_z) = self.bounds
        x, y, z = coords
        return min_x <= x < max_x and min_y <= y < max_y and min_z <= z < max_z
    
    def split(self) -> None:
        """
        Split node into 8 octants (children).
        
        Creates 8 child nodes by subdividing the current bounds at the midpoint
        along each axis. Redistributes existing entries to the appropriate children.
        
        Does nothing if node is already split (not a leaf).
        """
        if not self.is_leaf():
            return
        
        (min_x, min_y, min_z), (max_x, max_y, max_z) = self.bounds
        mid_x = (min_x + max_x) // 2
        mid_y = (min_y + max_y) // 2
        mid_z = (min_z + max_z) // 2
        
        # Create 8 octants
        # Order: (x_low/high, y_low/high, z_low/high)
        self.children = [
            # Bottom 4 octants (z_low)
            OctreeNode(((min_x, min_y, min_z), (mid_x, mid_y, mid_z)), [], None, self.max_entries),  # 0: low, low, low
            OctreeNode(((mid_x, min_y, min_z), (max_x, mid_y, mid_z)), [], None, self.max_entries),  # 1: high, low, low
            OctreeNode(((min_x, mid_y, min_z), (mid_x, max_y, mid_z)), [], None, self.max_entries),  # 2: low, high, low
            OctreeNode(((mid_x, mid_y, min_z), (max_x, max_y, mid_z)), [], None, self.max_entries),  # 3: high, high, low
            # Top 4 octants (z_high)
            OctreeNode(((min_x, min_y, mid_z), (mid_x, mid_y, max_z)), [], None, self.max_entries),  # 4: low, low, high
            OctreeNode(((mid_x, min_y, mid_z), (max_x, mid_y, max_z)), [], None, self.max_entries),  # 5: high, low, high
            OctreeNode(((min_x, mid_y, mid_z), (mid_x, max_y, max_z)), [], None, self.max_entries),  # 6: low, high, high
            OctreeNode(((mid_x, mid_y, mid_z), (max_x, max_y, max_z)), [], None, self.max_entries),  # 7: high, high, high
        ]
        
        # Redistribute entries to children
        for coords, value in self.entries:
            for child in self.children:
                if child.contains(coords):
                    child.entries.append((coords, value))
                    break
        
        # Clear entries from parent node
        self.entries = []



class Octree:
    """
    Octree for efficient 3D spatial queries.
    
    An octree is a tree data structure where each internal node has exactly 8 children.
    It recursively subdivides 3D space to enable efficient spatial queries.
    
    Supports:
    - O(log n) insertion
    - O(log n + k) range queries where k is the result size
    
    Attributes:
        root: Root OctreeNode spanning the entire space
    """
    
    def __init__(self, dimensions: Coords3D):
        """
        Initialize octree.
        
        Args:
            dimensions: (width, height, depth) of space
        """
        self.root = OctreeNode(
            bounds=((0, 0, 0), dimensions),
            entries=[],
            children=None
        )
    
    def insert(self, coords: Coords3D, value: Any) -> None:
        """
        Insert value at coordinates.
        
        Args:
            coords: (x, y, z) coordinates
            value: Value to store
            
        Raises:
            ValueError: If coordinates are out of bounds
            
        Time Complexity: O(log n)
        """
        self._insert_recursive(self.root, coords, value)
    
    def remove(self, coords: Coords3D) -> bool:
        """
        Remove entry at coordinates.
        
        Args:
            coords: (x, y, z) coordinates to remove
            
        Returns:
            True if entry was found and removed, False otherwise
            
        Time Complexity: O(log n)
        """
        return self._remove_recursive(self.root, coords)
    
    def _insert_recursive(
        self,
        node: OctreeNode,
        coords: Coords3D,
        value: Any
    ) -> None:
        """
        Recursive insertion helper.
        
        Args:
            node: Current node being processed
            coords: Coordinates to insert
            value: Value to store
            
        Raises:
            ValueError: If coordinates are out of bounds
        """
        if not node.contains(coords):
            raise ValueError(f"Coordinates {coords} out of bounds for node {node.bounds}")
        
        if node.is_leaf():
            # Add entry to leaf node
            node.entries.append((coords, value))
            
            # Split if exceeds threshold
            if len(node.entries) > node.max_entries:
                node.split()
        else:
            # Recurse into appropriate child
            for child in node.children:
                if child.contains(coords):
                    self._insert_recursive(child, coords, value)
                    break
    
    def _remove_recursive(
        self,
        node: OctreeNode,
        coords: Coords3D
    ) -> bool:
        """
        Recursive removal helper.
        
        Args:
            node: Current node being processed
            coords: Coordinates to remove
            
        Returns:
            True if entry was found and removed, False otherwise
        """
        if not node.contains(coords):
            return False
        
        if node.is_leaf():
            # Remove all entries with matching coordinates
            original_len = len(node.entries)
            node.entries = [(c, v) for c, v in node.entries if c != coords]
            return len(node.entries) < original_len
        else:
            # Recurse into appropriate child
            for child in node.children:
                if child.contains(coords):
                    return self._remove_recursive(child, coords)
            return False
    
    @staticmethod
    def _distance(p1: Coords3D, p2: Coords3D) -> float:
        """
        Compute Euclidean distance between two points.
        
        Args:
            p1: First point (x, y, z)
            p2: Second point (x, y, z)
            
        Returns:
            Euclidean distance between p1 and p2
        """
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2) ** 0.5

    
    def range_query(
        self,
        center: Coords3D,
        radius: float
    ) -> List[Tuple[Coords3D, Any]]:
        """
        Query all entries within radius of center.
        
        Args:
            center: Center coordinates (x, y, z)
            radius: Search radius
            
        Returns:
            List of (coords, value) tuples within range
            
        Time Complexity: O(log n + k) where k is result size
        """
        results = []
        self._range_query_recursive(self.root, center, radius, results)
        return results
    
    def _range_query_recursive(
        self,
        node: OctreeNode,
        center: Coords3D,
        radius: float,
        results: List[Tuple[Coords3D, Any]]
    ) -> None:
        """
        Recursive range query helper.
        
        Args:
            node: Current node being processed
            center: Center of query sphere
            radius: Search radius
            results: Accumulator for results (modified in place)
        """
        # Check if node bounds intersect with query sphere
        if not self._bounds_intersect_sphere(node.bounds, center, radius):
            return
        
        if node.is_leaf():
            # Check each entry in leaf node
            for coords, value in node.entries:
                if self._distance(coords, center) <= radius:
                    results.append((coords, value))
        else:
            # Recurse into children
            for child in node.children:
                self._range_query_recursive(child, center, radius, results)
    
    @staticmethod
    def _bounds_intersect_sphere(
        bounds: Tuple[Coords3D, Coords3D],
        center: Coords3D,
        radius: float
    ) -> bool:
        """
        Check if bounding box intersects with sphere.
        
        Uses the closest point in the box to the sphere center to determine
        if the sphere intersects the box.
        
        Args:
            bounds: ((min_x, min_y, min_z), (max_x, max_y, max_z))
            center: Center of sphere (x, y, z)
            radius: Radius of sphere
            
        Returns:
            True if bounding box intersects sphere, False otherwise
        """
        (min_x, min_y, min_z), (max_x, max_y, max_z) = bounds
        cx, cy, cz = center
        
        # Find closest point in box to sphere center
        closest_x = max(min_x, min(cx, max_x))
        closest_y = max(min_y, min(cy, max_y))
        closest_z = max(min_z, min(cz, max_z))
        
        # Check if closest point is within radius
        dist_sq = (closest_x - cx)**2 + (closest_y - cy)**2 + (closest_z - cz)**2
        return dist_sq <= radius * radius
