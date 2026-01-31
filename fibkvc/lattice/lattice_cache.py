# Copyright (c) 2026 California Vision, Inc. - David Anderson
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
FibonacciLatticeCache: 3D spatial cache using Fibonacci hashing.

This module provides a cache class for storing values indexed by 3D coordinates,
using the fibonacci_hash_3d function for efficient spatial hashing with low
collision rates and spatial locality preservation.
"""

from typing import Tuple, Optional, List, Any
from .fibonacci_hash_3d import fibonacci_hash_3d
from .octree import Octree

# Type alias for 3D coordinates
Coords3D = Tuple[int, int, int]


class FibonacciLatticeCache:
    """
    3D spatial cache using Fibonacci hashing.
    
    This cache stores values indexed by 3D coordinates, using Fibonacci hashing
    with the plastic constant for uniform distribution and spatial locality
    preservation. Supports O(1) average-case insertion and lookup.
    
    Optional octree spatial indexing enables efficient range queries with
    O(log n + k) complexity where k is the result size.
    
    Attributes:
        dimensions: (width, height, depth) of lattice space
        table_size: Current hash table size (always power of 2)
        max_load_factor: Threshold for triggering resize
        _size: Number of entries currently stored
        _table: Internal hash table storing entries
        _collision_chain_lengths: Track collision chain lengths for monitoring
        _octree: Optional octree for spatial range queries
    
    Examples:
        >>> cache = FibonacciLatticeCache(dimensions=(100, 100, 100))
        >>> cache.set((10, 20, 30), "value")
        >>> cache.get((10, 20, 30))
        'value'
        >>> cache.get((99, 99, 99))  # Non-existent entry
        None
        
        >>> # With octree for range queries
        >>> cache = FibonacciLatticeCache(dimensions=(100, 100, 100), use_octree=True)
        >>> cache.set((10, 20, 30), "value1")
        >>> cache.set((11, 21, 31), "value2")
        >>> results = cache.get_range((10, 20, 30), radius=5.0)
        >>> len(results)
        2
    """
    
    def __init__(
        self,
        dimensions: Coords3D,
        initial_size: int = 1024,
        max_load_factor: float = 0.75,
        use_octree: bool = False
    ):
        """
        Initialize 3D lattice cache.
        
        Args:
            dimensions: (width, height, depth) of lattice space
            initial_size: Initial hash table size (will be rounded to power of 2)
            max_load_factor: Load factor threshold for resizing (0.0 to 1.0)
            use_octree: Enable spatial indexing for range queries
        
        Raises:
            ValueError: If dimensions are invalid or max_load_factor out of range
        """
        # Validate dimensions
        if not isinstance(dimensions, tuple) or len(dimensions) != 3:
            raise ValueError("dimensions must be a tuple of 3 integers")
        
        w, h, d = dimensions
        if not all(isinstance(dim, int) and dim > 0 for dim in [w, h, d]):
            raise ValueError("All dimensions must be positive integers")
        
        # Validate load factor
        if not 0.0 < max_load_factor <= 1.0:
            raise ValueError("max_load_factor must be between 0.0 and 1.0")
        
        self.dimensions = dimensions
        self.table_size = self._next_power_of_2(initial_size)
        self.max_load_factor = max_load_factor
        self._size = 0
        
        # Hash table: list of optional entries
        # Each entry is a tuple: (coords, value, next_index)
        # next_index is used for collision chaining (None if no collision)
        self._table: List[Optional[Tuple[Coords3D, Any, Optional[int]]]] = \
            [None] * self.table_size
        
        # Track collision chain lengths for monitoring
        self._collision_chain_lengths: List[int] = []
        
        # Octree support for range queries
        self._octree = Octree(dimensions) if use_octree else None
    
    def _validate_coords(self, coords: Coords3D) -> None:
        """
        Validate that coordinates are within lattice dimensions.
        
        Args:
            coords: (x, y, z) coordinates to validate
        
        Raises:
            TypeError: If coords is not a tuple of 3 integers
            ValueError: If coordinates are out of bounds
        """
        if not isinstance(coords, tuple) or len(coords) != 3:
            raise TypeError("coords must be a tuple of 3 integers")
        
        x, y, z = coords
        if not all(isinstance(c, int) for c in [x, y, z]):
            raise TypeError("All coordinates must be integers")
        
        w, h, d = self.dimensions
        if not (0 <= x < w and 0 <= y < h and 0 <= z < d):
            raise ValueError(
                f"Coordinates {coords} out of bounds for dimensions {self.dimensions}"
            )
    
    def _spatial_distance(self, coords1: Coords3D, coords2: Coords3D) -> float:
        """
        Compute Euclidean distance between two 3D coordinates.
        
        Args:
            coords1: First coordinate tuple (x, y, z)
            coords2: Second coordinate tuple (x, y, z)
        
        Returns:
            Euclidean distance between the two points
        """
        x1, y1, z1 = coords1
        x2, y2, z2 = coords2
        return ((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2) ** 0.5
    
    def _get_spatial_probe_sequence(self, coords: Coords3D, hash_val: int, max_probes: int = None) -> List[int]:
        """
        Generate probe sequence prioritizing spatially nearby indices.
        
        This method generates a sequence of table indices to probe when a collision
        occurs. The sequence prioritizes indices that are likely to contain entries
        with spatially nearby coordinates, based on the hash function's spatial
        locality preservation property.
        
        Args:
            coords: The coordinates being inserted
            hash_val: The initial hash value
            max_probes: Maximum number of probe indices to generate (default: table_size)
        
        Returns:
            List of table indices to probe, ordered by spatial proximity
        """
        if max_probes is None:
            max_probes = self.table_size
        
        probe_indices = []
        
        # Start with nearby hash values (spatial locality)
        # The Fibonacci hash preserves spatial locality, so nearby coordinates
        # hash to nearby indices. We probe in expanding rings around the initial hash.
        for offset in range(1, min(max_probes, self.table_size)):
            # Probe both forward and backward
            forward_idx = (hash_val + offset) % self.table_size
            backward_idx = (hash_val - offset) % self.table_size
            
            if forward_idx not in probe_indices:
                probe_indices.append(forward_idx)
            if backward_idx not in probe_indices and len(probe_indices) < max_probes:
                probe_indices.append(backward_idx)
            
            if len(probe_indices) >= max_probes:
                break
        
        return probe_indices
    
    def set(self, coords: Coords3D, value: Any) -> None:
        """
        Store value at 3D coordinates using spatial probing for collision resolution.
        
        Handles both insertions (new key) and updates (existing key).
        Uses spatial probing to handle hash collisions, prioritizing indices
        that are likely to contain spatially nearby coordinates.
        
        Args:
            coords: (x, y, z) coordinates
            value: Value to store
        
        Raises:
            TypeError: If coords is not a tuple of 3 integers
            ValueError: If coordinates are out of bounds
        
        Time Complexity: O(1) average case, O(n) worst case
        
        Examples:
            >>> cache = FibonacciLatticeCache(dimensions=(100, 100, 100))
            >>> cache.set((10, 20, 30), "value1")
            >>> cache.set((10, 20, 30), "value2")  # Update
            >>> cache.get((10, 20, 30))
            'value2'
        """
        # Validate coordinates
        self._validate_coords(coords)
        
        # Compute hash
        hash_val = fibonacci_hash_3d(*coords, self.table_size)
        
        # Check if key already exists (update case)
        current_idx = hash_val
        prev_idx = None
        chain_length = 0
        is_update = False
        
        while current_idx is not None and self._table[current_idx] is not None:
            entry_coords, entry_value, next_idx = self._table[current_idx]
            chain_length += 1
            
            if entry_coords == coords:
                # Update existing entry (keep same chain)
                self._table[current_idx] = (coords, value, next_idx)
                is_update = True
                
                # Update octree if enabled
                if self._octree:
                    # Remove old entry and insert new one
                    self._octree.remove(coords)
                    self._octree.insert(coords, value)
                
                return
            
            prev_idx = current_idx
            current_idx = next_idx
        
        # Insert new entry
        if self._table[hash_val] is None:
            # No collision - insert directly
            self._table[hash_val] = (coords, value, None)
            self._collision_chain_lengths.append(0)
        else:
            # Collision - use spatial probing to find empty slot
            probe_sequence = self._get_spatial_probe_sequence(coords, hash_val)
            probe_idx = None
            
            for idx in probe_sequence:
                if self._table[idx] is None:
                    probe_idx = idx
                    break
            
            if probe_idx is None:
                # Table is full - should not happen with proper load factor
                raise RuntimeError("Hash table is full")
            
            # Insert at empty slot
            self._table[probe_idx] = (coords, value, None)
            
            # Update chain pointer from previous entry
            if prev_idx is not None:
                prev_coords, prev_value, _ = self._table[prev_idx]
                self._table[prev_idx] = (prev_coords, prev_value, probe_idx)
            else:
                # Update the head of the chain
                head_coords, head_value, _ = self._table[hash_val]
                self._table[hash_val] = (head_coords, head_value, probe_idx)
            
            # Track collision chain length
            self._collision_chain_lengths.append(chain_length + 1)
        
        self._size += 1
        
        # Update octree if enabled (only for new insertions)
        if self._octree:
            self._octree.insert(coords, value)
        
        # Check load factor and resize if needed
        if self._size / self.table_size > self.max_load_factor:
            self._resize()
    
    def get(self, coords: Coords3D) -> Optional[Any]:
        """
        Retrieve value at 3D coordinates.
        
        Walks the collision chain to find the entry with matching coordinates.
        
        Args:
            coords: (x, y, z) coordinates
        
        Returns:
            Cached value if found, None otherwise
        
        Raises:
            TypeError: If coords is not a tuple of 3 integers
            ValueError: If coordinates are out of bounds
        
        Time Complexity: O(1) average case, O(n) worst case
        
        Examples:
            >>> cache = FibonacciLatticeCache(dimensions=(100, 100, 100))
            >>> cache.set((10, 20, 30), "value")
            >>> cache.get((10, 20, 30))
            'value'
            >>> cache.get((99, 99, 99))
            None
        """
        # Validate coordinates
        self._validate_coords(coords)
        
        # Compute hash
        hash_val = fibonacci_hash_3d(*coords, self.table_size)
        
        # Walk collision chain to find matching coordinates
        current_idx = hash_val
        
        while current_idx is not None and self._table[current_idx] is not None:
            entry_coords, entry_value, next_idx = self._table[current_idx]
            
            if entry_coords == coords:
                return entry_value
            
            current_idx = next_idx
        
        # Not found
        return None
    
    def get_range(
        self,
        center: Coords3D,
        radius: float
    ) -> List[Tuple[Coords3D, Any]]:
        """
        Query all cached values within radius of center.
        
        Uses the octree spatial index for efficient range queries.
        Returns current values from the cache, not stale octree values.
        
        Args:
            center: Center coordinates (x, y, z)
            radius: Search radius (Euclidean distance)
        
        Returns:
            List of (coords, value) tuples within range
        
        Raises:
            RuntimeError: If octree is not enabled (use_octree=False)
            TypeError: If center is not a tuple of 3 integers
            ValueError: If radius is negative
        
        Time Complexity: O(log n + k) where k is result size
        
        Examples:
            >>> cache = FibonacciLatticeCache(dimensions=(100, 100, 100), use_octree=True)
            >>> cache.set((10, 20, 30), "value1")
            >>> cache.set((11, 21, 31), "value2")
            >>> cache.set((50, 50, 50), "value3")
            >>> results = cache.get_range((10, 20, 30), radius=5.0)
            >>> len(results)
            2
        """
        if not self._octree:
            raise RuntimeError(
                "Range queries require octree support. "
                "Initialize cache with use_octree=True"
            )
        
        # Validate center coordinates
        self._validate_coords(center)
        
        # Validate radius
        if not isinstance(radius, (int, float)) or radius < 0:
            raise ValueError("radius must be a non-negative number")
        
        # Use octree for efficient range query
        octree_results = self._octree.range_query(center, radius)
        
        # Deduplicate and get current values from cache
        # The octree may contain stale values if entries were updated
        seen_coords = set()
        current_results = []
        
        for coords, _ in octree_results:
            if coords not in seen_coords:
                seen_coords.add(coords)
                # Get current value from cache (not stale octree value)
                current_value = self.get(coords)
                if current_value is not None:
                    current_results.append((coords, current_value))
        
        return current_results
    
    def _resize(self) -> None:
        """
        Double hash table size and rehash all entries.
        
        This method is called automatically when the load factor exceeds
        the threshold. All existing entries are rehashed into the new
        larger table. If octree is enabled, it is also rebuilt.
        
        Time Complexity: O(n) where n is number of entries
        Space Complexity: O(n) for new table
        """
        # Save old table
        old_table = self._table
        old_size = self.table_size
        
        # Double table size
        self.table_size *= 2
        self._table = [None] * self.table_size
        self._size = 0
        self._collision_chain_lengths = []
        
        # Rebuild octree if enabled
        if self._octree:
            self._octree = Octree(self.dimensions)
        
        # Rehash all entries from old table
        for entry in old_table:
            if entry is not None:
                coords, value, _ = entry
                # Use set() to insert into new table
                # This will handle collisions and chaining properly
                self.set(coords, value)
    
    @staticmethod
    def _next_power_of_2(n: int) -> int:
        """
        Find the next power of 2 greater than or equal to n.
        
        Args:
            n: Input value
        
        Returns:
            Next power of 2 >= n
        
        Examples:
            >>> FibonacciLatticeCache._next_power_of_2(100)
            128
            >>> FibonacciLatticeCache._next_power_of_2(1024)
            1024
            >>> FibonacciLatticeCache._next_power_of_2(1)
            1
        """
        if n <= 0:
            return 1
        
        power = 1
        while power < n:
            power *= 2
        return power
    
    def get_collision_stats(self) -> dict:
        """
        Get collision statistics for monitoring cache performance.
        
        Returns:
            Dictionary with collision statistics:
            - 'total_collisions': Total number of collisions
            - 'avg_chain_length': Average collision chain length
            - 'max_chain_length': Maximum collision chain length
            - 'load_factor': Current load factor
        
        Examples:
            >>> cache = FibonacciLatticeCache(dimensions=(100, 100, 100))
            >>> cache.set((10, 20, 30), "value1")
            >>> cache.set((11, 21, 31), "value2")
            >>> stats = cache.get_collision_stats()
            >>> stats['load_factor']
            0.001953125
        """
        total_collisions = len([l for l in self._collision_chain_lengths if l > 0])
        avg_chain_length = (
            sum(self._collision_chain_lengths) / len(self._collision_chain_lengths)
            if self._collision_chain_lengths else 0.0
        )
        max_chain_length = max(self._collision_chain_lengths) if self._collision_chain_lengths else 0
        load_factor = self._size / self.table_size
        
        return {
            'total_collisions': total_collisions,
            'avg_chain_length': avg_chain_length,
            'max_chain_length': max_chain_length,
            'load_factor': load_factor
        }
