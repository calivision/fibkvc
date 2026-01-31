# Copyright (c) 2026 California Vision, Inc. - David Anderson
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
HierarchicalLatticeCache: Multi-resolution lattice cache for coarse-to-fine algorithms.

This module provides a hierarchical cache that maintains multiple resolution levels,
enabling efficient coarse-to-fine algorithms commonly used in diffusion models and
other multi-scale applications.
"""

from typing import List, Tuple, Optional, Any
from .lattice_cache import FibonacciLatticeCache, Coords3D


class HierarchicalLatticeCache:
    """
    Multi-resolution lattice cache for coarse-to-fine algorithms.
    
    Maintains cache at multiple resolution levels:
    - Level 0: Full resolution
    - Level 1: Half resolution (dimensions / 2)
    - Level 2: Quarter resolution (dimensions / 4)
    - etc.
    
    Each level uses a separate FibonacciLatticeCache with appropriately scaled
    dimensions. This enables efficient multi-scale processing where coarse levels
    can be computed first and refined at finer levels.
    
    Attributes:
        num_levels: Number of resolution levels
        levels: List of FibonacciLatticeCache instances, one per level
        base_dimensions: Full resolution dimensions (level 0)
    
    Examples:
        >>> cache = HierarchicalLatticeCache(dimensions=(128, 128, 128), num_levels=3)
        >>> # Store at full resolution (level 0)
        >>> cache.set((64, 64, 64), "fine_value", level=0)
        >>> # Store at half resolution (level 1)
        >>> cache.set((64, 64, 64), "coarse_value", level=1)
        >>> # Retrieve from different levels
        >>> cache.get_hierarchical((64, 64, 64), level=0)
        'fine_value'
        >>> cache.get_hierarchical((64, 64, 64), level=1)
        'coarse_value'
    """
    
    def __init__(
        self,
        dimensions: Coords3D,
        num_levels: int = 3,
        **cache_kwargs
    ):
        """
        Initialize hierarchical cache with multiple resolution levels.
        
        Creates a FibonacciLatticeCache for each level, with dimensions scaled
        by powers of 2. Level 0 has full resolution, level 1 has half resolution,
        level 2 has quarter resolution, etc.
        
        Args:
            dimensions: Full resolution dimensions (width, height, depth)
            num_levels: Number of resolution levels (must be >= 1)
            **cache_kwargs: Additional arguments passed to FibonacciLatticeCache
                           (e.g., initial_size, max_load_factor, use_octree)
        
        Raises:
            ValueError: If dimensions are invalid or num_levels < 1
            ValueError: If any level would have dimensions < 1
        
        Examples:
            >>> # Basic hierarchical cache
            >>> cache = HierarchicalLatticeCache(dimensions=(128, 128, 128), num_levels=3)
            
            >>> # With custom cache parameters
            >>> cache = HierarchicalLatticeCache(
            ...     dimensions=(256, 256, 256),
            ...     num_levels=4,
            ...     use_octree=True,
            ...     max_load_factor=0.8
            ... )
        """
        # Validate dimensions
        if not isinstance(dimensions, tuple) or len(dimensions) != 3:
            raise ValueError("dimensions must be a tuple of 3 integers")
        
        w, h, d = dimensions
        if not all(isinstance(dim, int) and dim > 0 for dim in [w, h, d]):
            raise ValueError("All dimensions must be positive integers")
        
        # Validate num_levels
        if not isinstance(num_levels, int) or num_levels < 1:
            raise ValueError("num_levels must be a positive integer")
        
        # Check that all levels will have valid dimensions
        for level in range(num_levels):
            scale = 2 ** level
            level_dims = (w // scale, h // scale, d // scale)
            if any(dim < 1 for dim in level_dims):
                raise ValueError(
                    f"Level {level} would have invalid dimensions {level_dims}. "
                    f"Reduce num_levels or increase base dimensions."
                )
        
        self.num_levels = num_levels
        self.base_dimensions = dimensions
        self.levels: List[FibonacciLatticeCache] = []
        
        # Create cache for each level with scaled dimensions
        for level in range(num_levels):
            scale = 2 ** level
            level_dims = (w // scale, h // scale, d // scale)
            self.levels.append(FibonacciLatticeCache(level_dims, **cache_kwargs))
    
    def set(self, coords: Coords3D, value: Any, level: int = 0) -> None:
        """
        Store value at coordinates in specified resolution level.
        
        Coordinates are provided at full resolution (level 0) and are automatically
        scaled to the appropriate level. For example, coordinates (64, 64, 64) at
        level 1 (half resolution) become (32, 32, 32) in the level 1 cache.
        
        Args:
            coords: Coordinates at full resolution (level 0)
            value: Value to store
            level: Resolution level (0 = full resolution, higher = coarser)
        
        Raises:
            ValueError: If level is out of range [0, num_levels)
            TypeError: If coords is not a tuple of 3 integers
            ValueError: If scaled coordinates are out of bounds for the level
        
        Examples:
            >>> cache = HierarchicalLatticeCache(dimensions=(128, 128, 128), num_levels=3)
            >>> # Store at full resolution
            >>> cache.set((64, 64, 64), "fine", level=0)
            >>> # Store at half resolution (coords scaled to 32, 32, 32)
            >>> cache.set((64, 64, 64), "coarse", level=1)
        """
        # Validate level
        if not 0 <= level < self.num_levels:
            raise ValueError(
                f"Invalid level {level}. Must be in range [0, {self.num_levels})"
            )
        
        # Scale coordinates to level resolution
        scale = 2 ** level
        scaled_coords = (coords[0] // scale, coords[1] // scale, coords[2] // scale)
        
        # Store in the appropriate level cache
        self.levels[level].set(scaled_coords, value)
    
    def get_hierarchical(
        self,
        coords: Coords3D,
        level: int = 0
    ) -> Optional[Any]:
        """
        Retrieve value at coordinates from specified resolution level.
        
        Coordinates are provided at full resolution (level 0) and are automatically
        scaled to the appropriate level before lookup.
        
        Args:
            coords: Coordinates at full resolution (level 0)
            level: Resolution level (0 = full resolution, higher = coarser)
        
        Returns:
            Cached value if found, None otherwise
        
        Raises:
            ValueError: If level is out of range [0, num_levels)
            TypeError: If coords is not a tuple of 3 integers
            ValueError: If scaled coordinates are out of bounds for the level
        
        Examples:
            >>> cache = HierarchicalLatticeCache(dimensions=(128, 128, 128), num_levels=3)
            >>> cache.set((64, 64, 64), "value", level=0)
            >>> cache.get_hierarchical((64, 64, 64), level=0)
            'value'
            >>> cache.get_hierarchical((64, 64, 64), level=1)
            None
        """
        # Validate level
        if not 0 <= level < self.num_levels:
            raise ValueError(
                f"Invalid level {level}. Must be in range [0, {self.num_levels})"
            )
        
        # Scale coordinates to level resolution
        scale = 2 ** level
        scaled_coords = (coords[0] // scale, coords[1] // scale, coords[2] // scale)
        
        # Retrieve from the appropriate level cache
        return self.levels[level].get(scaled_coords)
    
    def get_level_dimensions(self, level: int) -> Coords3D:
        """
        Get the dimensions of a specific resolution level.
        
        Args:
            level: Resolution level
        
        Returns:
            Dimensions (width, height, depth) for the specified level
        
        Raises:
            ValueError: If level is out of range [0, num_levels)
        
        Examples:
            >>> cache = HierarchicalLatticeCache(dimensions=(128, 128, 128), num_levels=3)
            >>> cache.get_level_dimensions(0)
            (128, 128, 128)
            >>> cache.get_level_dimensions(1)
            (64, 64, 64)
            >>> cache.get_level_dimensions(2)
            (32, 32, 32)
        """
        if not 0 <= level < self.num_levels:
            raise ValueError(
                f"Invalid level {level}. Must be in range [0, {self.num_levels})"
            )
        
        return self.levels[level].dimensions
    
    def get_level_stats(self, level: int) -> dict:
        """
        Get collision statistics for a specific resolution level.
        
        Args:
            level: Resolution level
        
        Returns:
            Dictionary with collision statistics from the level's cache
        
        Raises:
            ValueError: If level is out of range [0, num_levels)
        
        Examples:
            >>> cache = HierarchicalLatticeCache(dimensions=(128, 128, 128), num_levels=3)
            >>> cache.set((64, 64, 64), "value", level=0)
            >>> stats = cache.get_level_stats(0)
            >>> 'load_factor' in stats
            True
        """
        if not 0 <= level < self.num_levels:
            raise ValueError(
                f"Invalid level {level}. Must be in range [0, {self.num_levels})"
            )
        
        return self.levels[level].get_collision_stats()
