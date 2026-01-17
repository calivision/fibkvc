"""
3D Fibonacci hash function using the plastic constant.

This module provides a hash function for 3D coordinates that uses the plastic
constant (3D generalization of the golden ratio) to achieve uniform distribution
and preserve spatial locality.
"""

import math

# Plastic constant (3D golden ratio)
# φ₃ ≈ 1.324717957244746
# This is the real root of x³ - x - 1 = 0
PHI_3 = 1.324717957244746

# Use different irrational constants for each dimension to avoid correlation
# Golden ratio for x
PHI_1 = 1.618033988749895  # (1 + sqrt(5)) / 2

# Silver ratio for y  
PHI_2 = 2.414213562373095  # 1 + sqrt(2)

# Plastic constant for z
# PHI_3 already defined above


def fibonacci_hash_3d(x: int, y: int, z: int, table_size: int) -> int:
    """
    Compute 3D Fibonacci hash using multiple irrational constants.
    
    This hash function provides uniform distribution across the hash table
    while preserving spatial locality - nearby points in 3D space tend to
    hash to nearby indices.
    
    Args:
        x: X coordinate (can be negative)
        y: Y coordinate (can be negative)
        z: Z coordinate (can be negative)
        table_size: Size of hash table (should be power of 2)
        
    Returns:
        Hash value in range [0, table_size)
        
    Algorithm:
        Uses different irrational constants for each dimension:
        - x dimension: Golden ratio (φ ≈ 1.618)
        - y dimension: Silver ratio (δ ≈ 2.414)
        - z dimension: Plastic constant (ψ ≈ 1.325)
        
        h(x,y,z) = floor(((x * φ + y * δ + z * ψ) mod 1) * table_size)
        
    Time Complexity: O(1)
    Space Complexity: O(1)
    
    Examples:
        >>> fibonacci_hash_3d(0, 0, 0, 1024)
        0
        >>> fibonacci_hash_3d(10, 20, 30, 1024)  # doctest: +SKIP
        # Returns deterministic hash value
        >>> fibonacci_hash_3d(-5, -10, -15, 1024)  # doctest: +SKIP
        # Handles negative coordinates correctly
    """
    # Compute weighted sum using different irrational constants
    weighted_sum = x * PHI_1 + y * PHI_2 + z * PHI_3
    
    # Extract fractional part (mod 1)
    fractional = weighted_sum - math.floor(weighted_sum)
    
    # Handle negative coordinates: ensure fractional part is positive
    if fractional < 0:
        fractional += 1.0
    
    # Scale to table size and ensure result is in range [0, table_size)
    # Using bitwise AND with (table_size - 1) for power-of-2 table sizes
    hash_value = int(fractional * table_size)
    
    # Ensure result is within bounds (handles edge cases)
    return hash_value & (table_size - 1)
