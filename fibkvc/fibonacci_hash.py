# Copyright (c) 2026 California Vision, Inc. - David Anderson
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Fibonacci hashing implementation for uniform key distribution.

This module provides fibonacci hashing functions that use the golden ratio
to distribute keys uniformly across hash tables, reducing collisions and
improving cache performance.
"""

from typing import Union
import math

# Golden ratio constant: φ = (√5 - 1) / 2
GOLDEN_RATIO = 0.6180339887498948482

# 64-bit golden ratio for integer arithmetic
GOLDEN_RATIO_64 = 11400714819323198549


def is_power_of_two(n: int) -> bool:
    """
    Check if a number is a power of 2.
    
    Args:
        n: Integer to check
        
    Returns:
        True if n is a power of 2, False otherwise
        
    Raises:
        TypeError: If n is not an integer
        ValueError: If n is less than 1
    """
    if not isinstance(n, int):
        raise TypeError(f"Expected int, got {type(n).__name__}")
    
    if n < 1:
        raise ValueError(f"Expected positive integer, got {n}")
    
    # A number is a power of 2 if it has exactly one bit set
    # This is true when n & (n - 1) == 0
    return (n & (n - 1)) == 0


def string_to_int(s: str) -> int:
    """
    Convert a string to an integer for hashing.
    
    Uses a simple polynomial rolling hash to convert strings to integers.
    This ensures consistent integer representation for string keys.
    
    Args:
        s: String to convert
        
    Returns:
        Integer representation of the string
        
    Raises:
        TypeError: If s is not a string
    """
    if not isinstance(s, str):
        raise TypeError(f"Expected str, got {type(s).__name__}")
    
    # Use polynomial rolling hash with prime base
    # This provides good distribution for string keys
    hash_value = 0
    base = 31
    mod = 2**63 - 1  # Large prime for modulo
    
    for char in s:
        hash_value = (hash_value * base + ord(char)) % mod
    
    return hash_value


def fibonacci_hash(key: Union[int, str], table_size: int) -> int:
    """
    Compute a fibonacci hash value for a key.
    
    Uses the golden ratio (φ ≈ 0.618) to distribute keys uniformly
    across the hash table. This multiplicative hashing technique
    provides better distribution than modulo-based hashing, especially
    for non-random key distributions.
    
    Algorithm:
        1. Convert key to integer if string
        2. Multiply by golden ratio
        3. Extract fractional part
        4. Multiply by table_size
        5. Return as integer in range [0, table_size)
    
    Args:
        key: Integer or string key to hash
        table_size: Size of the hash table (must be power of 2)
        
    Returns:
        Hash value in range [0, table_size)
        
    Raises:
        TypeError: If key is not int or str, or table_size is not int
        ValueError: If table_size is not a power of 2 or is less than 1
    """
    # Validate table_size
    if not isinstance(table_size, int):
        raise TypeError(f"table_size must be int, got {type(table_size).__name__}")
    
    if not is_power_of_two(table_size):
        raise ValueError(f"table_size must be a power of 2, got {table_size}")
    
    # Convert key to integer
    if isinstance(key, str):
        key_int = string_to_int(key)
    elif isinstance(key, int):
        key_int = key
    else:
        raise TypeError(f"key must be int or str, got {type(key).__name__}")
    
    # Ensure key_int is positive for consistent hashing
    key_int = abs(key_int)
    
    # Fibonacci hashing: multiply by golden ratio and extract fractional part
    # Using floating point arithmetic for clarity
    fractional_part = (key_int * GOLDEN_RATIO) % 1.0
    
    # Scale to table_size and convert to integer
    hash_value = int(fractional_part * table_size)
    
    # Ensure result is in valid range [0, table_size)
    return hash_value % table_size
