"""
Unit tests for fibonacci_hash_3d function.

Tests cover:
- Origin coordinates (0, 0, 0)
- Corner coordinates
- Negative coordinates
- Various table sizes (powers of 2)
- Determinism (same input → same output)

Validates: Requirements 10.1
"""

import pytest
from fibkvc.lattice import fibonacci_hash_3d, PHI_3


class TestFibonacciHash3D:
    """Unit tests for 3D Fibonacci hash function."""
    
    def test_origin_coordinates(self):
        """Test hashing the origin (0, 0, 0)."""
        # Origin should hash to 0
        result = fibonacci_hash_3d(0, 0, 0, 1024)
        assert result == 0
        assert 0 <= result < 1024
    
    def test_corner_coordinates(self):
        """Test hashing corner coordinates of a cube."""
        table_size = 1024
        
        # Test all 8 corners of a 100x100x100 cube
        corners = [
            (0, 0, 0),
            (99, 0, 0),
            (0, 99, 0),
            (99, 99, 0),
            (0, 0, 99),
            (99, 0, 99),
            (0, 99, 99),
            (99, 99, 99),
        ]
        
        for x, y, z in corners:
            result = fibonacci_hash_3d(x, y, z, table_size)
            assert 0 <= result < table_size, \
                f"Hash for ({x}, {y}, {z}) out of range: {result}"
    
    def test_negative_coordinates(self):
        """Test hashing negative coordinates."""
        table_size = 1024
        
        # Test various negative coordinates
        negative_coords = [
            (-1, -1, -1),
            (-10, -20, -30),
            (-100, 0, 0),
            (0, -100, 0),
            (0, 0, -100),
            (-50, -50, -50),
        ]
        
        for x, y, z in negative_coords:
            result = fibonacci_hash_3d(x, y, z, table_size)
            assert 0 <= result < table_size, \
                f"Hash for ({x}, {y}, {z}) out of range: {result}"
    
    def test_mixed_positive_negative(self):
        """Test hashing mixed positive and negative coordinates."""
        table_size = 512
        
        mixed_coords = [
            (10, -20, 30),
            (-10, 20, -30),
            (50, -50, 0),
            (0, 100, -100),
        ]
        
        for x, y, z in mixed_coords:
            result = fibonacci_hash_3d(x, y, z, table_size)
            assert 0 <= result < table_size
    
    def test_various_table_sizes(self):
        """Test with various table sizes (powers of 2)."""
        coords = (42, 17, 89)
        
        # Test powers of 2 from 2^4 to 2^16
        for power in range(4, 17):
            table_size = 2 ** power
            result = fibonacci_hash_3d(*coords, table_size)
            assert 0 <= result < table_size, \
                f"Hash out of range for table_size={table_size}: {result}"
    
    def test_determinism(self):
        """Test that same input produces same output (determinism)."""
        table_size = 1024
        
        test_coords = [
            (0, 0, 0),
            (10, 20, 30),
            (-5, -10, -15),
            (100, 200, 300),
        ]
        
        for coords in test_coords:
            # Hash the same coordinates multiple times
            results = [fibonacci_hash_3d(*coords, table_size) for _ in range(10)]
            
            # All results should be identical
            assert len(set(results)) == 1, \
                f"Non-deterministic hashing for {coords}: {results}"
    
    def test_different_coords_different_hashes(self):
        """Test that different coordinates produce different hashes (usually)."""
        table_size = 1024
        
        # Generate several different coordinates
        coords_list = [
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 1, 1),
            (10, 20, 30),
            (30, 20, 10),
        ]
        
        hashes = [fibonacci_hash_3d(*coords, table_size) for coords in coords_list]
        
        # Most should be different (some collisions are acceptable)
        # Require at least 70% unique hashes
        unique_hashes = len(set(hashes))
        assert unique_hashes >= len(coords_list) * 0.7, \
            f"Too many collisions: {unique_hashes} unique out of {len(coords_list)}"
    
    def test_plastic_constant_value(self):
        """Test that PHI_3 constant has the correct value."""
        # φ₃ is the real root of x³ - x - 1 = 0
        # So φ₃³ should equal φ₃ + 1
        phi_cubed = PHI_3 ** 3
        phi_plus_one = PHI_3 + 1
        
        # Allow small floating point error
        assert abs(phi_cubed - phi_plus_one) < 1e-10, \
            f"PHI_3 doesn't satisfy x³ - x - 1 = 0: {phi_cubed} != {phi_plus_one}"
    
    def test_large_coordinates(self):
        """Test with large coordinate values."""
        table_size = 2048
        
        large_coords = [
            (1000, 2000, 3000),
            (10000, 20000, 30000),
            (-1000, -2000, -3000),
        ]
        
        for x, y, z in large_coords:
            result = fibonacci_hash_3d(x, y, z, table_size)
            assert 0 <= result < table_size
    
    def test_boundary_values(self):
        """Test boundary values for table size."""
        coords = (42, 17, 89)
        
        # Test smallest reasonable table size
        result = fibonacci_hash_3d(*coords, 16)
        assert 0 <= result < 16
        
        # Test larger table size
        result = fibonacci_hash_3d(*coords, 65536)
        assert 0 <= result < 65536
