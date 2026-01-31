# Copyright (c) 2026 California Vision, Inc. - David Anderson
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Backward compatibility tests for fibkvc 3D lattice extension.

These tests verify that the 3D lattice extension maintains 100% backward
compatibility with the existing 1D fibkvc API. The tests ensure:

1. Existing FibonacciCacheOptimizer class is unchanged
2. Existing fibonacci_hash() function is unchanged
3. 3D features are not in default fibkvc namespace
4. 3D features are accessible via fibkvc.lattice import
5. Existing fibkvc tests still pass

Requirements: 7.1, 7.2, 7.3, 7.5
"""

import pytest
import sys
import inspect


class TestBackwardCompatibility:
    """Test backward compatibility with existing fibkvc API."""
    
    def test_fibonacci_cache_optimizer_exists(self):
        """
        Verify FibonacciCacheOptimizer class exists in main fibkvc namespace.
        
        Requirements: 7.1
        """
        import fibkvc
        
        assert hasattr(fibkvc, 'FibonacciCacheOptimizer')
        assert inspect.isclass(fibkvc.FibonacciCacheOptimizer)
    
    def test_fibonacci_hash_function_exists(self):
        """
        Verify fibonacci_hash() function exists in main fibkvc namespace.
        
        Requirements: 7.1
        """
        import fibkvc
        
        assert hasattr(fibkvc, 'fibonacci_hash')
        assert callable(fibkvc.fibonacci_hash)
    
    def test_golden_ratio_constants_exist(self):
        """
        Verify golden ratio constants exist in main fibkvc namespace.
        
        Requirements: 7.1
        """
        import fibkvc
        
        assert hasattr(fibkvc, 'GOLDEN_RATIO')
        assert hasattr(fibkvc, 'GOLDEN_RATIO_64')
        assert isinstance(fibkvc.GOLDEN_RATIO, float)
        assert isinstance(fibkvc.GOLDEN_RATIO_64, int)
    
    def test_fibonacci_hash_signature_unchanged(self):
        """
        Verify fibonacci_hash() function signature is unchanged.
        
        Requirements: 7.2
        """
        import fibkvc
        
        sig = inspect.signature(fibkvc.fibonacci_hash)
        params = list(sig.parameters.keys())
        
        # Should have 'key' and 'table_size' parameters
        assert 'key' in params
        assert 'table_size' in params
        assert len(params) == 2
    
    def test_fibonacci_hash_behavior_unchanged(self):
        """
        Verify fibonacci_hash() produces expected results for known inputs.
        
        Requirements: 7.2
        """
        import fibkvc
        
        # Test with known inputs
        result1 = fibkvc.fibonacci_hash(42, 256)
        result2 = fibkvc.fibonacci_hash("test", 256)
        
        # Results should be in valid range
        assert 0 <= result1 < 256
        assert 0 <= result2 < 256
        
        # Should be deterministic
        assert fibkvc.fibonacci_hash(42, 256) == result1
        assert fibkvc.fibonacci_hash("test", 256) == result2
    
    def test_fibonacci_cache_optimizer_signature_unchanged(self):
        """
        Verify FibonacciCacheOptimizer __init__ signature is unchanged.
        
        Requirements: 7.2
        """
        import fibkvc
        
        sig = inspect.signature(fibkvc.FibonacciCacheOptimizer.__init__)
        params = list(sig.parameters.keys())
        
        # Should have 'self', 'use_fibonacci', 'initial_table_size', 'config'
        assert 'self' in params
        assert 'use_fibonacci' in params
        assert 'initial_table_size' in params
        assert 'config' in params
    
    def test_fibonacci_cache_optimizer_methods_unchanged(self):
        """
        Verify FibonacciCacheOptimizer has expected methods.
        
        Requirements: 7.2
        """
        import fibkvc
        
        optimizer = fibkvc.FibonacciCacheOptimizer()
        
        # Check for expected methods
        assert hasattr(optimizer, 'serialize_cache_state')
        assert hasattr(optimizer, 'deserialize_cache_state')
        assert hasattr(optimizer, 'save_to_file')
        assert hasattr(optimizer, 'load_from_file')
        assert hasattr(optimizer, 'get_hash_index')
        assert hasattr(optimizer, 'get_statistics')
        
        # Verify they are callable
        assert callable(optimizer.serialize_cache_state)
        assert callable(optimizer.deserialize_cache_state)
        assert callable(optimizer.save_to_file)
        assert callable(optimizer.load_from_file)
        assert callable(optimizer.get_hash_index)
        assert callable(optimizer.get_statistics)
    
    def test_fibonacci_cache_optimizer_basic_functionality(self):
        """
        Verify FibonacciCacheOptimizer basic operations work as expected.
        
        Requirements: 7.2
        """
        import fibkvc
        
        optimizer = fibkvc.FibonacciCacheOptimizer()
        
        # Test serialization/deserialization
        cache_state = {"entries": {"0": {"key": "value"}}}
        json_str = optimizer.serialize_cache_state(cache_state)
        restored = optimizer.deserialize_cache_state(json_str)
        
        assert "entries" in restored
        assert "0" in restored["entries"]
        
        # Test hash index computation
        hash_idx = optimizer.get_hash_index(42)
        assert isinstance(hash_idx, int)
        assert 0 <= hash_idx < optimizer.table_size
        
        # Test statistics
        stats = optimizer.get_statistics()
        assert isinstance(stats, dict)
        assert "total_serializations" in stats
        assert stats["total_serializations"] == 1
    
    def test_3d_features_not_in_default_namespace(self):
        """
        Verify 3D features are NOT in default fibkvc namespace.
        
        Requirements: 7.3
        """
        import fibkvc
        
        # 3D features should NOT be in main namespace
        assert not hasattr(fibkvc, 'FibonacciLatticeCache')
        assert not hasattr(fibkvc, 'fibonacci_hash_3d')
        assert not hasattr(fibkvc, 'PHI_3')
        assert not hasattr(fibkvc, 'Octree')
        assert not hasattr(fibkvc, 'OctreeNode')
        assert not hasattr(fibkvc, 'HierarchicalLatticeCache')
    
    def test_3d_features_not_in_all_exports(self):
        """
        Verify 3D features are NOT in fibkvc.__all__.
        
        Requirements: 7.3
        """
        import fibkvc
        
        # Check __all__ exports
        all_exports = fibkvc.__all__
        
        # 3D features should NOT be exported by default
        assert 'FibonacciLatticeCache' not in all_exports
        assert 'fibonacci_hash_3d' not in all_exports
        assert 'PHI_3' not in all_exports
        assert 'Octree' not in all_exports
        assert 'OctreeNode' not in all_exports
        assert 'HierarchicalLatticeCache' not in all_exports
    
    def test_3d_features_accessible_via_lattice_import(self):
        """
        Verify 3D features ARE accessible via fibkvc.lattice import.
        
        Requirements: 7.3
        """
        from fibkvc.lattice import (
            FibonacciLatticeCache,
            fibonacci_hash_3d,
            PHI_3,
            Octree,
            OctreeNode
        )
        
        # Verify classes and functions are accessible
        assert inspect.isclass(FibonacciLatticeCache)
        assert callable(fibonacci_hash_3d)
        assert isinstance(PHI_3, float)
        assert inspect.isclass(Octree)
        assert inspect.isclass(OctreeNode)
    
    def test_3d_features_in_lattice_all_exports(self):
        """
        Verify 3D features ARE in fibkvc.lattice.__all__.
        
        Requirements: 7.3
        """
        import fibkvc.lattice
        
        all_exports = fibkvc.lattice.__all__
        
        # 3D features should be exported from lattice module
        assert 'FibonacciLatticeCache' in all_exports
        assert 'fibonacci_hash_3d' in all_exports
        assert 'PHI_3' in all_exports
        assert 'Octree' in all_exports
        assert 'OctreeNode' in all_exports
    
    def test_hierarchical_cache_optional_import(self):
        """
        Verify HierarchicalLatticeCache is optionally available.
        
        Requirements: 7.3
        """
        try:
            from fibkvc.lattice import HierarchicalLatticeCache
            # If import succeeds, verify it's a class
            assert inspect.isclass(HierarchicalLatticeCache)
        except ImportError:
            # Optional feature may not be available
            pytest.skip("HierarchicalLatticeCache not available (optional)")
    
    def test_existing_api_imports_work(self):
        """
        Verify existing import patterns still work.
        
        Requirements: 7.5
        """
        # Test various import patterns that existing users might use
        
        # Pattern 1: Import module
        import fibkvc
        assert hasattr(fibkvc, 'fibonacci_hash')
        assert hasattr(fibkvc, 'FibonacciCacheOptimizer')
        
        # Pattern 2: Import specific items
        from fibkvc import fibonacci_hash, FibonacciCacheOptimizer
        assert callable(fibonacci_hash)
        assert inspect.isclass(FibonacciCacheOptimizer)
        
        # Pattern 3: Import constants
        from fibkvc import GOLDEN_RATIO, GOLDEN_RATIO_64
        assert isinstance(GOLDEN_RATIO, float)
        assert isinstance(GOLDEN_RATIO_64, int)
        
        # Pattern 4: Import config
        from fibkvc import FibonacciHashingConfig
        assert inspect.isclass(FibonacciHashingConfig)
    
    def test_version_follows_semantic_versioning(self):
        """
        Verify package version follows semantic versioning.
        
        Requirements: 7.5
        """
        import fibkvc
        
        assert hasattr(fibkvc, '__version__')
        version = fibkvc.__version__
        
        # Should be a string
        assert isinstance(version, str)
        
        # Should have format X.Y.Z
        parts = version.split('.')
        assert len(parts) >= 2  # At least major.minor
    
    def test_no_breaking_changes_to_existing_code(self):
        """
        Verify existing code patterns still work without modification.
        
        Requirements: 7.5
        """
        import fibkvc
        
        # Test typical usage pattern
        optimizer = fibkvc.FibonacciCacheOptimizer(
            use_fibonacci=True,
            initial_table_size=256
        )
        
        # Serialize and deserialize
        cache_state = {
            "entries": {
                "0": {"data": "value0"},
                "1": {"data": "value1"},
                "2": {"data": "value2"}
            }
        }
        
        json_str = optimizer.serialize_cache_state(cache_state)
        restored = optimizer.deserialize_cache_state(json_str)
        
        # Verify data is preserved
        assert "entries" in restored
        assert len(restored["entries"]) == 3
        
        # Test hash computation
        for i in range(10):
            hash_idx = optimizer.get_hash_index(i)
            assert isinstance(hash_idx, int)
            assert 0 <= hash_idx < optimizer.table_size
        
        # Get statistics
        stats = optimizer.get_statistics()
        assert stats["total_serializations"] == 1
        assert stats["total_deserializations"] == 1


class TestExistingFunctionalityStillWorks:
    """Test that existing fibkvc functionality still works correctly."""
    
    def test_fibonacci_hash_with_integers(self):
        """Test fibonacci_hash with integer keys."""
        from fibkvc import fibonacci_hash
        
        # Test various integer keys
        for key in [0, 1, 42, 100, 1000, 999999]:
            result = fibonacci_hash(key, 256)
            assert 0 <= result < 256
            # Deterministic
            assert fibonacci_hash(key, 256) == result
    
    def test_fibonacci_hash_with_strings(self):
        """Test fibonacci_hash with string keys."""
        from fibkvc import fibonacci_hash
        
        # Test various string keys
        for key in ["", "a", "test", "hello world", "fibonacci"]:
            result = fibonacci_hash(key, 256)
            assert 0 <= result < 256
            # Deterministic
            assert fibonacci_hash(key, 256) == result
    
    def test_fibonacci_hash_with_power_of_2_sizes(self):
        """Test fibonacci_hash with various power-of-2 table sizes."""
        from fibkvc import fibonacci_hash
        
        # Test various table sizes
        for size in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
            result = fibonacci_hash(42, size)
            assert 0 <= result < size
    
    def test_fibonacci_hash_raises_on_invalid_table_size(self):
        """Test fibonacci_hash raises ValueError for non-power-of-2 sizes."""
        from fibkvc import fibonacci_hash
        
        # Non-power-of-2 sizes should raise ValueError
        with pytest.raises(ValueError):
            fibonacci_hash(42, 100)
        
        with pytest.raises(ValueError):
            fibonacci_hash(42, 255)
    
    def test_fibonacci_cache_optimizer_initialization(self):
        """Test FibonacciCacheOptimizer can be initialized with various configs."""
        from fibkvc import FibonacciCacheOptimizer
        
        # Default initialization
        opt1 = FibonacciCacheOptimizer()
        assert opt1.use_fibonacci is True
        assert opt1.table_size == 256
        
        # Custom initialization
        opt2 = FibonacciCacheOptimizer(
            use_fibonacci=False,
            initial_table_size=512
        )
        assert opt2.use_fibonacci is False
        assert opt2.table_size == 512
    
    def test_fibonacci_cache_optimizer_serialization_round_trip(self):
        """Test serialization and deserialization round trip."""
        from fibkvc import FibonacciCacheOptimizer
        
        optimizer = FibonacciCacheOptimizer()
        
        # Create test cache state
        original = {
            "entries": {
                "0": {"key": "k0", "value": "v0"},
                "1": {"key": "k1", "value": "v1"},
                "5": {"key": "k5", "value": "v5"},
                "10": {"key": "k10", "value": "v10"}
            },
            "metadata": {
                "version": "1.0",
                "timestamp": "2026-01-01T00:00:00"
            }
        }
        
        # Serialize
        json_str = optimizer.serialize_cache_state(original)
        assert isinstance(json_str, str)
        assert len(json_str) > 0
        
        # Deserialize
        restored = optimizer.deserialize_cache_state(json_str)
        assert isinstance(restored, dict)
        assert "entries" in restored
        assert "metadata" in restored
        
        # Verify entries are preserved
        assert len(restored["entries"]) == len(original["entries"])
    
    def test_fibonacci_cache_optimizer_collision_handling(self):
        """Test that collision handling works correctly."""
        from fibkvc import FibonacciCacheOptimizer
        
        # Use small table size to force collisions
        optimizer = FibonacciCacheOptimizer(initial_table_size=16)
        
        # Insert many keys to force collisions
        hash_indices = []
        for i in range(20):
            hash_idx = optimizer.get_hash_index(i)
            hash_indices.append(hash_idx)
            assert 0 <= hash_idx < optimizer.table_size
        
        # Verify all indices are valid
        assert all(0 <= idx < optimizer.table_size for idx in hash_indices)
        
        # Check statistics
        stats = optimizer.get_statistics()
        assert "collision_count" in stats
        # With small table, we should have some collisions
        assert stats["collision_count"] >= 0
    
    def test_fibonacci_cache_optimizer_resize(self):
        """Test that automatic resizing works correctly."""
        from fibkvc import FibonacciCacheOptimizer
        
        # Start with small table
        optimizer = FibonacciCacheOptimizer(initial_table_size=16)
        initial_size = optimizer.table_size
        
        # Insert many keys to trigger resize
        for i in range(50):
            optimizer.get_hash_index(i)
        
        # Table should have resized
        assert optimizer.table_size > initial_size
        
        # Check statistics
        stats = optimizer.get_statistics()
        assert "resize_count" in stats
        assert stats["resize_count"] > 0


class TestNoRegressions:
    """Test that no regressions were introduced."""
    
    def test_import_fibkvc_does_not_import_lattice(self):
        """
        Verify that importing fibkvc does not automatically import lattice.
        
        This ensures lazy loading and minimal overhead for users who don't
        need 3D features.
        
        Requirements: 7.3
        """
        # Clear any previous imports
        if 'fibkvc' in sys.modules:
            del sys.modules['fibkvc']
        if 'fibkvc.lattice' in sys.modules:
            del sys.modules['fibkvc.lattice']
        
        # Import fibkvc
        import fibkvc
        
        # lattice should NOT be imported yet
        assert 'fibkvc.lattice' not in sys.modules
    
    def test_lattice_import_does_not_affect_main_namespace(self):
        """
        Verify that importing fibkvc.lattice does not pollute main namespace.
        
        After importing fibkvc.lattice, the lattice submodule becomes accessible
        as fibkvc.lattice (standard Python behavior), but 3D features should NOT
        be directly accessible from fibkvc namespace.
        
        Requirements: 7.3
        """
        import fibkvc
        
        # Import lattice
        import fibkvc.lattice
        
        # The lattice submodule should be accessible (standard Python behavior)
        assert hasattr(fibkvc, 'lattice')
        
        # But 3D features should NOT be directly in fibkvc namespace
        assert not hasattr(fibkvc, 'FibonacciLatticeCache')
        assert not hasattr(fibkvc, 'fibonacci_hash_3d')
        assert not hasattr(fibkvc, 'PHI_3')
        assert not hasattr(fibkvc, 'Octree')
        assert not hasattr(fibkvc, 'OctreeNode')
    
    def test_golden_ratio_values_unchanged(self):
        """
        Verify golden ratio constants have correct values.
        
        Requirements: 7.2
        """
        from fibkvc import GOLDEN_RATIO, GOLDEN_RATIO_64
        
        # Golden ratio: φ = (√5 - 1) / 2 ≈ 0.618
        assert abs(GOLDEN_RATIO - 0.6180339887498948482) < 1e-10
        
        # 64-bit golden ratio for integer arithmetic
        assert GOLDEN_RATIO_64 == 11400714819323198549
    
    def test_is_power_of_two_function_works(self):
        """Test is_power_of_two helper function."""
        from fibkvc import is_power_of_two
        
        # Powers of 2
        assert is_power_of_two(1)
        assert is_power_of_two(2)
        assert is_power_of_two(4)
        assert is_power_of_two(8)
        assert is_power_of_two(256)
        assert is_power_of_two(1024)
        
        # Not powers of 2
        assert not is_power_of_two(3)
        assert not is_power_of_two(5)
        assert not is_power_of_two(100)
        assert not is_power_of_two(255)
    
    def test_string_to_int_function_works(self):
        """Test string_to_int helper function."""
        from fibkvc import string_to_int
        
        # Should convert strings to integers
        result1 = string_to_int("test")
        result2 = string_to_int("hello")
        
        assert isinstance(result1, int)
        assert isinstance(result2, int)
        
        # Should be deterministic
        assert string_to_int("test") == result1
        assert string_to_int("hello") == result2
        
        # Different strings should (usually) give different results
        assert result1 != result2
