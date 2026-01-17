"""
Property-based tests for fibonacci_hash_3d function.

These tests use Hypothesis to verify universal properties across many
randomly generated inputs.

Feature: fibkvc-3d-lattice-extension
"""

import pytest
from hypothesis import given, strategies as st, settings
from scipy import stats
from collections import Counter
from fibkvc.lattice import fibonacci_hash_3d


# Strategy for generating 3D coordinates
@st.composite
def coordinates_3d(draw, min_val=-1000, max_val=1000):
    """Generate random 3D coordinates."""
    x = draw(st.integers(min_value=min_val, max_value=max_val))
    y = draw(st.integers(min_value=min_val, max_value=max_val))
    z = draw(st.integers(min_value=min_val, max_value=max_val))
    return (x, y, z)


class TestFibonacciHash3DProperties:
    """Property-based tests for 3D Fibonacci hash function."""
    
    @settings(max_examples=100)
    @given(coords=coordinates_3d())
    def test_hash_in_range(self, coords):
        """
        Property: Hash values are always in valid range.
        
        For any 3D coordinates, the hash value should be in [0, table_size).
        
        Feature: fibkvc-3d-lattice-extension
        """
        table_size = 1024
        hash_val = fibonacci_hash_3d(*coords, table_size)
        assert 0 <= hash_val < table_size
    
    @settings(max_examples=100)
    @given(coords=coordinates_3d(), table_size=st.integers(min_value=4, max_value=16))
    def test_hash_respects_table_size(self, coords, table_size):
        """
        Property: Hash values respect different table sizes.
        
        For any coordinates and any power-of-2 table size, hash is in range.
        
        Feature: fibkvc-3d-lattice-extension
        """
        table_size_pow2 = 2 ** table_size
        hash_val = fibonacci_hash_3d(*coords, table_size_pow2)
        assert 0 <= hash_val < table_size_pow2
    
    def test_hash_distribution_uniformity(self):
        """
        Property 1: Hash Distribution Uniformity
        
        For any large set of random 3D coordinates, the distribution of hash
        values should be uniform (chi-square test p-value > 0.05) and all
        hash buckets should be utilized.
        
        Validates: Requirements 1.4, 9.2
        Feature: fibkvc-3d-lattice-extension, Property 1
        """
        import random
        random.seed(42)  # For reproducibility
        
        table_size = 1024
        num_samples = 10000
        
        # Generate 10,000 random 3D coordinates
        coords_list = [
            (
                random.randint(-5000, 5000),
                random.randint(-5000, 5000),
                random.randint(-5000, 5000)
            )
            for _ in range(num_samples)
        ]
        
        # Hash all coordinates
        hashes = [fibonacci_hash_3d(*coords, table_size) for coords in coords_list]
        
        # Check that we're using all buckets (or close to it)
        unique_hashes = len(set(hashes))
        bucket_utilization = unique_hashes / table_size
        
        # With 10,000 samples and 1024 buckets, we should use >95% of buckets
        assert bucket_utilization > 0.95, \
            f"Poor bucket utilization: {bucket_utilization:.2%} (expected >95%)"
        
        # Check uniform distribution using chi-square test
        # Count occurrences in each bin
        hash_counts = Counter(hashes)
        
        # Expected count per bin if uniform
        expected_count = num_samples / table_size
        
        # Compute chi-square statistic
        observed = [hash_counts.get(i, 0) for i in range(table_size)]
        expected = [expected_count] * table_size
        
        chi2_stat, p_value = stats.chisquare(observed, expected)
        
        # p-value > 0.05 indicates distribution is not significantly different from uniform
        assert p_value > 0.05, \
            f"Distribution not uniform: chi-square p-value = {p_value:.4f} (expected >0.05)"
        
        print(f"\nHash Distribution Uniformity Test:")
        print(f"  Samples: {num_samples}")
        print(f"  Table size: {table_size}")
        print(f"  Unique hashes: {unique_hashes}")
        print(f"  Bucket utilization: {bucket_utilization:.2%}")
        print(f"  Chi-square p-value: {p_value:.4f}")

    def test_spatial_locality_preservation(self):
        """
        Property 2: Spatial Locality Preservation
        
        For any two pairs of 3D coordinates where one pair is spatially close
        (distance < threshold) and the other pair is spatially distant
        (distance > 10 * threshold), the hash distance between the close pair
        should be smaller on average than the hash distance between the distant pair.
        
        Validates: Requirements 1.5
        Feature: fibkvc-3d-lattice-extension, Property 2
        """
        import random
        import math
        random.seed(42)  # For reproducibility
        
        table_size = 1024
        num_pairs = 1000
        close_threshold = 5
        distant_threshold = 50
        
        close_hash_distances = []
        distant_hash_distances = []
        
        for _ in range(num_pairs):
            # Generate a random center point
            center_x = random.randint(-1000, 1000)
            center_y = random.randint(-1000, 1000)
            center_z = random.randint(-1000, 1000)
            
            # Generate a nearby point (distance < close_threshold)
            offset_x = random.randint(-close_threshold, close_threshold)
            offset_y = random.randint(-close_threshold, close_threshold)
            offset_z = random.randint(-close_threshold, close_threshold)
            nearby_x = center_x + offset_x
            nearby_y = center_y + offset_y
            nearby_z = center_z + offset_z
            
            # Compute hash distance for nearby pair
            hash_center = fibonacci_hash_3d(center_x, center_y, center_z, table_size)
            hash_nearby = fibonacci_hash_3d(nearby_x, nearby_y, nearby_z, table_size)
            close_hash_dist = abs(hash_center - hash_nearby)
            close_hash_distances.append(close_hash_dist)
            
            # Generate a distant point (distance > distant_threshold)
            angle1 = random.uniform(0, 2 * math.pi)
            angle2 = random.uniform(0, math.pi)
            distance = random.uniform(distant_threshold, distant_threshold * 2)
            
            distant_x = center_x + int(distance * math.sin(angle2) * math.cos(angle1))
            distant_y = center_y + int(distance * math.sin(angle2) * math.sin(angle1))
            distant_z = center_z + int(distance * math.cos(angle2))
            
            # Compute hash distance for distant pair
            hash_distant = fibonacci_hash_3d(distant_x, distant_y, distant_z, table_size)
            distant_hash_dist = abs(hash_center - hash_distant)
            distant_hash_distances.append(distant_hash_dist)
        
        # Compute average hash distances
        avg_close_hash_dist = sum(close_hash_distances) / len(close_hash_distances)
        avg_distant_hash_dist = sum(distant_hash_distances) / len(distant_hash_distances)
        
        # Nearby pairs should have smaller hash distance on average
        assert avg_close_hash_dist < avg_distant_hash_dist, \
            f"Spatial locality not preserved: " \
            f"avg close hash dist = {avg_close_hash_dist:.2f}, " \
            f"avg distant hash dist = {avg_distant_hash_dist:.2f}"
        
        print(f"\nSpatial Locality Preservation Test:")
        print(f"  Number of pairs: {num_pairs}")
        print(f"  Close threshold: {close_threshold}")
        print(f"  Distant threshold: {distant_threshold}")
        print(f"  Avg close hash distance: {avg_close_hash_dist:.2f}")
        print(f"  Avg distant hash distance: {avg_distant_hash_dist:.2f}")
        print(f"  Ratio: {avg_distant_hash_dist / avg_close_hash_dist:.2f}x")
