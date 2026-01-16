# Copyright (c) 2026 California Vision, Inc. - David Anderson
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Fibonacci Cache Optimizer with JSON integration.

This module provides the FibonacciCacheOptimizer class that handles
JSON serialization/deserialization with fibonacci hashing optimizations
for token position indexing.

The optimizer maintains bidirectional mappings between token positions and
fibonacci hash values, enabling efficient cache lookups during KV cache operations.
"""

import json
import logging
import time
from typing import Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime

from fibkvc.fibonacci_hash import fibonacci_hash, is_power_of_two
from fibkvc.fibonacci_config import FibonacciHashingConfig, get_global_config

logger = logging.getLogger(__name__)


class FibonacciCacheOptimizer:
    """
    Wraps JSON handling with fibonacci hashing for optimized token position lookups.
    
    This class manages serialization and deserialization of cache state using
    JSON as the underlying handler, while applying fibonacci hashing
    to token position indices for improved distribution and lookup performance.
    
    Attributes:
        use_fibonacci: Whether to use fibonacci hashing (default: True)
        initial_table_size: Initial size of hash table (must be power of 2)
        hash_table: Maps token_position -> fibonacci_hash_value
        reverse_table: Maps fibonacci_hash_value -> token_position
        collision_table: Tracks collision chains for linear probing
        statistics: Tracks optimization metrics
        config: FibonacciHashingConfig for configuration and monitoring
        LOAD_FACTOR_THRESHOLD: Threshold for automatic resizing (default: 0.75)
    """
    
    LOAD_FACTOR_THRESHOLD = 0.75

logger = logging.getLogger(__name__)


class FibonacciCacheOptimizer:
    """
    Wraps JSON handling with fibonacci hashing for optimized token position lookups.
    
    This class manages serialization and deserialization of cache state using
    JSON as the underlying handler, while applying fibonacci hashing
    to token position indices for improved distribution and lookup performance.
    
    Attributes:
        use_fibonacci: Whether to use fibonacci hashing (default: True)
        initial_table_size: Initial size of hash table (must be power of 2)
        hash_table: Maps token_position -> fibonacci_hash_value
        reverse_table: Maps fibonacci_hash_value -> token_position
        collision_table: Tracks collision chains for linear probing
        statistics: Tracks optimization metrics
        config: FibonacciHashingConfig for configuration and monitoring
        LOAD_FACTOR_THRESHOLD: Threshold for automatic resizing (default: 0.75)
    """
    
    LOAD_FACTOR_THRESHOLD = 0.75
    
    def __init__(
        self,
        use_fibonacci: bool = True,
        initial_table_size: int = 256,
        config: Optional[FibonacciHashingConfig] = None
    ):
        """
        Initialize the Fibonacci Cache Optimizer.
        
        Args:
            use_fibonacci: Enable/disable fibonacci hashing (default: True)
            initial_table_size: Initial hash table size, must be power of 2
            config: Optional FibonacciHashingConfig for configuration and monitoring
            
        Raises:
            ValueError: If initial_table_size is not a power of 2
            TypeError: If parameters have invalid types
        """
        if not isinstance(use_fibonacci, bool):
            raise TypeError(f"use_fibonacci must be bool, got {type(use_fibonacci).__name__}")
        
        if not isinstance(initial_table_size, int):
            raise TypeError(f"initial_table_size must be int, got {type(initial_table_size).__name__}")
        
        if not is_power_of_two(initial_table_size):
            raise ValueError(f"initial_table_size must be power of 2, got {initial_table_size}")
        
        if config is not None and not isinstance(config, FibonacciHashingConfig):
            raise TypeError(f"config must be FibonacciHashingConfig or None, got {type(config).__name__}")
        
        self.use_fibonacci = use_fibonacci
        self.table_size = initial_table_size
        
        # Use provided config or get global config
        self.config = config if config is not None else get_global_config()
        
        # Bidirectional mappings for token positions and their fibonacci hashes
        self.hash_table: Dict[int, int] = {}  # token_position -> fibonacci_hash_value
        self.reverse_table: Dict[int, int] = {}  # fibonacci_hash_value -> token_position
        
        # Collision handling: maps hash_index -> list of (token_position, collision_count)
        # This tracks collision chains for linear probing
        self.collision_table: Dict[int, list] = {}
        
        # Statistics for monitoring
        self.statistics: Dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "total_serializations": 0,
            "total_deserializations": 0,
            "total_file_saves": 0,
            "total_file_loads": 0,
            "collision_count": 0,
            "resize_count": 0,
            "load_factor": 0.0,
            "max_collision_chain_length": 0,
        }
        
        logger.info(
            f"Initialized FibonacciCacheOptimizer: "
            f"use_fibonacci={use_fibonacci}, table_size={initial_table_size}"
        )
    
    def serialize_cache_state(self, cache_state: Dict[str, Any]) -> str:
        """
        Serialize cache state to JSON with fibonacci indexing.

        If fibonacci hashing is enabled, token positions are mapped to their
        fibonacci hash values as keys in the output JSON. This optimization
        improves lookup performance for token position queries.

        Args:
            cache_state: Dictionary containing cache state to serialize

        Returns:
            JSON string with optimized token position keys

        Raises:
            TypeError: If cache_state is not a dictionary
            ValueError: If cache_state contains invalid data
        """
        if not isinstance(cache_state, dict):
            raise TypeError(f"cache_state must be dict, got {type(cache_state).__name__}")

        try:
            if self.use_fibonacci:
                # Create optimized cache state with fibonacci-hashed indices
                optimized_state = self._apply_fibonacci_indexing(cache_state)
            else:
                # Use standard serialization
                optimized_state = cache_state

            # Use json for serialization
            json_str = json.dumps(optimized_state)

            self.statistics["total_serializations"] += 1
            self.statistics["last_updated"] = datetime.now().isoformat()

            logger.debug(f"Serialized cache state: {len(json_str)} bytes")
            return json_str

        except Exception as e:
            logger.error(f"Failed to serialize cache state: {e}")
            raise ValueError(f"Cache state serialization failed: {e}") from e
    
    def deserialize_cache_state(self, json_str: str) -> Dict[str, Any]:
        """
        Deserialize JSON back to cache state.

        Reconstructs fibonacci hash indices from JSON representation if
        fibonacci hashing was used during serialization.

        Args:
            json_str: JSON string to deserialize

        Returns:
            Dictionary containing deserialized cache state

        Raises:
            TypeError: If json_str is not a string
            ValueError: If JSON is malformed or contains invalid data
        """
        if not isinstance(json_str, str):
            raise TypeError(f"json_str must be str, got {type(json_str).__name__}")

        try:
            # Use json for deserialization
            cache_state = json.loads(json_str)

            if self.use_fibonacci:
                # Reconstruct original token positions from fibonacci indices
                cache_state = self._restore_original_indexing(cache_state)

            self.statistics["total_deserializations"] += 1
            self.statistics["last_updated"] = datetime.now().isoformat()

            logger.debug(f"Deserialized cache state from {len(json_str)} bytes")
            return cache_state

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            raise ValueError(f"Invalid JSON format: {e}") from e
        except Exception as e:
            logger.error(f"Failed to deserialize cache state: {e}")
            raise ValueError(f"Cache state deserialization failed: {e}") from e
    
    def save_to_file(self, cache_state: Dict[str, Any], filepath: str) -> None:
        """
        Save cache state to JSON file.

        Args:
            cache_state: Dictionary containing cache state to save
            filepath: Path to file where cache state will be saved

        Raises:
            TypeError: If parameters have invalid types
            IOError: If file cannot be written
            ValueError: If cache_state is invalid
        """
        if not isinstance(filepath, str):
            raise TypeError(f"filepath must be str, got {type(filepath).__name__}")

        try:
            # Serialize cache state
            json_str = self.serialize_cache_state(cache_state)

            # Write to file
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, 'w') as f:
                f.write(json_str)

            self.statistics["total_file_saves"] += 1
            self.statistics["last_updated"] = datetime.now().isoformat()

            logger.info(f"Saved cache state to {filepath}")

        except IOError as e:
            logger.error(f"File I/O error: {e}")
            raise IOError(f"Failed to save cache state to {filepath}: {e}") from e
        except Exception as e:
            logger.error(f"Failed to save cache state: {e}")
            raise
    
    def load_from_file(self, filepath: str) -> Dict[str, Any]:
        """
        Load cache state from JSON file.

        Args:
            filepath: Path to file containing cache state

        Returns:
            Dictionary containing loaded cache state

        Raises:
            TypeError: If filepath is not a string
            IOError: If file cannot be read
            ValueError: If file contains invalid JSON
        """
        if not isinstance(filepath, str):
            raise TypeError(f"filepath must be str, got {type(filepath).__name__}")

        try:
            path = Path(filepath)

            if not path.exists():
                raise IOError(f"File not found: {filepath}")

            # Read file
            with open(path, 'r') as f:
                json_str = f.read()

            # Deserialize cache state
            cache_state = self.deserialize_cache_state(json_str)

            self.statistics["total_file_loads"] += 1
            self.statistics["last_updated"] = datetime.now().isoformat()

            logger.info(f"Loaded cache state from {filepath}")
            return cache_state

        except IOError as e:
            logger.error(f"File I/O error: {e}")
            raise IOError(f"Failed to load cache state from {filepath}: {e}") from e
        except Exception as e:
            logger.error(f"Failed to load cache state: {e}")
            raise
    
    def get_hash_index(self, token_position: int) -> int:
        """
        Get fibonacci hash index for a token position.

        If the token position hasn't been hashed yet, computes and stores
        the hash value for future lookups. Uses linear probing to handle
        collisions and automatically resizes the hash table when load factor
        exceeds the threshold.

        Args:
            token_position: Token position to hash

        Returns:
            Fibonacci hash index for the token position

        Raises:
            TypeError: If token_position is not an integer
        """
        if not isinstance(token_position, int):
            raise TypeError(f"token_position must be int, got {type(token_position).__name__}")

        # Record lookup start time for latency monitoring
        lookup_start_time = time.time() if self.config.monitor_lookup_latency else None

        # Check if already computed
        if token_position in self.hash_table:
            hash_index = self.hash_table[token_position]
            
            # Record cache hit
            if self.config.monitor_cache_hit_rate:
                self.config.record_cache_hit()
            
            # Record lookup time
            if lookup_start_time is not None:
                lookup_time_ms = (time.time() - lookup_start_time) * 1000
                self.config.record_lookup_time(lookup_time_ms)
            
            return hash_index

        # Record cache miss
        if self.config.monitor_cache_hit_rate:
            self.config.record_cache_miss()

        # Check if resizing is needed before adding new entry
        current_load_factor = len(self.hash_table) / self.table_size if self.table_size > 0 else 0.0
        if current_load_factor >= self.LOAD_FACTOR_THRESHOLD:
            self._resize_hash_table()

        # Compute fibonacci hash
        hash_index = fibonacci_hash(token_position, self.table_size)

        # Linear probing: find next available slot if collision occurs
        probed_index = hash_index
        collision_chain_length = 0

        while probed_index in self.reverse_table:
            existing_pos = self.reverse_table[probed_index]
            if existing_pos == token_position:
                # Already exists, return existing hash
                if lookup_start_time is not None:
                    lookup_time_ms = (time.time() - lookup_start_time) * 1000
                    self.config.record_lookup_time(lookup_time_ms)
                return self.hash_table[token_position]

            # Collision detected, use linear probing with quadratic step
            collision_chain_length += 1
            self.statistics["collision_count"] += 1

            # Linear probing: try next slot
            probed_index = (probed_index + 1) % self.table_size

            # Safety check: prevent infinite loop
            if collision_chain_length > self.table_size:
                logger.error(
                    f"Hash table exhausted: unable to find slot for position {token_position}"
                )
                raise RuntimeError(
                    f"Hash table exhausted: unable to find slot for position {token_position}"
                )

        # Track collision chain length for statistics
        if collision_chain_length > 0:
            if self.config.log_collisions:
                logger.debug(
                    f"Hash collision for position {token_position}: "
                    f"initial hash {hash_index}, probed to {probed_index}, "
                    f"chain length {collision_chain_length}"
                )
            
            if collision_chain_length > self.statistics["max_collision_chain_length"]:
                self.statistics["max_collision_chain_length"] = collision_chain_length
            
            # Call collision callback if configured
            if self.config.on_collision_callback is not None:
                collision_event = {
                    "token_position": token_position,
                    "initial_hash": hash_index,
                    "probed_index": probed_index,
                    "chain_length": collision_chain_length,
                    "timestamp": datetime.now().isoformat(),
                }
                try:
                    self.config.on_collision_callback(collision_event)
                except Exception as e:
                    logger.error(f"Error in collision callback: {e}")

        # Store bidirectional mapping at probed index
        self.hash_table[token_position] = probed_index
        self.reverse_table[probed_index] = token_position

        # Track collision chain
        if hash_index not in self.collision_table:
            self.collision_table[hash_index] = []
        self.collision_table[hash_index].append((token_position, collision_chain_length))

        # Update load factor
        self._update_load_factor()

        # Record lookup time
        if lookup_start_time is not None:
            lookup_time_ms = (time.time() - lookup_start_time) * 1000
            self.config.record_lookup_time(lookup_time_ms)

        return probed_index
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Return optimization statistics for monitoring.

        Returns:
            Dictionary containing:
            - created_at: Timestamp when optimizer was created
            - last_updated: Timestamp of last operation
            - total_serializations: Number of serialization operations
            - total_deserializations: Number of deserialization operations
            - total_file_saves: Number of file save operations
            - total_file_loads: Number of file load operations
            - collision_count: Total hash collisions detected
            - resize_count: Number of hash table resizes
            - load_factor: Current load factor (entries / table_size)
            - max_collision_chain_length: Maximum collision chain length observed
            - table_size: Current hash table size
            - num_entries: Number of entries in hash table
            - monitoring_stats: Cache hit rate and lookup latency statistics
        """
        stats = self.statistics.copy()
        stats["table_size"] = self.table_size
        stats["num_entries"] = len(self.hash_table)
        stats["load_factor"] = len(self.hash_table) / self.table_size if self.table_size > 0 else 0.0
        
        # Include monitoring statistics
        stats["monitoring_stats"] = self.config.get_monitoring_stats()
        
        # Log statistics if configured
        if self.config.log_statistics:
            logger.info(
                f"Fibonacci cache statistics: "
                f"entries={stats['num_entries']}, "
                f"collisions={stats['collision_count']}, "
                f"load_factor={stats['load_factor']:.2%}, "
                f"cache_hit_rate={stats['monitoring_stats']['cache_hit_rate']:.1f}%"
            )
        
        # Call statistics callback if configured
        if self.config.on_statistics_callback is not None:
            try:
                self.config.on_statistics_callback(stats)
            except Exception as e:
                logger.error(f"Error in statistics callback: {e}")
        
        return stats
    
    def _apply_fibonacci_indexing(self, cache_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply fibonacci hashing to token position indices in cache state.

        Transforms cache state by storing entries with their original token positions
        as keys, but also storing the fibonacci hash indices for optimization.
        This preserves all data while enabling fibonacci-hashed lookups.

        Args:
            cache_state: Original cache state dictionary

        Returns:
            Cache state with fibonacci hash indices stored in metadata
        """
        optimized_state = {}

        # Copy metadata
        for key, value in cache_state.items():
            if key == "entries" and isinstance(value, dict):
                # Keep original entries but add hash indices to metadata
                optimized_entries = {}
                hash_indices = {}  # Maps original_token_pos -> hash_index
                
                for token_pos, entry_data in value.items():
                    if isinstance(token_pos, (int, str)):
                        try:
                            token_pos_int = int(token_pos)
                            hash_index = self.get_hash_index(token_pos_int)
                            
                            # Store entry with original position as key
                            optimized_entries[str(token_pos_int)] = entry_data
                            
                            # Store hash index for optimization
                            hash_indices[str(token_pos_int)] = hash_index
                        except (ValueError, TypeError):
                            # If conversion fails, keep original key
                            optimized_entries[token_pos] = entry_data
                    else:
                        optimized_entries[token_pos] = entry_data
                
                optimized_state[key] = optimized_entries
                # Store hash indices as metadata for optimization
                optimized_state["_hash_indices"] = hash_indices
            else:
                optimized_state[key] = value

        return optimized_state
    
    def _restore_original_indexing(self, cache_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Restore original token position indices from fibonacci hashes.

        Since we store entries with original token positions as keys and only
        store hash indices in metadata, restoration is straightforward - just
        remove the metadata and return the entries as-is.

        Args:
            cache_state: Cache state with fibonacci hash indices in metadata

        Returns:
            Cache state with original token position indices
        """
        restored_state = {}

        # Copy all entries except metadata
        for key, value in cache_state.items():
            if key not in ("_hash_indices",):
                # Keep all non-metadata keys
                restored_state[key] = value

        return restored_state
    
    def _update_load_factor(self) -> None:
        """
        Update load factor statistic.

        Calculates and stores the current load factor (entries / table_size).
        """
        if self.table_size > 0:
            self.statistics["load_factor"] = len(self.hash_table) / self.table_size
    
    def _resize_hash_table(self) -> None:
        """
        Resize hash table to next power of 2 when load factor exceeds threshold.

        This method:
        1. Doubles the table size to the next power of 2
        2. Rehashes all existing entries with the new table size
        3. Clears collision tracking and recomputes it
        4. Updates statistics
        5. Calls resize callback if configured

        This ensures O(1) average lookup time is maintained as the table grows.
        """
        old_size = self.table_size
        new_size = old_size * 2
        
        # Record resize start time
        resize_start_time = time.time()

        if self.config.log_resizes:
            logger.info(
                f"Resizing hash table from {old_size} to {new_size} "
                f"(load factor: {len(self.hash_table) / old_size:.2%})"
            )

        # Save old mappings
        old_hash_table = self.hash_table.copy()
        old_reverse_table = self.reverse_table.copy()

        # Clear tables for rehashing
        self.hash_table.clear()
        self.reverse_table.clear()
        self.collision_table.clear()

        # Update table size
        self.table_size = new_size

        # Rehash all entries with new table size
        for token_position in old_hash_table.keys():
            # Recompute hash with new table size
            hash_index = fibonacci_hash(token_position, self.table_size)

            # Linear probing for new table
            probed_index = hash_index
            collision_chain_length = 0

            while probed_index in self.reverse_table:
                collision_chain_length += 1
                probed_index = (probed_index + 1) % self.table_size

                if collision_chain_length > self.table_size:
                    logger.error(
                        f"Hash table exhausted during resize: "
                        f"unable to find slot for position {token_position}"
                    )
                    raise RuntimeError(
                        f"Hash table exhausted during resize: "
                        f"unable to find slot for position {token_position}"
                    )

            # Store in new table
            self.hash_table[token_position] = probed_index
            self.reverse_table[probed_index] = token_position

            # Track collision chain
            if collision_chain_length > 0:
                if hash_index not in self.collision_table:
                    self.collision_table[hash_index] = []
                self.collision_table[hash_index].append((token_position, collision_chain_length))

        # Update statistics
        self.statistics["resize_count"] += 1
        self.statistics["last_updated"] = datetime.now().isoformat()
        self._update_load_factor()

        # Record resize time
        resize_time_ms = (time.time() - resize_start_time) * 1000

        if self.config.log_resizes:
            logger.info(
                f"Hash table resize complete: {len(self.hash_table)} entries, "
                f"new load factor: {self.statistics['load_factor']:.2%}, "
                f"resize time: {resize_time_ms:.2f}ms"
            )

        # Call resize callback if configured
        if self.config.on_resize_callback is not None:
            resize_event = {
                "old_size": old_size,
                "new_size": new_size,
                "num_entries": len(self.hash_table),
                "load_factor": self.statistics["load_factor"],
                "resize_time_ms": resize_time_ms,
                "timestamp": datetime.now().isoformat(),
            }
            try:
                self.config.on_resize_callback(resize_event)
            except Exception as e:
                logger.error(f"Error in resize callback: {e}")
