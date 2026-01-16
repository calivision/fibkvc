# Copyright (c) 2026 California Vision, Inc. - David Anderson
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Configuration management for Fibonacci Hashing in Elastic-Cache.

This module provides configuration classes and utilities for managing
fibonacci hashing behavior, including enable/disable parameters,
hash table sizing, logging, and monitoring hooks.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Logging level enumeration for fibonacci hashing."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR


@dataclass
class FibonacciHashingConfig:
    """
    Configuration for Fibonacci Hashing in Elastic-Cache.
    
    This class manages all configuration parameters for fibonacci hashing,
    including enable/disable flags, hash table sizing, logging levels,
    and monitoring hooks.
    
    Attributes:
        enabled: Enable/disable fibonacci hashing (default: False)
        initial_table_size: Initial hash table size, must be power of 2 (default: 256)
        load_factor_threshold: Threshold for automatic resizing (default: 0.75)
        log_level: Logging level for fibonacci hashing operations (default: INFO)
        log_statistics: Log hash table statistics on each operation (default: False)
        log_collisions: Log collision events (default: False)
        log_resizes: Log hash table resize operations (default: True)
        monitor_cache_hit_rate: Enable cache hit rate monitoring (default: False)
        monitor_lookup_latency: Enable lookup latency monitoring (default: False)
        monitor_collision_rate: Enable collision rate monitoring (default: False)
        on_collision_callback: Optional callback function for collision events
        on_resize_callback: Optional callback function for resize events
        on_statistics_callback: Optional callback function for statistics updates
    """
    
    # Enable/disable parameters
    enabled: bool = False
    
    # Hash table sizing parameters
    initial_table_size: int = 256
    load_factor_threshold: float = 0.75
    
    # Logging parameters
    log_level: LogLevel = LogLevel.INFO
    log_statistics: bool = False
    log_collisions: bool = False
    log_resizes: bool = True
    
    # Monitoring parameters
    monitor_cache_hit_rate: bool = False
    monitor_lookup_latency: bool = False
    monitor_collision_rate: bool = False
    
    # Callback functions for monitoring hooks
    on_collision_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    on_resize_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    on_statistics_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    
    # Internal tracking for monitoring
    _cache_hit_count: int = field(default=0, init=False, repr=False)
    _cache_miss_count: int = field(default=0, init=False, repr=False)
    _lookup_times: list = field(default_factory=list, init=False, repr=False)
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        self._validate_config()
        self._setup_logging()
    
    def _validate_config(self) -> None:
        """
        Validate configuration parameters.
        
        Raises:
            ValueError: If configuration parameters are invalid
            TypeError: If configuration parameters have invalid types
        """
        # Validate types
        if not isinstance(self.enabled, bool):
            raise TypeError(f"enabled must be bool, got {type(self.enabled).__name__}")
        
        if not isinstance(self.initial_table_size, int):
            raise TypeError(f"initial_table_size must be int, got {type(self.initial_table_size).__name__}")
        
        if not isinstance(self.load_factor_threshold, (int, float)):
            raise TypeError(f"load_factor_threshold must be float, got {type(self.load_factor_threshold).__name__}")
        
        if not isinstance(self.log_level, LogLevel):
            raise TypeError(f"log_level must be LogLevel, got {type(self.log_level).__name__}")
        
        # Validate values
        if self.initial_table_size < 1:
            raise ValueError(f"initial_table_size must be >= 1, got {self.initial_table_size}")
        
        # Check if initial_table_size is power of 2
        if self.initial_table_size & (self.initial_table_size - 1) != 0:
            raise ValueError(f"initial_table_size must be power of 2, got {self.initial_table_size}")
        
        if not (0.0 < self.load_factor_threshold < 1.0):
            raise ValueError(f"load_factor_threshold must be in (0.0, 1.0), got {self.load_factor_threshold}")
        
        # Validate callbacks
        if self.on_collision_callback is not None and not callable(self.on_collision_callback):
            raise TypeError(f"on_collision_callback must be callable, got {type(self.on_collision_callback).__name__}")
        
        if self.on_resize_callback is not None and not callable(self.on_resize_callback):
            raise TypeError(f"on_resize_callback must be callable, got {type(self.on_resize_callback).__name__}")
        
        if self.on_statistics_callback is not None and not callable(self.on_statistics_callback):
            raise TypeError(f"on_statistics_callback must be callable, got {type(self.on_statistics_callback).__name__}")
    
    def _setup_logging(self) -> None:
        """Set up logging for fibonacci hashing module."""
        fib_logger = logging.getLogger("fibkvc.fibonacci_hash")
        fib_logger.setLevel(self.log_level.value)
        
        cache_logger = logging.getLogger("fibkvc.fibonacci_cache")
        cache_logger.setLevel(self.log_level.value)
        
        logger.debug(f"Fibonacci hashing logging configured at level {self.log_level.name}")
    
    def enable(self) -> None:
        """Enable fibonacci hashing."""
        self.enabled = True
        logger.info("Fibonacci hashing enabled")
    
    def disable(self) -> None:
        """Disable fibonacci hashing."""
        self.enabled = False
        logger.info("Fibonacci hashing disabled")
    
    def set_log_level(self, level: LogLevel) -> None:
        """
        Set logging level for fibonacci hashing.
        
        Args:
            level: LogLevel enum value
            
        Raises:
            TypeError: If level is not LogLevel
        """
        if not isinstance(level, LogLevel):
            raise TypeError(f"level must be LogLevel, got {type(level).__name__}")
        
        self.log_level = level
        self._setup_logging()
        logger.info(f"Fibonacci hashing log level set to {level.name}")
    
    def set_initial_table_size(self, size: int) -> None:
        """
        Set initial hash table size.
        
        Args:
            size: Hash table size (must be power of 2)
            
        Raises:
            ValueError: If size is not a power of 2
            TypeError: If size is not an integer
        """
        if not isinstance(size, int):
            raise TypeError(f"size must be int, got {type(size).__name__}")
        
        if size < 1:
            raise ValueError(f"size must be >= 1, got {size}")
        
        if size & (size - 1) != 0:
            raise ValueError(f"size must be power of 2, got {size}")
        
        self.initial_table_size = size
        logger.info(f"Initial hash table size set to {size}")
    
    def set_load_factor_threshold(self, threshold: float) -> None:
        """
        Set load factor threshold for automatic resizing.
        
        Args:
            threshold: Load factor threshold (must be in (0.0, 1.0))
            
        Raises:
            ValueError: If threshold is not in valid range
            TypeError: If threshold is not a float
        """
        if not isinstance(threshold, (int, float)):
            raise TypeError(f"threshold must be float, got {type(threshold).__name__}")
        
        if not (0.0 < threshold < 1.0):
            raise ValueError(f"threshold must be in (0.0, 1.0), got {threshold}")
        
        self.load_factor_threshold = threshold
        logger.info(f"Load factor threshold set to {threshold}")
    
    def enable_statistics_logging(self) -> None:
        """Enable logging of hash table statistics."""
        self.log_statistics = True
        logger.info("Hash table statistics logging enabled")
    
    def disable_statistics_logging(self) -> None:
        """Disable logging of hash table statistics."""
        self.log_statistics = False
        logger.info("Hash table statistics logging disabled")
    
    def enable_collision_logging(self) -> None:
        """Enable logging of collision events."""
        self.log_collisions = True
        logger.info("Collision event logging enabled")
    
    def disable_collision_logging(self) -> None:
        """Disable logging of collision events."""
        self.log_collisions = False
        logger.info("Collision event logging disabled")
    
    def enable_resize_logging(self) -> None:
        """Enable logging of hash table resize operations."""
        self.log_resizes = True
        logger.info("Hash table resize logging enabled")
    
    def disable_resize_logging(self) -> None:
        """Disable logging of hash table resize operations."""
        self.log_resizes = False
        logger.info("Hash table resize logging disabled")
    
    def enable_cache_hit_rate_monitoring(self) -> None:
        """Enable cache hit rate monitoring."""
        self.monitor_cache_hit_rate = True
        logger.info("Cache hit rate monitoring enabled")
    
    def disable_cache_hit_rate_monitoring(self) -> None:
        """Disable cache hit rate monitoring."""
        self.monitor_cache_hit_rate = False
        logger.info("Cache hit rate monitoring disabled")
    
    def enable_lookup_latency_monitoring(self) -> None:
        """Enable lookup latency monitoring."""
        self.monitor_lookup_latency = True
        logger.info("Lookup latency monitoring enabled")
    
    def disable_lookup_latency_monitoring(self) -> None:
        """Disable lookup latency monitoring."""
        self.monitor_lookup_latency = False
        logger.info("Lookup latency monitoring disabled")
    
    def enable_collision_rate_monitoring(self) -> None:
        """Enable collision rate monitoring."""
        self.monitor_collision_rate = True
        logger.info("Collision rate monitoring enabled")
    
    def disable_collision_rate_monitoring(self) -> None:
        """Disable collision rate monitoring."""
        self.monitor_collision_rate = False
        logger.info("Collision rate monitoring disabled")
    
    def set_collision_callback(self, callback: Optional[Callable[[Dict[str, Any]], None]]) -> None:
        """
        Set callback function for collision events.
        
        Args:
            callback: Callable that receives collision event data, or None to disable
            
        Raises:
            TypeError: If callback is not callable or None
        """
        if callback is not None and not callable(callback):
            raise TypeError(f"callback must be callable or None, got {type(callback).__name__}")
        
        self.on_collision_callback = callback
        logger.debug(f"Collision callback {'set' if callback else 'cleared'}")
    
    def set_resize_callback(self, callback: Optional[Callable[[Dict[str, Any]], None]]) -> None:
        """
        Set callback function for resize events.
        
        Args:
            callback: Callable that receives resize event data, or None to disable
            
        Raises:
            TypeError: If callback is not callable or None
        """
        if callback is not None and not callable(callback):
            raise TypeError(f"callback must be callable or None, got {type(callback).__name__}")
        
        self.on_resize_callback = callback
        logger.debug(f"Resize callback {'set' if callback else 'cleared'}")
    
    def set_statistics_callback(self, callback: Optional[Callable[[Dict[str, Any]], None]]) -> None:
        """
        Set callback function for statistics updates.
        
        Args:
            callback: Callable that receives statistics data, or None to disable
            
        Raises:
            TypeError: If callback is not callable or None
        """
        if callback is not None and not callable(callback):
            raise TypeError(f"callback must be callable or None, got {type(callback).__name__}")
        
        self.on_statistics_callback = callback
        logger.debug(f"Statistics callback {'set' if callback else 'cleared'}")
    
    def record_cache_hit(self) -> None:
        """Record a cache hit event."""
        self._cache_hit_count += 1
    
    def record_cache_miss(self) -> None:
        """Record a cache miss event."""
        self._cache_miss_count += 1
    
    def get_cache_hit_rate(self) -> float:
        """
        Get current cache hit rate.
        
        Returns:
            Cache hit rate as a percentage (0.0 to 100.0)
        """
        total = self._cache_hit_count + self._cache_miss_count
        if total == 0:
            return 0.0
        return (self._cache_hit_count / total) * 100.0
    
    def record_lookup_time(self, time_ms: float) -> None:
        """
        Record a lookup operation time.
        
        Args:
            time_ms: Lookup time in milliseconds
        """
        self._lookup_times.append(time_ms)
    
    def get_average_lookup_time(self) -> float:
        """
        Get average lookup time.
        
        Returns:
            Average lookup time in milliseconds, or 0.0 if no lookups recorded
        """
        if not self._lookup_times:
            return 0.0
        return sum(self._lookup_times) / len(self._lookup_times)
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """
        Get current monitoring statistics.
        
        Returns:
            Dictionary containing:
            - cache_hit_count: Total cache hits
            - cache_miss_count: Total cache misses
            - cache_hit_rate: Cache hit rate percentage
            - lookup_count: Total lookups recorded
            - average_lookup_time_ms: Average lookup time
        """
        return {
            "cache_hit_count": self._cache_hit_count,
            "cache_miss_count": self._cache_miss_count,
            "cache_hit_rate": self.get_cache_hit_rate(),
            "lookup_count": len(self._lookup_times),
            "average_lookup_time_ms": self.get_average_lookup_time(),
        }
    
    def reset_monitoring_stats(self) -> None:
        """Reset all monitoring statistics."""
        self._cache_hit_count = 0
        self._cache_miss_count = 0
        self._lookup_times.clear()
        logger.info("Monitoring statistics reset")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return {
            "enabled": self.enabled,
            "initial_table_size": self.initial_table_size,
            "load_factor_threshold": self.load_factor_threshold,
            "log_level": self.log_level.name,
            "log_statistics": self.log_statistics,
            "log_collisions": self.log_collisions,
            "log_resizes": self.log_resizes,
            "monitor_cache_hit_rate": self.monitor_cache_hit_rate,
            "monitor_lookup_latency": self.monitor_lookup_latency,
            "monitor_collision_rate": self.monitor_collision_rate,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "FibonacciHashingConfig":
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters
            
        Returns:
            FibonacciHashingConfig instance
            
        Raises:
            ValueError: If configuration is invalid
            TypeError: If configuration has invalid types
        """
        # Convert log_level string to enum if needed
        if "log_level" in config_dict and isinstance(config_dict["log_level"], str):
            config_dict["log_level"] = LogLevel[config_dict["log_level"]]
        
        return cls(**config_dict)


# Global configuration instance
_global_config: Optional[FibonacciHashingConfig] = None


def get_global_config() -> FibonacciHashingConfig:
    """
    Get or create global fibonacci hashing configuration.
    
    Returns:
        Global FibonacciHashingConfig instance
    """
    global _global_config
    if _global_config is None:
        _global_config = FibonacciHashingConfig()
    return _global_config


def set_global_config(config: FibonacciHashingConfig) -> None:
    """
    Set global fibonacci hashing configuration.
    
    Args:
        config: FibonacciHashingConfig instance to use globally
        
    Raises:
        TypeError: If config is not FibonacciHashingConfig
    """
    global _global_config
    if not isinstance(config, FibonacciHashingConfig):
        raise TypeError(f"config must be FibonacciHashingConfig, got {type(config).__name__}")
    _global_config = config
    logger.info("Global fibonacci hashing configuration updated")
