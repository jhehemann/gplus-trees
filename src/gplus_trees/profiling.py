"""Performance profiling utilities for G+ tree operations."""

import time
import functools
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
import statistics
from collections import defaultdict

@dataclass
class MethodMetrics:
    """Statistics for a single method's performance."""
    call_count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    times: List[float] = None
    
    def __post_init__(self):
        if self.times is None:
            self.times = []
    
    def add_measurement(self, elapsed: float) -> None:
        """Add a new execution time measurement."""
        self.call_count += 1
        self.total_time += elapsed
        self.min_time = min(self.min_time, elapsed)
        self.max_time = max(self.max_time, elapsed)
        self.times.append(elapsed)
    
    @property
    def avg_time(self) -> float:
        """Calculate average execution time."""
        return self.total_time / self.call_count if self.call_count > 0 else 0
    
    @property
    def median_time(self) -> float:
        """Calculate median execution time."""
        return statistics.median(self.times) if self.times else 0
    
    def __str__(self) -> str:
        return (f"Calls: {self.call_count}, "
                f"Total: {self.total_time:.6f}s, "
                f"Avg: {self.avg_time:.6f}s, "
                f"Min: {self.min_time:.6f}s, "
                f"Max: {self.max_time:.6f}s, "
                f"Median: {self.median_time:.6f}s")


class PerformanceTracker:
    """Central performance metrics collector for GPlusTree operations."""
    
    _instance = None
    
    @classmethod
    def get_instance(cls) -> 'PerformanceTracker':
        """Singleton accessor."""
        if cls._instance is None:
            cls._instance = PerformanceTracker()
        return cls._instance
    
    def __init__(self):
        self.metrics: Dict[str, MethodMetrics] = defaultdict(MethodMetrics)
        self.enabled = True
    
    def add_measurement(self, method_name: str, elapsed: float) -> None:
        """Record a method's execution time."""
        if self.enabled:
            self.metrics[method_name].add_measurement(elapsed)
    
    def reset(self) -> None:
        """Clear all measurements."""
        self.metrics.clear()
    
    def enable(self) -> None:
        """Enable performance tracking."""
        self.enabled = True
    
    def disable(self) -> None:
        """Disable performance tracking."""
        self.enabled = False
    
    def report(self, sort_by: str = 'total_time') -> str:
        """Generate a performance report."""
        if not self.metrics:
            return "No performance data collected."
        
        # # Calculate total execution time across all methods
        # total_execution_time = sum(metrics.total_time for metrics in self.metrics.values())
        # total_call_count = sum(metrics.call_count for metrics in self.metrics.values())

        lines = ["Performance Metrics:"]
        lines.append("-" * 80)
        lines.append(f"{'Method':<40} {'Calls':>8} {'Total (s)':>12} {'Avg (s)':>12} {'Median (s)':>12}")
        lines.append("-" * 80)
        
        # Sort metrics based on the requested attribute
        sorted_items = sorted(
            self.metrics.items(),
            key=lambda x: getattr(x[1], sort_by) if sort_by != 'call_count' else x[1].call_count,
            reverse=True
        )
        
        for method_name, metrics in sorted_items:
            lines.append(f"{method_name:<40} {metrics.call_count:>8} {metrics.total_time:>12.6f} "
                         f"{metrics.avg_time:>12.6f} {metrics.median_time:>12.6f}")

        # lines.append(f"Execution time (G-Tree/KList): {total_execution_time:.6f}s across {total_call_count} calls")    
        
        return "\n".join(lines)


def track_performance(method: Optional[Callable] = None, *, 
                      tag: Optional[str] = None) -> Callable:
    """
    Decorator to track method execution time.
    
    Args:
        method: The method to track
        tag: Optional custom tag to use instead of method name
    
    Returns:
        Decorated method with performance tracking
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not PerformanceTracker.get_instance().enabled:
                return func(*args, **kwargs)
                
            method_name = tag or f"{func.__qualname__}"
            start_time = time.perf_counter()  # Using perf_counter instead of time
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start_time  # Using perf_counter here too
            PerformanceTracker.get_instance().add_measurement(method_name, elapsed)
            return result
        return wrapper
    
    # Handle both @track_performance and @track_performance(tag="name") forms
    if method is None:
        return decorator
    return decorator(method)