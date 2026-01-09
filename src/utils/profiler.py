"""Performance profiling utilities for CABiNet models."""

from contextlib import contextmanager
import logging
import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """Profile model inference and training performance.

    Tracks metrics like FPS, memory usage, and latency.
    """

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        """Initialize profiler.

        Args:
            model: PyTorch model to profile
            device: Device to run profiling on
        """
        self.model = model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

    @contextmanager
    def timer(self, name: str = "operation"):
        """Context manager for timing operations.

        Args:
            name: Name of the operation being timed

        Yields:
            None

        Example:
            >>> profiler = PerformanceProfiler(model)
            >>> with profiler.timer("forward_pass"):
            ...     output = model(input)
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        yield

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end = time.perf_counter()
        elapsed = (end - start) * 1000  # Convert to ms
        logger.info(f"{name}: {elapsed:.2f} ms")

    def measure_inference_time(
        self,
        input_size: tuple = (1, 3, 512, 512),
        num_iterations: int = 100,
        warmup_iterations: int = 10,
    ) -> Dict[str, float]:
        """Measure model inference time.

        Args:
            input_size: Input tensor shape (B, C, H, W)
            num_iterations: Number of iterations to average over
            warmup_iterations: Number of warmup iterations

        Returns:
            Dictionary with timing statistics
        """
        self.model.eval()

        # Create dummy input
        dummy_input = torch.randn(input_size, device=self.device)

        # Warmup
        logger.info(f"Warmup ({warmup_iterations} iterations)...")
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = self.model(dummy_input)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Actual measurements
        logger.info(f"Measuring inference time ({num_iterations} iterations)...")
        times = []

        with torch.no_grad():
            for _ in range(num_iterations):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                start = time.perf_counter()
                _ = self.model(dummy_input)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                end = time.perf_counter()
                times.append((end - start) * 1000)  # ms

        # Calculate statistics
        import numpy as np

        times_array = np.array(times)

        results = {
            "mean_ms": float(np.mean(times_array)),
            "std_ms": float(np.std(times_array)),
            "min_ms": float(np.min(times_array)),
            "max_ms": float(np.max(times_array)),
            "median_ms": float(np.median(times_array)),
            "fps": float(1000 / np.mean(times_array)),
            "input_size": input_size,
            "num_iterations": num_iterations,
        }

        logger.info(
            f"Average inference time: {results['mean_ms']:.2f} ± {results['std_ms']:.2f} ms"
        )
        logger.info(f"FPS: {results['fps']:.2f}")

        return results

    def measure_memory_usage(
        self,
        input_size: tuple = (1, 3, 512, 512),
    ) -> Dict[str, float]:
        """Measure GPU memory usage during inference.

        Args:
            input_size: Input tensor shape (B, C, H, W)

        Returns:
            Dictionary with memory statistics (in MB)
        """
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, cannot measure GPU memory")
            return {}

        self.model.eval()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        # Measure baseline
        baseline_memory = torch.cuda.memory_allocated() / 1024**2  # MB

        # Create input and run forward pass
        dummy_input = torch.randn(input_size, device=self.device)

        with torch.no_grad():
            _ = self.model(dummy_input)

        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        current_memory = torch.cuda.memory_allocated() / 1024**2  # MB

        results = {
            "baseline_mb": float(baseline_memory),
            "peak_mb": float(peak_memory),
            "current_mb": float(current_memory),
            "inference_mb": float(peak_memory - baseline_memory),
            "input_size": input_size,
        }

        logger.info(f"Peak memory usage: {results['peak_mb']:.2f} MB")
        logger.info(f"Inference memory: {results['inference_mb']:.2f} MB")

        return results

    def profile_model_flops(
        self, input_size: tuple = (1, 3, 512, 512)
    ) -> Dict[str, Any]:
        """Profile model FLOPs using torch.profiler.

        Args:
            input_size: Input tensor shape (B, C, H, W)

        Returns:
            Dictionary with profiling information
        """
        try:
            from torch.profiler import ProfilerActivity, profile

            dummy_input = torch.randn(input_size, device=self.device)

            with profile(
                activities=(
                    [ProfilerActivity.CPU, ProfilerActivity.CUDA]
                    if torch.cuda.is_available()
                    else [ProfilerActivity.CPU]
                ),
                record_shapes=True,
            ) as prof:
                with torch.no_grad():
                    _ = self.model(dummy_input)

            # Get key averages
            events = prof.key_averages()

            results = {
                "total_cpu_time_ms": sum(evt.cpu_time_total for evt in events) / 1000,
                "total_cuda_time_ms": (
                    sum(evt.cuda_time_total for evt in events) / 1000
                    if torch.cuda.is_available()
                    else 0
                ),
                "input_size": input_size,
            }

            logger.info(f"Total CPU time: {results['total_cpu_time_ms']:.2f} ms")
            if torch.cuda.is_available():
                logger.info(f"Total CUDA time: {results['total_cuda_time_ms']:.2f} ms")

            return results

        except ImportError:
            logger.warning("torch.profiler not available")
            return {}

    def run_full_benchmark(
        self,
        input_size: tuple = (1, 3, 512, 512),
        num_iterations: int = 100,
    ) -> Dict[str, Any]:
        """Run comprehensive benchmark.

        Args:
            input_size: Input tensor shape (B, C, H, W)
            num_iterations: Number of iterations for timing

        Returns:
            Dictionary with all benchmark results
        """
        logger.info("=" * 60)
        logger.info(f"Running benchmark for input size: {input_size}")
        logger.info("=" * 60)

        results = {
            "input_size": input_size,
            "device": str(self.device),
            "model": type(self.model).__name__,
        }

        # Timing benchmark
        logger.info("\n[1/3] Measuring inference time...")
        timing_results = self.measure_inference_time(input_size, num_iterations)
        results["timing"] = timing_results

        # Memory benchmark
        if torch.cuda.is_available():
            logger.info("\n[2/3] Measuring memory usage...")
            memory_results = self.measure_memory_usage(input_size)
            results["memory"] = memory_results

        # FLOPs profiling
        logger.info("\n[3/3] Profiling model FLOPs...")
        flops_results = self.profile_model_flops(input_size)
        results["flops"] = flops_results

        logger.info("\n" + "=" * 60)
        logger.info("Benchmark complete!")
        logger.info("=" * 60)

        return results


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    results = {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": total_params - trainable_params,
        "total_millions": total_params / 1e6,
    }

    logger.info(f"Total parameters: {results['total_millions']:.2f}M")
    logger.info(f"Trainable parameters: {trainable_params / 1e6:.2f}M")

    return results


if __name__ == "__main__":
    # Example usage
    from src.models.cabinet import CABiNet

    logging.basicConfig(level=logging.INFO)

    model = CABiNet(n_classes=19, mode="large")
    profiler = PerformanceProfiler(model)

    # Count parameters
    param_count = count_parameters(model)

    # Run benchmark
    results = profiler.run_full_benchmark(
        input_size=(1, 3, 512, 512),
        num_iterations=50,
    )

    print("\nResults:")
    print(f"Parameters: {param_count['total_millions']:.2f}M")
    print(f"Average inference: {results['timing']['mean_ms']:.2f} ms")
    print(f"FPS: {results['timing']['fps']:.2f}")
