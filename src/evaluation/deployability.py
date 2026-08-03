"""Deployability assessment for trained models.

Benchmarks inference time, measures model file sizes, and evaluates
suitability for embedded BMS deployment.
"""

import logging
import time
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def benchmark_inference_time(
    model: object,
    X: np.ndarray,
    n_repeats: int = 1000,
    is_torch: bool = False,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Benchmark sequential CPU inference time.

    Args:
        model: Trained model with a predict() or __call__() method.
        X: Single input sample or batch for timing.
        n_repeats: Number of repeated inferences.
        is_torch: Whether the model is a PyTorch module.
        device: Torch device (required if is_torch is True).

    Returns:
        Dict with mean_inference_ms, std_inference_ms, p95_ms, p99_ms.
    """
    if is_torch:
        import torch

        model.eval()
        model = model.to(device)
        X_tensor = torch.tensor(X[:1], dtype=torch.float32, device=device)

        with torch.no_grad():
            _ = model(X_tensor)

        times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(X_tensor)
            times.append(time.perf_counter() - t0)
    else:
        x_single = X[:1]
        _ = model.predict(x_single)

        times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            _ = model.predict(x_single)
            times.append(time.perf_counter() - t0)

    times_ms = np.array(times) * 1000
    result = {
        "mean_inference_ms": float(np.mean(times_ms)),
        "std_inference_ms": float(np.std(times_ms)),
        "p95_ms": float(np.percentile(times_ms, 95)),
        "p99_ms": float(np.percentile(times_ms, 99)),
    }
    logger.info(
        "Inference time: %.3f ± %.3f ms (p95=%.3f, p99=%.3f)",
        result["mean_inference_ms"], result["std_inference_ms"],
        result["p95_ms"], result["p99_ms"],
    )
    return result


def model_size_mb(path: str | Path) -> float:
    """Get the file size of a saved model in megabytes.

    Args:
        path: Path to the model file.

    Returns:
        File size in MB.
    """
    size_bytes = Path(path).stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    logger.info("Model size: %.3f MB (%s)", size_mb, path)
    return size_mb


def deployability_report(
    inference_times: dict[str, dict],
    model_sizes: dict[str, float],
    target_rmse: float = 0.02,
    target_maxae: float = 0.05,
    target_inference_ms: float = 200,
) -> dict[str, dict]:
    """Generate a deployability assessment report.

    Args:
        inference_times: Dict mapping model names to inference benchmark results.
        model_sizes: Dict mapping model names to file sizes in MB.
        target_rmse: Target RMSE threshold.
        target_maxae: Target MaxAE threshold.
        target_inference_ms: Target inference time threshold.

    Returns:
        Dict mapping model names to assessment dicts.
    """
    report = {}
    for model_name in inference_times:
        inf = inference_times.get(model_name, {})
        size = model_sizes.get(model_name, float("inf"))
        mean_ms = inf.get("mean_inference_ms", float("inf"))

        report[model_name] = {
            "inference_time_ms": mean_ms,
            "model_size_mb": size,
            "meets_inference_target": mean_ms <= target_inference_ms,
            "meets_size_target": size <= 4.0,
        }
    return report
