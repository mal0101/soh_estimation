"""Phase-based error analysis for SOH estimation.

Breaks down prediction errors into SOH degradation phases: early
(SOH 0.90–1.01, inclusive of the 1.0 cap), mid (0.80–0.90), and
end (<0.80). Phases are half-open [min, max) EXCEPT the numerically
largest phase, which is closed on both ends so samples sitting exactly
at the SOH cap are not silently excluded.
"""

import logging

import numpy as np

from src.evaluation.metrics import compute_all_metrics

logger = logging.getLogger(__name__)

PHASE_RANGES = {
    "early": (0.90, 1.01),
    "mid": (0.80, 0.90),
    "end": (0.0, 0.80),
}


def phase_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    soh: np.ndarray,
    phase_ranges: dict[str, tuple[float, float]] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute RMSE within each SOH degradation phase.

    Args:
        y_true: Ground-truth SOH values.
        y_pred: Predicted SOH values.
        soh: SOH values used to determine phase membership.
        phase_ranges: Dict mapping phase names to (min_soh, max_soh).
            Defaults to PHASE_RANGES.

    Returns:
        Dict mapping phase names to metric dicts (rmse, mae, n_samples).
    """
    if phase_ranges is None:
        phase_ranges = PHASE_RANGES

    # The topmost phase includes its upper bound so capped samples
    # (SOH == soh_cap == 1.0) fall inside it.
    upper_bound = max(mx for _, mx in phase_ranges.values())

    results = {}
    for phase_name, (min_soh, max_soh) in phase_ranges.items():
        if max_soh >= upper_bound:
            mask = (soh >= min_soh) & (soh <= max_soh)
        else:
            mask = (soh >= min_soh) & (soh < max_soh)
        n = int(mask.sum())
        if n == 0:
            results[phase_name] = {"rmse": np.nan, "mae": np.nan, "n_samples": 0}
            continue
        metrics = compute_all_metrics(y_true[mask], y_pred[mask])
        metrics["n_samples"] = n
        results[phase_name] = metrics
        logger.info("Phase %s (n=%d): RMSE=%.6f", phase_name, n, metrics["rmse"])

    return results


def phase_summary_table(phase_results: dict[str, dict]) -> str:
    """Format phase results as a readable table string.

    Args:
        phase_results: Output of phase_rmse.

    Returns:
        Formatted table string.
    """
    lines = [f"{'Phase':<10} {'RMSE':>10} {'MAE':>10} {'N':>6}"]
    lines.append("-" * 40)
    for phase, metrics in phase_results.items():
        rmse = metrics["rmse"]
        mae = metrics["mae"]
        n = metrics["n_samples"]
        rmse_str = f"{rmse:.6f}" if not np.isnan(rmse) else "N/A"
        mae_str = f"{mae:.6f}" if not np.isnan(mae) else "N/A"
        lines.append(f"{phase:<10} {rmse_str:>10} {mae_str:>10} {n:>6}")
    return "\n".join(lines)
