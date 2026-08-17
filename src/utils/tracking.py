"""MLflow experiment tracking helpers.

Provides thin wrappers around MLflow to log parameters, metrics, artifacts,
and models consistently across all training runs.
"""

import logging
from pathlib import Path
from typing import Any

import mlflow

logger = logging.getLogger(__name__)


def init_tracking(tracking_uri: str = "mlruns", experiment_name: str = "soh_benchmark") -> None:
    """Initialize MLflow tracking with a local backend.

    Args:
        tracking_uri: Path to the MLflow backend store.
        experiment_name: Name of the experiment to log runs under.
    """
    if tracking_uri.startswith("sqlite"):
        mlflow.set_tracking_uri(tracking_uri)
    else:
        mlflow.set_tracking_uri(Path(tracking_uri).resolve().as_uri())
    mlflow.set_experiment(experiment_name)
    logger.info("MLflow tracking initialized: uri=%s, experiment=%s", tracking_uri, experiment_name)


def start_run(run_name: str | None = None, tags: dict[str, str] | None = None) -> Any:
    """Start a new MLflow run.

    Args:
        run_name: Optional human-readable name for the run.
        tags: Optional key-value tags to attach to the run.

    Returns:
        The active MLflow run context.
    """
    return mlflow.start_run(run_name=run_name, tags=tags or {})


def log_params(params: dict[str, Any]) -> None:
    """Log a flat dictionary of parameters.

    Nested dicts are flattened with dot-separated keys.

    Args:
        params: Parameters to log (e.g., {'model': {'lr': 0.001, 'epochs': 100}}).
    """
    flat = _flatten_dict(params)
    mlflow.log_params(flat)
    logger.debug("Logged %d parameters", len(flat))


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Log a dictionary of scalar metrics.

    Args:
        metrics: Metric name-value pairs.
        step: Optional step number (e.g., epoch or fold index).
    """
    mlflow.log_metrics(metrics, step=step)
    logger.debug("Logged %d metrics at step %s", len(metrics), step)


def log_artifact(local_path: str | Path, artifact_dir: str | None = None) -> None:
    """Log a local file as an MLflow artifact.

    Args:
        local_path: Path to the file to log.
        artifact_dir: Optional subdirectory within the run's artifact store.
    """
    mlflow.log_artifact(str(local_path), artifact_path=artifact_dir)
    logger.debug("Logged artifact: %s -> %s", local_path, artifact_dir)


def log_model(model: Any, artifact_path: str) -> None:
    """Log a trained model as an MLflow artifact.

    Args:
        model: The trained model object (sklearn, PyTorch, etc.).
        artifact_path: Subdirectory name for the model artifact.
    """
    mlflow.sklearn.log_model(model, artifact_path=artifact_path)
    logger.debug("Logged model to artifact path: %s", artifact_path)


def log_figures(figures: dict[str, Any], artifact_dir: str = "figures") -> None:
    """Log matplotlib figures as artifacts.

    Args:
        figures: Dict mapping figure names to matplotlib Figure objects.
        artifact_dir: Subdirectory for the figure artifacts.
    """
    import tempfile

    for _name, fig in figures.items():
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            fig.savefig(tmp.name, dpi=150, bbox_inches="tight")
            log_artifact(tmp.name, artifact_dir=artifact_dir)
            Path(tmp.name).unlink()


def _flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """Flatten a nested dictionary with dot-separated keys.

    Args:
        d: Dictionary to flatten.
        parent_key: Prefix for keys.
        sep: Separator between key levels.

    Returns:
        Flattened dictionary.
    """
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
