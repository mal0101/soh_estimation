"""Model interpretability via SHAP and Captum.

Provides SHAP TreeExplainer for tree-based models and Captum
Integrated Gradients for deep learning models.
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def shap_tree_explainer(
    model: Any,
    X: np.ndarray,
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    """Compute SHAP values for a tree-based model.

    Args:
        model: Trained sklearn model (RF, GBR, etc.).
        X: Feature matrix for background/reference data.
        feature_names: Optional list of feature names.

    Returns:
        Dict with keys 'shap_values', 'feature_importance', 'expected_value'.
    """
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    shap_vals = shap_values[0] if isinstance(shap_values, list) else shap_values

    importance = np.abs(shap_vals).mean(axis=0)

    return {
        "shap_values": shap_vals,
        "feature_importance": importance,
        "expected_value": float(np.asarray(explainer.expected_value).ravel()[0]),
        "feature_names": feature_names,
    }


def captum_integrated_gradients(
    model: Any,
    X: np.ndarray,
    device: Any,
    n_steps: int = 50,
) -> dict[str, Any]:
    """Compute Integrated Gradients attributions for a PyTorch model.

    Args:
        model: Trained PyTorch model.
        X: Input tensor of shape [n_samples, seq_len, n_features].
        device: Torch device.
        n_steps: Number of interpolation steps.

    Returns:
        Dict with keys 'attributions', 'feature_importance'.
    """
    import torch
    from captum.attr import IntegratedGradients

    if X.ndim != 3:
        raise ValueError(f"Expected 3D input array [n_samples, seq_len, n_features], got {X.ndim}D")

    model.eval()
    model = model.to(device)

    def forward_func(inputs: torch.Tensor) -> torch.Tensor:
        out = model(inputs).squeeze(-1)
        return torch.as_tensor(out)

    ig = IntegratedGradients(forward_func)

    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    X_tensor.requires_grad_(True)

    attributions = ig.attribute(X_tensor, n_steps=n_steps, return_convergence_delta=False)
    attr_np = attributions.detach().cpu().numpy()

    importance = np.abs(attr_np).mean(axis=(0, 1))

    return {
        "attributions": attr_np,
        "feature_importance": np.asarray(importance),
    }
