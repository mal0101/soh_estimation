"""Project path utilities.

Training/preprocessing scripts must produce identical outputs regardless
of the caller's working directory. ``project_root`` walks up from this
file until it finds ``pyproject.toml`` and anchors all artifact paths to
that directory.
"""

from pathlib import Path


def project_root() -> Path:
    """Locate the repository root (the directory containing pyproject.toml).

    Returns:
        Absolute Path to the project root.

    Raises:
        FileNotFoundError: If pyproject.toml cannot be found walking up
            from this module's location.
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError(
        "Could not locate project root: no pyproject.toml found walking up "
        f"from {current}"
    )
