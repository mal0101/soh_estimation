"""Consistency gate: mechanically verify reported numbers match artifacts.

Checks (fail loudly on any drift):
    1. README results block equals the YAML-derived tables.
    2. Executed notebooks' stored outputs contain the CURRENT canonical
       metrics from experiments/*_results_*.yaml and contain none of the
       known-stale pre-remediation values.
    3. No stale unsuffixed artifact references remain in tracked docs/code.

Usage:
    python scripts/verify_consistency.py
"""

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stale pre-remediation values, checked ONLY inside nb05's results-table
# outputs (whole-notebook scans false-positive: e.g. a current phase RMSE can
# legitimately equal an old headline number by coincidence).
NB05_STALE_MARKERS = ["0.059923", "0.036633", "-2.504108", "-0.134789", "-0.142490"]
# Files allowed to mention removed artifact names (they document the removal).
LEGACY_DOC_ALLOWLIST = {"docs/data_dictionary.md"}
UNSUFFIXED_PATTERNS = ["feature_matrix.parquet", "classical_results.yaml", "dl_results.yaml"]


def _nb05_table_text(nb_path: Path) -> str:
    """Output text of nb05's results-printing cells only (cells 3 and 11)."""
    d = json.loads(nb_path.read_text())
    parts = []
    for i in (3, 11):
        for o in d["cells"][i].get("outputs", []):
            parts += o.get("text", [])
            if "text/plain" in o.get("data", {}):
                parts += o["data"]["text/plain"]
    return "".join(parts)


def notebook_output_text(nb_path: Path) -> str:
    d = json.loads(nb_path.read_text())
    parts = []
    for cell in d["cells"]:
        if cell["cell_type"] != "code":
            continue
        for o in cell.get("outputs", []):
            parts += o.get("text", [])
            if "text/plain" in o.get("data", {}):
                parts += o["data"]["text/plain"]
    return "".join(parts)


def main() -> int:
    failures: list[str] = []

    # --- 1. README block freshness -------------------------------------
    import scripts.update_results_docs as urd  # noqa: PLC0415

    expected_block = urd.build_readme_block()
    readme = (ROOT / "README.md").read_text()
    actual_block = readme.split("<!-- RESULTS:AUTO -->")[1].split("<!-- /RESULTS:AUTO -->")[0]
    if actual_block.strip() != expected_block.strip():
        failures.append("README RESULTS block does not match YAML-derived tables")

    # --- 2. Notebook outputs vs canonical YAMLs -------------------------
    checks = {
        "05_dl_training_analysis.ipynb": ("nasa", ["cell 3/11 table"]),
        "06_final_comparison.ipynb": ("calce", []),
        "04_classical_ml.ipynb": ("all", []),
    }
    for nb_name, (ds, _extra) in checks.items():
        nb_path = ROOT / "notebooks" / nb_name
        text = notebook_output_text(nb_path)
        if nb_name == "05_dl_training_analysis.ipynb":
            table_text = _nb05_table_text(nb_path)
            for marker in NB05_STALE_MARKERS:
                if marker in table_text:
                    failures.append(f"{nb_name}: stale marker {marker!r} in results tables")
        cn = yaml.safe_load((ROOT / "experiments" / f"classical_results_{ds}.yaml").read_text())
        dl = yaml.safe_load((ROOT / "experiments" / f"dl_results_{ds}.yaml").read_text())
        missing = 0
        total = 0
        for _m, v in {**cn, **dl}.items():
            if not (isinstance(v, dict) and "rmse_mean" in v):
                continue
            total += 1
            if f"{v['rmse_mean']:.6f}" not in text and f"{v['rmse_mean']:.4f}" not in text:
                missing += 1
        if missing:
            failures.append(
                f"{nb_name}: {missing}/{total} canonical rmse values absent from outputs"
            )

    # --- 3. Stale references in code/docs -------------------------------
    scan_ext = {".py", ".md"}
    skip_parts = {".git", "__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache"}
    for path in ROOT.rglob("*"):
        if path.suffix not in scan_ext or any(p in skip_parts for p in path.parts):
            continue
        rel = path.relative_to(ROOT)
        # CHANGELOG historical entries and the data-dictionary removal list
        # legitimately quote old names; everything else must be clean.
        if str(rel) == "CHANGELOG.md" or str(rel) in LEGACY_DOC_ALLOWLIST or rel == Path("scripts") / "verify_consistency.py":
            continue
        text = path.read_text(errors="ignore")
        for pat in UNSUFFIXED_PATTERNS:
            if pat in text and rel != Path("docs") / "execution_plan.md":
                failures.append(f"{rel}: stale reference {pat!r}")

    if failures:
        print("CONSISTENCY GATE FAILED:")
        for f in failures:
            print(" -", f)
        return 1
    print("CONSISTENCY GATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
