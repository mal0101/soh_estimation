# Changelog

All notable changes to this project will be documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Project scaffolding and directory structure
- Conda environment setup with Python 3.11
- Configuration system (`config/default.yaml`, `src/utils/config.py`)
- MLflow experiment tracking integration (`src/utils/tracking.py`)
- Reproducibility utilities (`src/utils/seeding.py`)
- `pyproject.toml` and `setup.py` for project packaging
- Unit test structure (`tests/`)
- Documentation framework (`docs/`)

### Changed
- Fixed `.gitignore` to exclude data and experiment artifacts while tracking docs

## [0.1.0] - 2025-07-23

### Added
- Initial repository with `docs/project_plan.md`
- MIT License
- Base `.gitignore` template
- Empty directory scaffolding: `src/`, `notebooks/`, `data/`, `experiments/`, `report/`
