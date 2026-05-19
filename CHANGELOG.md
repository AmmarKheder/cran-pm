# Changelog

All notable changes to **CRAN-PM** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-05-18

### Added
- Initial public package layout (`src/cranpm/`).
- `pyproject.toml` with hatchling build backend.
- MIT license.
- Configuration system (`cranpm.config`) for paths via env vars.
- Public API for inference: `CRANPMForecaster.from_pretrained()` and `.predict()`.
- CLI entry point `cranpm`.
- GPU module (`cranpm.gpu`): single-GPU inference, multi-GPU DDP wrapper, ROCm + CUDA support.
- Reproducible benchmarks for inference throughput / memory.
- Reproducible CPU/CUDA `docker/Dockerfile`, published to GHCR
  (`ghcr.io/ammarkheder/cran-pm`) by a release CI workflow.
- pytest test suite with unit + integration tests.
- Interactive Hugging Face Space demonstrator (genuine forward passes)
  and project website.

First public release. Migrated from the research codebase
(`topoflow_europe/`, ECCV 2026 submission) into a properly packaged
scientific software tool intended for community use and GMD publication.
