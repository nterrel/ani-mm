# Development

## Environment Setup

```bash
git clone --recurse-submodules https://github.com/nterrel/ani-mm.git
cd ani-mm
conda env create -f environment.yml
conda activate ani-mm
pip install -e .[docs]
```

Optional (install TorchANI from PyPI instead of submodule):

```bash
pip install torchani
```

## Running Tests

```bash
pytest -q
```

## Lint / Style

The code favors brevity and explicit provenance. Add type hints when they materially clarify shapes or units. Avoid large-formatting sweeps unrelated to a change.

## Documentation Authoring

Serve docs locally:

```bash
mkdocs serve
```

API pages are generated via mkdocstrings; keep public functions minimal. If adding new surface functions, ensure concise docstrings (inputs, outputs, units).

## Release (Manual Skeleton)

1. Bump version in `animm/version.py`.
2. Update CHANGELOG (add section for new version).
3. Tag: `git tag -a vX.Y.Z -m "vX.Y.Z"` and push tags.
4. (Future) Build & upload via automated workflow.

## Advanced Install Notes

* Apple Silicon: default environment targets arm64; if mixing wheels, ensure consistent architecture.
* Thread limiting: set `ANIMM_NO_OMP=1` for reproducible small benchmarks.
* Duplicate OpenMP (macOS): `--allow-dup-omp` sets `KMP_DUPLICATE_LIB_OK=TRUE` (only if you encounter aborts).
* GPU: choose platform with `--platform CUDA` (or rely on OpenMM selection). No explicit GPU-only code path in this layer.

## Provenance Attributes

Each `TorchForce` constructed here has:

* `_animm_traced_dtype`: actual dtype of traced TorchScript graph (str)
* `_animm_cache_hit`: bool indicating if retrieved from in‑process cache

These bubble into result dictionaries (`traced_dtype`, `cache_hit`).

## Full Roadmap

Near-term (core capabilities):

| Category | Item | Status |
|----------|------|--------|
| Core | Float64‑first trace + fallback | Done |
| Core | Trace cache with provenance | Done |
| Core | Periodic / cutoffs (neighbor list) | Planned |
| Core | Restart / checkpoint support | Planned |
| Reporting | Rich reporters (CSV / JSON structured) | Planned |
| Reporting | Live energy plot (matplotlib) | Exploratory |
| MD | Multiple ANI forces (composite) | Investigate |
| Performance | Mixed precision / autocast knob | Planned |
| Examples | Additional molecules / small benchmark set | Planned |
| Tooling | GitHub Actions CI matrix | Planned |
| Tooling | Publish to PyPI with `ml` extra | Planned |
| Docs | Performance tips page | Planned |

Stretch / research ideas:

* Adaptive precision (start float32, re‑trace float64 on instability triggers)
* Simple PBC adaptation if/when suitable ANI periodic variant stabilized
* On‑the‑fly model ensemble energy variance (uncertainty hint)

## Contribution Guidelines

* Small, focused PRs (single feature or fix)
* Include/update tests when behavior changes
* Keep README light; deep details belong in docs
* Document new provenance attributes explicitly

## Testing Focus Areas

* Energy/forces determinism (within tolerance) across successive cache hits
* Alanine MD short runs (verify summary line & dtype metadata)
* SMILES conversion edge cases (unsupported elements → clear error)

## Performance Micro-Benchmarks (Suggested)

Not yet automated; consider adding a lightweight script to compare first-run (trace) vs cached MD step times for a small system (<50 atoms).

## Local Debug Tips

* Set `PYTORCH_JIT_LOG_LEVEL=>>` only when diagnosing tracing (can be verbose)
* Use `--debug` instead of globally elevating logging; third‑party noise is down‑leveled automatically

---
For user-facing quick starts see the README; for usage recipes see `user-guide.md`.
