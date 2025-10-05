# Development

## Environment Setup

```bash
# clone repository
git clone https://github.com/nterrel/ani-mm.git
cd ani-mm

# create environment (conda example)
conda env create -f environment.yml
conda activate ani-mm

# editable install with dev + docs deps
pip install -e .[docs]
```

## Running Tests

```bash
pytest -q
```

## Style and Type Hints

Keep changes minimal and consistent with existing style. Explicit type hints are welcome where they improve clarity.

## Documentation

MkDocs + Material + mkdocstrings are used. To serve locally:

```bash
mkdocs serve
```

## Releasing (Manual Outline)

1. Update `animm/version.py` if needed.
2. Update CHANGELOG (future file) and push a tag: `git tag -a vX.Y.Z -m "vX.Y.Z"`.
3. Build & upload (to be automated later).

## Roadmap Snapshot

See README for full roadmap. Near-term:

- PBC / periodic support.
- Mixed precision exploration.
- Expanded example gallery.

## Contributing

Open a PR with focused commits and clear descriptions. Include or update tests for behavior changes.
