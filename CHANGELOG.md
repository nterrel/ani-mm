# Changelog

All notable changes will be documented in this file. The format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and versions follow semantic versioning.

## Unreleased

### Added

* Live viewer element coloring and stable scatter updates.
* `--live-hold` flag to keep Matplotlib window open after simulation.
* ASE fallback diagnostics including `tkinter` probe hints.
* Dedicated `docs/live-viewer.md` page and documentation overhaul.

### Changed

* README reorganized for clarity (scope, install variants, precision cache docs).
* Standardized markdown bullet style across docs.

### Fixed

* Avoid recreation of Matplotlib scatter every frame (performance + flicker).

## 0.2.0 - 2025-??-??

* Initial publicized 0.2 baseline (ANI model loading, TorchForce integration, alanine example, basic MD runner, tracing cache, initial docs).
