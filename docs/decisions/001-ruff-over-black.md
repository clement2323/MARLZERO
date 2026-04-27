# 001 — ruff as sole formatter (no black)

**Date:** 2026-04  
**Status:** accepted

## Context

`CLAUDE.md` lists "Black, ruff, mypy" for pre-commit hooks. Since ruff v0.1, ruff
ships a built-in formatter that is Black-compatible (same opinionated style, same
line length semantics). Running both would add a redundant round-trip with no
difference in output.

## Decision

Use ruff for both linting and formatting. Black is not declared as a dependency
and is not in `.pre-commit-config.yaml`.

## Consequences

- One fewer dev dependency and pre-commit hook.
- Output is Black-compatible, so contributors using Black locally will see no diff.
- A single tool is easier to version-pin and update (one `rev:` instead of two).
