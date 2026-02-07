# Agent Guide

This document provides instructions and standards for AI agents working on this codebase.

## Development Tools and Tests

To maintain code quality, use the following tools via Docker. Ensure you have built the image first with `docker build -t txt_splitt_dev .`.

**Note:** In some environments, Docker commands may require `sudo` (e.g., `sudo docker build ...` or `sudo docker run ...`). If you encounter permission denied errors, try prefixing the commands with `sudo`.

### 1. Formatting and Linting (Ruff)
Run these commands to verify code style and fix common issues:
```bash
docker run --rm -v $(pwd):/app txt_splitt_dev ruff check .
docker run --rm -v $(pwd):/app txt_splitt_dev ruff format . --check
```

### 2. Static Type Checking (Mypy)
Verify type safety (enforced with `strict = true`):
```bash
docker run --rm -v $(pwd):/app -e PYTHONPATH=src txt_splitt_dev mypy .
```

### 3. Running Tests (Pytest)
Run the test suite to ensure no regressions:
```bash
docker run --rm -v $(pwd):/app -e PYTHONPATH=src txt_splitt_dev pytest
```

## Coding Standards

- **Type Hints:** All Python code **MUST** use type hints. This includes function signatures (arguments and return types), class attributes, and complex variable assignments. Static type checking is strictly enforced.
- **Verification:** Always run `ruff` and `mypy` before finalizing changes. Ensure all tests pass with `pytest`.
- **Dockerized Environment:** All tools should be run within the provided Docker environment to ensure consistency.
