# txt_splitt

A python package to split text.

## Development with Docker

Prerequisites:
- Docker

### 1. Build the Docker image

```bash
docker build -t txt_splitt_dev .
```

### 2. Run Checks and Tests

You can mount your current directory into the container to run checks on your code.

**Run format and lint (Ruff):**
```bash
docker run --rm -v $(pwd):/app txt_splitt_dev ruff check .
docker run --rm -v $(pwd):/app txt_splitt_dev ruff format . --check
```

**Run static type checks (Mypy):**
```bash
# Note: You need to set PYTHONPATH so mypy can find the src directory
docker run --rm -v $(pwd):/app -e PYTHONPATH=src txt_splitt_dev mypy .
```

**Run tests (Pytest):**
```bash
docker run --rm -v $(pwd):/app -e PYTHONPATH=src txt_splitt_dev pytest
```

### 3. Interactive Shell

To enter the container environment:

```bash
docker run --rm -it -v $(pwd):/app -e PYTHONPATH=src txt_splitt_dev bash
```
