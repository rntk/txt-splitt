FROM python:3.14-slim

WORKDIR /app

# Install development dependencies directly
# We do this to avoid needing the full source code to install the [dev] extra
# if usage of `pip install .[dev]` was intended.
# Since we are mounting the volume, we can also install the package in editable mode
# inside the container at runtime if needed, or just PYTHONPATH.

RUN pip install --no-cache-dir ruff mypy pytest prometheus_client

# Keep the container running or ready for commands
CMD ["bash"]
