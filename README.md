# txt_splitt

A python package to split text.

## How it Works

The pipeline processes input text through a sequence of stages to produce semantic segments:

1. **HTML Cleaning** (Optional): Strips tags while preserving offset data to allow mapping back to the original source.
2. **Sentence Splitting**: Breaks the text into individual sentence units.
3. **Marking**: Indexes sentences to prepare them for LLM reference.
4. **Semantic Analysis**: Uses an LLM to identify topics and ranges. Supports both single-pass and two-stage (topic extraction followed by assignment) strategies.
5. **Parsing**: Interprets the LLM's output into structured groups.
6. **Gap Handling**: Validates coverage and assigns any skipped sentences or gaps to ensure completeness.
7. **Enhancement** (Optional): Refines boundaries by re-evaluating ambiguous sentences at group edges.
8. **Joining** (Optional): Merges adjacent groups that share the same topic or fit specific criteria.
9. **Offset Restoration** (Optional): Maps the final segmented text back to the original source positions (e.g., restoring HTML context).

## Installation

Install from a GitHub URL:

```bash
pip install "git+https://github.com/rntk/txt-splitt.git"
```

Install a specific tag or branch:

```bash
pip install "git+https://github.com/rntk/txt-splitt.git@v0.1.0"
pip install "git+https://github.com/rntk/txt-splitt.git@main"
```

## LLM Cache Wrappers

The package includes cache wrappers that can sit around any `LLMCallable` or
`AsyncLLMCallable`:

- `CachingLLMCallable`
- `CachingAsyncLLMCallable`
- `MemoryLLMCacheStore`
- `SQLiteLLMCacheStore`

The cache backend is pluggable. If you already use MongoDB, Redis, or another
application store, implement the cache-store protocol and pass your store into
the wrapper. The wrapper owns key generation; your store only needs `get()` and
`set()` operations for `CacheEntry` records.

```python
from txt_splitt import CachingLLMCallable
from txt_splitt.sentences import HierarchicalTopicRangeLLM

cached_client = CachingLLMCallable(
    inner=my_llm_client,
    store=my_mongo_cache_store,
    namespace="topic-range",
    model_id="gpt-4o-mini",
    prompt_version="v1",
)

llm = HierarchicalTopicRangeLLM(cached_client)
```

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
