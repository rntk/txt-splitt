# txt_splitt

A python package to split text.

## How it Works

The pipeline processes input text through a sequence of stages to produce semantic segments:

1. **HTML Cleaning** (Optional): Strips tags while preserving offset data to allow mapping back to the original source.
2. **Sentence Splitting**: Breaks the text into individual sentence units.
3. **Marking**: Indexes sentences to prepare them for LLM reference.
4. **Semantic Analysis**: Uses an LLM to identify topics and ranges.
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
from txt_splitt.sentences import TopicRangeLLM

cached_client = CachingLLMCallable(
    inner=my_llm_client,
    store=my_mongo_cache_store,
    namespace="topic-range",
    model_id="gpt-4o-mini",
    prompt_version="v1",
)

llm = TopicRangeLLM(cached_client)
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

## Benchmarking topic splitting

`benchmark_text.py` reuses the pipeline in `split_text.py` and writes a JSON
artifact plus a self-contained HTML report with charts. No reference topic
boundaries or external chart services are required. Pass directories (searched
recursively for `.txt`, `.md`, `.html`, and `.htm`) or a list of files:

```bash
./run_benchmark.sh files/ --host http://localhost:8080 \
  --model model-a --label model-a --repeats 3 --warmup 1 \
  --output-dir benchmark_results/model-a

./run_benchmark.sh files/ \
  --host http://localhost:8080 --model model-b --label model-b \
  --repeats 3 --warmup 1 --output-dir benchmark_results/model-b \
  --compare benchmark_results/model-a/benchmark_TIMESTAMP.json
```

You can also pass explicit paths, for example `files/article.txt files/page.html`,
instead of a directory. Use the actual JSON filename printed by the first run. For a fair comparison,
use the same input files, pipeline options, execution mode, hardware, and server
settings for both models. The report flags differences in corpus or settings.
`--model` sends the requested model identifier to the server; with a server that
loads only one model, ensure that the intended model is actually loaded.

The shell wrapper runs Python locally. To run inside the development image:

```bash
docker build -t txt_splitt_dev .
docker run --rm -v "$(pwd):/app" txt_splitt_dev \
  python benchmark_text.py files/ --host http://YOUR_SERVER:8080 \
  --model model-a --label model-a --repeats 3
```

Rebuild or combine reports without calling an LLM (using `--compare` or `--report-only`):

```bash
python benchmark_text.py --compare \
  benchmark_results/model-a/benchmark_TIMESTAMP.json \
  benchmark_results/model-b/benchmark_TIMESTAMP.json \
  --output-dir benchmark_results/comparison

python benchmark_text.py --report-only \
  benchmark_results/model-a/benchmark_TIMESTAMP.json \
  benchmark_results/model-b/benchmark_TIMESTAMP.json \
  --output-dir benchmark_results/comparison
```

Metrics include runtime and repeated-run variation, LLM call counts and durations,
failure rates, throughput, topic and range counts, topic/range sizes in characters
and sentences, immediate subtopics per hierarchy node, and distinct topics in ten
equal sentence-position bins. Coverage, multiple assignment, invalid ranges, and
single-sentence topics help identify structural problems. Raw per-run measurements
and input hashes are retained in JSON for further analysis.

A topic is a unique complete label path; hierarchy prefixes describe its parents.
Topic sizes count each assigned sentence once per topic. Character counts use
cleaned sentence text, excluding HTML markup and inter-sentence gaps. Sentence
units come from the configured splitter and may be shorter than grammatical
sentences. These are descriptive metrics: more topics, fewer topics, and deeper
hierarchies do not automatically imply better semantic quality.

Runs process documents sequentially; request concurrency follows the selected
sync/async mode. Timing excludes file loading and report rendering. Warmup runs
are excluded from measured results. Persistent client response caching is disabled;
server-side prompt caching, load, and model warmup can still affect timing. Summed
LLM durations can exceed wall time when requests run concurrently. Failed runs
remain visible in the report and cause a nonzero process exit after artifacts are
written. Run `python benchmark_text.py --help` for pipeline and execution options.

For future quality comparisons, useful additions would be blind human ratings of
label usefulness/coherence, repeated-run boundary stability, and server-reported
token usage/cost. The current benchmark does not estimate semantic accuracy or
infer token counts from character counts.
