# glean

Local RAG (Retrieval-Augmented Generation) for codebases, infrastructure configs, and project documents. Ask questions in plain English and get answers grounded in your actual source files — all running on your machine with no data sent to external services.

Works beyond code: query Ansible inventories for server IPs, ask about project mandates, look up deployment configurations, or explore documentation across multiple repositories.

Built on [Ollama](https://ollama.com) for embeddings and generation, [ChromaDB](https://www.trychroma.com) for vector storage, and [rank-bm25](https://github.com/dorianbrown/rank_bm25) for keyword search.

---

## Features

- **Hybrid search** — combines dense vector similarity (semantic) with BM25 keyword search, merged via Reciprocal Rank Fusion; finds both conceptually related content and exact-match lookups like hostnames or config values
- **Incremental indexing** — only re-indexes files that have changed since the last run
- **Language-aware chunking** — Python, TypeScript/JavaScript, PHP, Markdown, YAML, JSON, shell scripts, and INI-style configs each get a tailored chunking strategy
- **Multi-collection support** — organize different projects or codebases into named collections and query them together or individually
- **Interactive REPL** — conversational mode with live collection and model switching
- **Web GUI** — FastAPI + HTMX local web app with streaming responses and index management
- **Fully local** — embeddings, vector storage, and generation all run on your own hardware

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally (default: `http://localhost:11434`)
- An Ollama embedding model (recommended: `nomic-embed-text`)
- An Ollama generation model (recommended: `qwen2.5:14b` or `qwen2.5-coder:7b`)

---

## Installation

**1. Install Python dependencies:**

```bash
pip install chromadb ollama pyyaml rich rank-bm25
```

**2. Pull the required Ollama models:**

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5-coder:7b   # or whichever generation model you prefer
```

`rank-bm25` is optional at runtime. If it is not installed, glean still works with dense retrieval only (BM25 and rare-term ranking are skipped).

**3. Create a `glean.yaml` configuration file** in the directory where you'll run glean (see [Configuration](#configuration) below).

**4. Create a symlink** (optional, for system-wide access):

```bash
sudo ln -s "$(pwd)/glean.py" /usr/local/bin/glean
chmod +x glean.py
```

---

## Quick Start

```bash
# Index your configured collections
python3 glean.py --index

# Ask a question
python3 glean.py "How does authentication work in this codebase?"

# Look up infrastructure details
python3 glean.py "What is the public IP address of the production server?"

# Ask about a specific project
python3 glean.py -c myproject "What is the database schema for user accounts?"

# Start the interactive REPL
python3 glean.py -i
```

---

## Web GUI

A local web interface (FastAPI + HTMX) is available. No build step, no npm.

```bash
# Install GUI dependencies (once)
pip install -r requirements-gui.txt

# Start the server (default port 7777)
python3 server.py
```

Then open `http://localhost:7777`. Use `GLEAN_CONFIG` and `GLEAN_PORT` to override config path and port.

Environment overrides:

```bash
GLEAN_CONFIG=/path/to/glean.yaml GLEAN_PORT=7777 python3 server.py
```

The GUI supports:
- Streaming token-by-token responses
- Collection and model selection from the sidebar
- One-click incremental update or full re-index per collection
- Live index status (file counts, chunk counts, last-indexed timestamps)
- Automatic Ollama model discovery in the model dropdown (fallback to configured model if discovery fails)

Note: when you reindex from the GUI (`/index` or `/reindex`), the running server reloads its BM25 corpus automatically. If you reindex externally with `glean.py --reindex`, restart `server.py` to refresh in-memory BM25 state.

---

## Configuration

glean is configured via a YAML file. By default it looks for `glean.yaml` in the current directory. Use `--config` to specify a different path.

The YAML example below is a practical starting point; exact built-in defaults are listed in **Configuration reference**.

```yaml
# Models
embedding_model: nomic-embed-text
generation_model: qwen2.5-coder:7b

# Ollama server URL
ollama_url: http://localhost:11434

# Where to store the vector database, index state, and BM25 corpus
state_dir: ~/.local/share/glean

# Maximum characters of retrieved context to send to the LLM
# Smaller values keep the model focused on the most relevant chunks.
# (~24000 chars ≈ ~6000 tokens at ~4 chars/token)
max_context_chars: 24000

# Dense retrieval: chunks fetched per collection per query
top_k: 30

# Sparse retrieval: BM25 candidates per query (searched across all collections)
# Higher values cast a wider net for exact-match lookups at the cost of more noise.
bm25_top_k: 50

# Reciprocal Rank Fusion constant k (higher = flatter score distribution)
rrf_k: 60

# Cosine distance threshold: chunks with distance > this value are dropped.
# null = no filtering. Lower = stricter (0.5 is very strict, 0.7 is permissive).
max_distance: null

collections:
  myproject:
    paths:
      - ~/code/myproject
    include:
      - "*.py"
      - "*.md"
      - "*.yaml"
    exclude:
      - ".git/"
      - "__pycache__/"
      - "node_modules/"
      - "*.min.js"

  docs:
    paths:
      - ~/docs/runbooks
      - ~/docs/architecture
    include:
      - "*.md"
      - "*.txt"

  infra:
    paths:
      - ~/code/ansible-playbooks
    include:
      - "*.yaml"
      - "*.yml"
      - "*.conf"
      - "*.sh"
    exclude:
      - ".git/"
```

### Configuration reference

| Key | Default | Description |
|-----|---------|-------------|
| `embedding_model` | `nomic-embed-text` | Ollama model used to generate embeddings |
| `generation_model` | `qwen2.5:14b` | Ollama model used to generate answers |
| `ollama_url` | `http://localhost:11434` | URL of the Ollama server |
| `state_dir` | `~/.local/share/glean` | Directory for ChromaDB, index state, and BM25 corpus |
| `max_context_chars` | `24000` | Character limit for context sent to the LLM |
| `top_k` | `30` | Chunks retrieved per collection via dense (vector) search |
| `bm25_top_k` | `50` | Candidate chunks retrieved via BM25 keyword search |
| `rrf_k` | `60` | RRF constant; higher values flatten score differences between ranks |
| `max_distance` | `null` | Cosine distance cutoff for dense results (`null` = no cutoff) |
| `collections` | *(required)* | Named collections to index (see below) |

### Collection options

| Key | Default | Description |
|-----|---------|-------------|
| `paths` | *(required)* | List of directories to index |
| `include` | `["*.py","*.md","*.yaml","*.yml"]` | Glob patterns for files to include |
| `exclude` | `[]` | Glob patterns or `dir/`-style directory names to skip |

**Directory exclusion:** append `/` to a pattern to exclude all directories with that name at any depth, e.g. `node_modules/` or `.git/`.

**Files larger than 500 KB are always skipped.** Common secret files (`.env`, `*.pem`, `*.key`, `id_rsa`, `credentials.json`, etc.) are also always excluded regardless of `include` patterns.

---

## Indexing

### First-time index

```bash
python3 glean.py --index
```

Discovers all matching files, chunks them, generates embeddings via Ollama, and stores everything in ChromaDB. Also builds the BM25 corpus file (`bm25_corpus.json`) alongside the vector database. Only new or changed files are processed on subsequent runs.

### Incremental update

Re-run the same command after editing files. Only files whose modification time has changed since the last run will be re-indexed.

```bash
python3 glean.py --index
```

### Full re-index

Deletes all existing chunks and rebuilds from scratch. Required after changing `embedding_model`, adding task prefixes, or switching the ChromaDB distance metric:

```bash
python3 glean.py --reindex
```

### Index a single collection

```bash
python3 glean.py --index -c myproject
```

### Check index status

```bash
python3 glean.py --status
```

Shows a table of collections with file counts, chunk counts, last-indexed timestamps, and total disk usage.

---

## Querying

### One-shot query

```bash
python3 glean.py "What does the authentication middleware do?"
```

### Query a specific collection

```bash
python3 glean.py -c myproject "Where is rate limiting implemented?"
```

### Show retrieved source chunks

The `--verbose` / `-v` flag prints the raw chunks that were retrieved before the answer is generated — useful for debugging retrieval quality:

```bash
python3 glean.py -v "How are database migrations handled?"
```

### Override the generation model

```bash
python3 glean.py -m qwen2.5:14b "Explain the worker pool design."
```

---

## Interactive Mode

Start an interactive session for back-and-forth exploration:

```bash
python3 glean.py -i
python3 glean.py -i -c myproject          # start scoped to a collection
python3 glean.py -i -m qwen2.5-coder:14b  # start with a specific model
```

### REPL commands

| Command | Alias | Description |
|---------|-------|-------------|
| `/collection <name>` | `/c <name>` | Switch to a specific collection |
| `/collection all` | `/c all` | Query all collections (default) |
| `/model <name>` | `/m <name>` | Switch generation model |
| `/verbose` | `/v` | Toggle verbose chunk display |
| `/status` | | Show index status table |
| `/clear` | | Clear the screen |
| `/quit` | `/q` | Exit (also Ctrl+D) |

Any input that doesn't start with `/` is treated as a question.

---

## CLI Reference

```
usage: glean [-h] [-c COLLECTION] [--index] [--reindex] [--status]
             [-i] [-m MODEL] [-v] [--config CONFIG]
             [question]

positional arguments:
  question              Question to ask

options:
  -c, --collection      Collection to restrict queries or indexing to
  --index               Incremental index of configured collections
  --reindex             Full re-index (delete & rebuild) of configured collections
  --status              Show index status
  -i, --interactive     Interactive question-answering mode
  -m, --model           Override generation model for this run
  -v, --verbose         Verbose output (show retrieved chunks)
  --config              Path to glean.yaml configuration file
```

**Exit codes:**

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | Query or embedding failure |
| `2` | Indexing failure |
| `3` | Configuration or usage error |

---

## How It Works

### Indexing pipeline

1. **Discovery** — walks configured `paths`, applying `include`/`exclude` filters and skipping files over 500 KB or matching secret-file patterns
2. **Chunking** — splits each file into semantically meaningful chunks using a language-specific strategy (see [Chunking strategies](#chunking-strategies))
3. **Embedding** — sends chunks to Ollama in batches of 32 using the `search_document:` task prefix required by nomic-embed-text for asymmetric retrieval
4. **Vector storage** — upserts chunks, embeddings, and metadata into a ChromaDB collection configured with cosine distance (`hnsw:space: cosine`)
5. **BM25 corpus** — saves all chunk texts to `bm25_corpus.json` (key: chunk ID, value: `{text, collection}`) so keyword search is available at query time without touching ChromaDB
6. **State tracking** — saves file mtimes and chunk IDs to `index_meta.json` so the next run only processes changed files

### Query pipeline

1. **Embed the question** using the `search_query:` task prefix (asymmetric pair to `search_document:` used at index time)
2. **Dense retrieval** — query each ChromaDB collection for the top `top_k` most similar chunks by cosine distance; apply `max_distance` cutoff
3. **Sparse retrieval (BM25)** — tokenize the question, remove stopwords, then run two BM25 passes over the corpus:
   - *Full-query pass*: scores all chunks against the complete set of query tokens (up to `bm25_top_k` results)
   - *Rare-term pass*: identifies the most specific token(s) in the query (document frequency < 1% of corpus), then re-runs BM25 with those rare tokens doubled in weight combined with the remaining tokens; this ensures that a specific identifier like a hostname or variable name pulls in its exact matches even when common words like "public" or "function" dominate the full-query scores
4. **RRF merge** — combines the three ranked lists (dense, BM25, rare-term BM25 at 2× weight) using Reciprocal Rank Fusion: `score = Σ 1/(k + rank)` where k=`rrf_k`; chunks appearing in multiple lists accumulate score
5. **Metadata fetch** — BM25-surfaced chunks not already in the dense result set are fetched from ChromaDB in a single batch `get()` call per collection to retrieve their metadata
6. **Context assembly** — concatenates retrieved chunks (with file header annotations) in RRF score order, truncated to `max_context_chars`
7. **Generate** — sends the assembled context and question to the LLM with a system prompt instructing it to answer only from the provided context and cite sources
8. **Display** — renders the answer as Markdown (CLI) or streams tokens via SSE (web GUI), followed by source file locations with line numbers

### Why hybrid search?

Dense (vector) retrieval works well for conceptual questions where the answer uses different vocabulary than the query. For example:

> *"How does the platform handle win submissions?"*

will find relevant code even if the source uses terms like `submit_win()` or `WinRecord` rather than the word "submission".

BM25 keyword retrieval works well for exact-match lookups where a specific identifier must appear in the result:

> *"What is the public IP address of rafael?"*

The answer is in a YAML file containing `public_ip_address: "198.55.58.201"`. Dense search fails here because the embedding of a list of 50 host-to-IP mappings looks similar to many config files, and the query embedding maps "public IP address" to generic networking documentation. BM25 finds the exact chunk containing "rafael" regardless of semantic similarity.

The rare-term mechanism specifically addresses queries containing a proper noun or unique identifier mixed with common words. Given the query above, "rafael" has document frequency 0.27% (232/85,000 chunks) and is classified as rare; "public", "ip", and "address" are common. The rare-term BM25 pass runs `["rafael", "rafael", "public", "ip", "address"]` — doubling "rafael" to ensure chunks containing it score higher than chunks containing only the common terms.

### Tokenizer

BM25 uses a word-boundary tokenizer (`re.findall(r"\w+", text.lower())`) with English stopword removal. This handles structured text well:

| Input | Tokens |
|-------|--------|
| `ip: 198.55.58.201` | `["ip", "198", "55", "58", "201"]` |
| `rafael.pluio.net` | `["rafael", "pluio", "net"]` |
| `public function getIp()` | `["public", "function", "getip"]` |
| `What is rafael public IP?` | `["rafael", "public", "ip"]` (stopwords removed) |

Simple `.split()` would leave `ip:` and `198.55.58.201` as atomic tokens that never match query terms.

### BM25 corpus persistence

The corpus is stored as a flat JSON file (`state_dir/bm25_corpus.json`) mapping chunk IDs to `{text, collection}`. At 85,000 chunks it is typically 60–70 MB. It is kept in sync with ChromaDB:

- **Index run**: new/updated/deleted chunks are mirrored into the corpus
- **Auto-bootstrap**: if the corpus has fewer than 50% of the expected chunks (e.g., after a fresh clone or if the file was deleted), glean fetches all existing documents from ChromaDB in batches and rebuilds the corpus before the index run begins
- **Process lifetime**: the `BM25Okapi` index object is built lazily on the first query and cached in memory for the lifetime of the process; `server.py` benefits most from this since the server stays running across many requests

### Chunking strategies

| Language | Split boundaries |
|----------|-----------------|
| Python | `def` and `class` definitions |
| TypeScript / JavaScript | `export function/class/const`, `describe()`, `test()` |
| PHP | `class` and `function` keywords |
| Markdown | H1 (`#`) and H2 (`##`) headings; H3 (`###`) used for oversized sections |
| YAML | Top-level keys |
| JSON | Top-level object keys (parsed, then re-serialized per key) |
| Shell | Function definitions and comment headers |
| INI / conf | `[section]` blocks and comment headers |
| Other | Paragraph-based splitting |

Small adjacent chunks are merged up to a minimum character threshold to avoid embedding very short fragments. Oversized chunks are recursively split on double newlines.

### Embedding model notes

glean uses the **asymmetric retrieval** mode of nomic-embed-text:

- Documents are embedded with the `search_document:` prefix
- Queries are embedded with the `search_query:` prefix

Without these prefixes, the model uses symmetric similarity, which produces significantly worse retrieval for question-answer pairs where the question and the document use different vocabulary. **If you switch embedding models**, run `--reindex` so all stored embeddings are regenerated with the same prefix convention.

ChromaDB collections are created with `hnsw:space: cosine` so distances are cosine distances in `[0, 1]` (lower = more similar). The `max_distance` cutoff drops chunks that are too semantically distant from the query, preventing low-quality noise from filling the context window.

---

## Tips

**Use collections to improve focus:**
Scoping a query with `-c myproject` reduces noise from unrelated codebases and gives the LLM a cleaner context. Define separate collections for code, documentation, infrastructure, and project management artifacts rather than one giant collection.

**Diagnose poor answers with `--verbose`:**
`python3 glean.py -v "your question"` prints every retrieved chunk before generation. If the relevant file is missing entirely, check that it is included by your `include`/`exclude` patterns. If it appears but ranks low, the query phrasing may need adjustment.

**Lookup queries vs. conceptual queries:**
For lookups (IP addresses, configuration values, hostnames), include the specific identifier in the query: *"What is the redis port in the staging config?"* is more reliably retrieved than *"What port does redis use?"*. The hybrid search rare-term mechanism handles this automatically, but a specific query gives it more signal to work with.

**Model selection:**
- `nomic-embed-text` is the recommended embedding model; it has an 8192-token context and supports asymmetric retrieval prefixes
- `mxbai-embed-large` is a higher-quality alternative at the cost of more RAM and slower indexing
- For generation, code-focused models (`qwen2.5-coder`, `codellama`, `deepseek-coder`) tend to outperform general models on programming questions; general models (`qwen2.5:14b`, `llama3.1`) work better for narrative documentation
- Use `-m` to test different generation models without changing your config

**Tune `max_context_chars`:**
Small local models (7B parameters) can be confused by very long contexts. The code default is 24,000 chars; lowering to 8,000–15,000 often improves focus for smaller models. If you run a 14B+ model with a larger context window, 20,000–30,000 often improves recall for questions that require synthesizing many sources.

**After major config changes, run `--reindex`:**
Changing `include`/`exclude` patterns or `paths` does not automatically remove stale chunks. Adding a new collection only requires `--index`. Removing a path or changing the embedding model requires `--reindex` to clean up orphaned embeddings.

**Secrets are never indexed:**
Common secret file patterns are hard-coded as always-excluded regardless of your `include` configuration: `.env`, `*.env`, `*.pem`, `*.key`, `*.p12`, `*.pfx`, `id_rsa`, `id_ecdsa`, `id_ed25519`, `credentials.json`, `secrets.yaml`, `secrets.yml`.

---

## Architecture

```
glean.py                         server.py
├── load_config()                ├── FastAPI app
├── discover_files()             ├── /ask  (SSE stream)
├── chunk_file_contents()        │   └── do_retrieve()
│   ├── chunk_python()           │       └── retrieve_chunks()
│   ├── chunk_markdown()         ├── /index
│   ├── chunk_yaml()             ├── /reindex
│   └── ...                      └── /status
├── embed_texts()     ──────────────────────────────────────
│   └── ollama.embed()           State (state_dir/)
├── get_collection()             ├── chroma/          (ChromaDB)
│   └── chromadb.PersistentClient│   └── <uuid>/      (one per collection)
├── index_collection()           │       ├── chroma.sqlite3
│   └── BM25Corpus.add()         │       └── ...
├── index_all()                  ├── index_meta.json  (mtime + chunk IDs)
│   ├── _bootstrap_corpus()      └── bm25_corpus.json (chunk text for BM25)
│   └── BM25Corpus.save()
├── retrieve_chunks()
│   ├── embed_single()
│   ├── ChromaDB.query()         (dense)
│   ├── BM25Corpus.query()       (full BM25)
│   ├── BM25Corpus.query_rare_terms()  (rare-term BM25 × 2 weight)
│   └── rrf_merge()
├── build_context_snippets()
└── ask_question()
    └── ollama.chat()
```

---

## Troubleshooting

**"No relevant documents found in the index"**
- Run `--status` to confirm files are indexed
- Run `--index` if the corpus may be stale (check that `bm25_corpus.json` exists in `state_dir`)
- Try `--verbose` to see what (if anything) is retrieved at threshold
- Relax `max_distance` in your config (`null` disables the cutoff entirely)

**Embeddings fail with token length errors**
nomic-embed-text caps at 8192 tokens. Files with dense special characters (JSON, minified JS, shell scripts with many glob patterns) can approach 1 token per character. glean caps each chunk at 3,000 characters before embedding to stay safely under the limit. If you still see errors, check for extremely dense files and add them to `exclude`.

**ChromaDB HNSW index corruption**
Symptoms: queries return no results or obviously wrong results despite confirmed indexed content. Fix:
```bash
rm -rf ~/.local/share/glean/chroma
python3 glean.py --reindex
```
The `bm25_corpus.json` and `index_meta.json` files do not need to be deleted; they will be auto-bootstrapped.

**BM25 corpus is missing or stale**
glean auto-detects when `bm25_corpus.json` has fewer than 50% of the chunks known to `index_meta.json` and rebuilds it by fetching all documents from ChromaDB. This happens automatically on the next `--index` run. To force it manually, delete `bm25_corpus.json` and run `--index`.

**Server returns stale BM25 results after re-indexing**
The `server.py` process reloads `_corpus` automatically after each `/index` or `/reindex` call. If you ran `glean.py --reindex` from the command line while the server was running, restart the server to pick up the new corpus.

**Answers ignore the relevant context**
Small models (7B) can lose track of information buried deep in a long context. Lower `max_context_chars` (e.g., `8000`) to force the model to focus on the most relevant chunks, or upgrade to a larger generation model. The `--verbose` flag shows exactly what context was sent.
