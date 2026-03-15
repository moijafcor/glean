# glean

Local RAG (Retrieval-Augmented Generation) for codebases and project documents. Ask questions in plain English and get answers grounded in your actual source files — all running on your machine with no data sent to external services.

Built on [Ollama](https://ollama.com) for embeddings and generation, and [ChromaDB](https://www.trychroma.com) for vector storage.

---

## Features

- **Incremental indexing** — only re-indexes files that have changed since the last run
- **Language-aware chunking** — Python, TypeScript/JavaScript, PHP, Markdown, YAML, JSON, shell scripts, and INI-style configs each get a tailored chunking strategy
- **Multi-collection support** — organize different projects or codebases into named collections and query them together or individually
- **Interactive REPL** — conversational mode with live collection and model switching
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
pip install chromadb ollama pyyaml rich
```

**2. Pull the required Ollama models:**

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5-coder:7b   # or whichever generation model you prefer
```

**3. Create a `glean.yaml` configuration file** in the directory where you'll run glean (see [Configuration](#configuration) below).

---

## Quick Start

```bash
# Index your configured collections
python3 glean.py --index

# Ask a question
python3 glean.py "How does authentication work in this codebase?"

# Start the interactive REPL
python3 glean.py -i
```

---

## Configuration

glean is configured via a YAML file. By default it looks for `glean.yaml` in the current directory. Use `--config` to specify a different path.

```yaml
# Models
embedding_model: nomic-embed-text
generation_model: qwen2.5-coder:7b

# Ollama server URL
ollama_url: http://localhost:11434

# Where to store the vector database and index state
state_dir: ~/.local/share/glean

# Maximum characters of retrieved context to send to the LLM
# (~24000 chars ≈ 6000 tokens at average English density)
max_context_chars: 24000

# Number of chunks to retrieve per collection per query
top_k: 8

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
```

### Configuration reference

| Key | Default | Description |
|-----|---------|-------------|
| `embedding_model` | `nomic-embed-text` | Ollama model used to generate embeddings |
| `generation_model` | `qwen2.5:14b` | Ollama model used to generate answers |
| `ollama_url` | `http://localhost:11434` | URL of the Ollama server |
| `state_dir` | `~/.local/share/glean` | Directory for ChromaDB and index state |
| `max_context_chars` | `24000` | Character limit for context sent to the LLM |
| `top_k` | `12` | Chunks retrieved per collection per query |
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

Discovers all matching files, chunks them, generates embeddings via Ollama, and stores everything in ChromaDB. Only new or changed files are processed on subsequent runs.

### Incremental update

Re-run the same command after editing files. Only files whose modification time has changed since the last run will be re-indexed.

```bash
python3 glean.py --index
```

### Full re-index

Deletes all existing chunks and rebuilds from scratch:

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
2. **Chunking** — splits each file into semantically meaningful chunks using a language-specific strategy (see below)
3. **Embedding** — sends chunks to Ollama in batches of 32 to generate vector embeddings
4. **Storage** — upserts chunks, embeddings, and metadata into a persistent ChromaDB collection
5. **State tracking** — saves file mtimes and chunk IDs to `index_meta.json` so the next run only processes changed files

### Query pipeline

1. **Embed the question** using the same embedding model
2. **Retrieve** the top-K most similar chunks from each collection via cosine similarity
3. **Truncate** the combined context to `max_context_chars` to stay within the LLM's effective window
4. **Generate** an answer using the system prompt plus the retrieved context
5. **Display** the answer (rendered as Markdown) and the source file locations

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

---

## Tips

**Improve retrieval quality:**
- Keep `top_k` between 6–12. Higher values add noise; lower values may miss relevant context.
- If answers are vague or wrong, try `--verbose` to inspect what was actually retrieved.
- For large repos, use collections to scope queries to the relevant subsystem.

**Model selection:**
- `nomic-embed-text` is fast and accurate for code embeddings. `mxbai-embed-large` is a higher-quality alternative at the cost of more memory.
- For generation, code-specific models like `qwen2.5-coder` tend to outperform general models on programming questions.
- Use `-m` to test different generation models without changing your config.

**Multiple projects:**
- Use a shared `state_dir` and separate collections per project so all indexes share one ChromaDB instance.
- Use a per-project `glean.yaml` with `--config` if you want fully isolated state.

**Re-indexing after config changes:**
- Changing `include`/`exclude` patterns or `paths` does not automatically remove stale chunks. Run `--reindex` after structural config changes.
