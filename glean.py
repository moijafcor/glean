#!/usr/bin/env python3
"""
glean.py - Local RAG for code & project documents using Ollama + ChromaDB.

Single-file implementation with:
- YAML-based inventory configuration
- Incremental indexing with mtime tracking
- Language-aware chunking
- Local embeddings & generation via Ollama
- Persistent embeddings in embedded ChromaDB
- CLI + interactive REPL
"""

from __future__ import annotations

import argparse
import dataclasses
import fnmatch
import json
import os
import re
import sys
import textwrap
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import chromadb
import ollama
import yaml
from chromadb.config import Settings
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.table import Table


console = Console()


# -----------------------------
# Dataclasses & Types
# -----------------------------


@dataclass
class ChunkMetadata:
    file_path: str
    file_name: str
    collection: str
    language: str
    start_line: int
    end_line: int
    heading: Optional[str]
    mtime: float
    char_count: int

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class FileIndexEntry:
    mtime: float
    chunk_ids: List[str]


@dataclass
class IndexState:
    files: Dict[str, FileIndexEntry] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "IndexState":
        if not path.exists():
            return cls()
        try:
            with path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            return cls()

        files: Dict[str, FileIndexEntry] = {}
        for file_path, info in raw.get("files", {}).items():
            files[file_path] = FileIndexEntry(
                mtime=info.get("mtime", 0.0),
                chunk_ids=list(info.get("chunk_ids", [])),
            )
        return cls(files=files)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = {
            "files": {
                fp: {"mtime": entry.mtime, "chunk_ids": entry.chunk_ids}
                for fp, entry in self.files.items()
            }
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2, sort_keys=True)


@dataclass
class QueryResult:
    answer: str
    sources: List[ChunkMetadata]
    raw_chunks: List[str]


# -----------------------------
# Config Loading
# -----------------------------


DEFAULT_CONFIG_PATH = "glean.yaml"


def expand_path(p: str) -> str:
    return str(Path(os.path.expanduser(p)).resolve())


def load_config(path: Optional[str]) -> Dict[str, Any]:
    cfg_path = Path(path or DEFAULT_CONFIG_PATH)
    if not cfg_path.exists():
        console.print(f"[red]Config file not found:[/red] {cfg_path}")
        sys.exit(3)
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as exc:
        console.print(f"[red]Failed to parse config:[/red] {exc}")
        sys.exit(3)

    if "collections" not in cfg or not isinstance(cfg["collections"], dict):
        console.print("[red]Config error:[/red] `collections` mapping is required.")
        sys.exit(3)

    cfg.setdefault("embedding_model", "nomic-embed-text")
    cfg.setdefault("generation_model", "qwen2.5:14b")
    cfg.setdefault("ollama_url", "http://localhost:11434")
    cfg.setdefault("state_dir", os.path.join("~", ".local", "share", "glean"))
    cfg.setdefault("max_context_tokens", 6000)
    cfg.setdefault("top_k", 12)

    # Normalize paths in collections
    for name, coll in cfg["collections"].items():
        paths = coll.get("paths") or []
        coll["paths"] = [expand_path(p) for p in paths]
        coll.setdefault("include", ["*.py", "*.md", "*.yaml", "*.yml"])
        coll.setdefault("exclude", [])
    return cfg


# -----------------------------
# File Discovery
# -----------------------------


def matches_any(path: str, patterns: Sequence[str]) -> bool:
    for pat in patterns:
        if pat.endswith("/"):
            # directory-style exclusion: if any parent contains this segment
            seg = pat.rstrip("/").split("/")[-1]
            if f"/{seg}/" in path or path.endswith(f"/{seg}"):
                return True
        if fnmatch.fnmatch(os.path.basename(path), pat) or fnmatch.fnmatch(path, pat):
            return True
    return False


def discover_files(
    collection_name: str, cfg: Dict[str, Any]
) -> List[Tuple[str, str]]:
    """Return list of (abs_path, rel_path) for files to index."""
    coll = cfg["collections"][collection_name]
    include = coll.get("include", [])
    exclude = coll.get("exclude", [])
    roots = coll["paths"]

    results: List[Tuple[str, str]] = []
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root_path):
            # Prune excluded directories
            pruned = []
            for d in list(dirnames):
                full = os.path.join(dirpath, d)
                rel = os.path.relpath(full, root_path)
                if matches_any(rel, exclude):
                    continue
                pruned.append(d)
            dirnames[:] = pruned

            for fname in filenames:
                abs_path = os.path.join(dirpath, fname)
                rel = os.path.relpath(abs_path, root_path)
                if os.path.getsize(abs_path) > 500 * 1024:
                    # skip large files
                    continue
                if matches_any(rel, exclude):
                    continue
                if not any(fnmatch.fnmatch(fname, pat) for pat in include):
                    continue
                # Avoid .env and similar secrets
                if fname == ".env" or fname.endswith(".env"):
                    continue
                results.append((abs_path, os.path.join(Path(root).name, rel)))
    return results


# -----------------------------
# Chunking Engine
# -----------------------------


LANG_MARKDOWN = {"md"}
LANG_PYTHON = {"py"}
LANG_PHP = {"php"}
LANG_TSJS = {"ts", "tsx", "js", "jsx"}
LANG_YAML = {"yaml", "yml"}
LANG_JSON = {"json"}
LANG_SHELL = {"sh", "bash", "zsh"}
LANG_CONFIG = {"conf", "ini", "cfg"}


def detect_language(path: str) -> str:
    ext = Path(path).suffix.lower().lstrip(".")
    if ext in LANG_MARKDOWN:
        return "markdown"
    if ext in LANG_PYTHON:
        return "python"
    if ext in LANG_PHP:
        return "php"
    if ext in LANG_TSJS:
        return "ts_js"
    if ext in LANG_YAML:
        return "yaml"
    if ext in LANG_JSON:
        return "json"
    if ext in LANG_SHELL:
        return "shell"
    if ext in LANG_CONFIG:
        return "config"
    return "text"


def _merge_small_chunks(
    chunks: List[Tuple[str, int, int, Optional[str]]],
    min_chars: int,
) -> List[Tuple[str, int, int, Optional[str]]]:
    if not chunks:
        return []
    merged: List[Tuple[str, int, int, Optional[str]]] = []
    cur_text, cur_start, cur_end, cur_head = chunks[0]
    for text, start, end, head in chunks[1:]:
        if len(cur_text) < min_chars:
            cur_text = cur_text.rstrip() + "\n\n" + text.lstrip()
            cur_end = end
            # preserve first heading
        else:
            merged.append((cur_text, cur_start, cur_end, cur_head))
            cur_text, cur_start, cur_end, cur_head = text, start, end, head
    merged.append((cur_text, cur_start, cur_end, cur_head))
    return merged


def _split_long(text: str, start_line: int, max_chars: int) -> List[Tuple[str, int, int]]:
    """Fallback: split long text on double newlines approx max_chars."""
    paragraphs = text.split("\n\n")
    chunks: List[Tuple[str, int, int]] = []
    cur_lines: List[str] = []
    cur_len = 0
    cur_start = start_line
    line_no = start_line
    for para in paragraphs:
        add = para + "\n\n"
        if cur_len + len(add) > max_chars and cur_lines:
            chunk_text = "\n".join(cur_lines).rstrip()
            end_line = line_no - 1
            chunks.append((chunk_text, cur_start, end_line))
            cur_lines = [para]
            cur_len = len(para)
            cur_start = line_no
        else:
            cur_lines.append(para)
            cur_len += len(add)
        line_no += para.count("\n") + 2
    if cur_lines:
        chunk_text = "\n".join(cur_lines).rstrip()
        end_line = line_no - 1
        chunks.append((chunk_text, cur_start, end_line))
    return chunks


def chunk_markdown(text: str, base_heading: Optional[str] = None) -> List[Tuple[str, int, int, Optional[str]]]:
    lines = text.splitlines()
    chunks: List[Tuple[str, int, int, Optional[str]]] = []
    current_heading = base_heading
    current_lines: List[str] = []
    start_line = 1
    heading_line = 1

    def flush(end_line: int, heading: Optional[str]) -> None:
        if not current_lines:
            return
        section_text = "\n".join(current_lines).rstrip()
        chunks.append((section_text, start_line, end_line, heading))

    for idx, line in enumerate(lines, start=1):
        if line.startswith("## "):  # H2
            flush(idx - 1, current_heading)
            current_heading = line.strip()
            current_lines = [line]
            start_line = idx
            heading_line = idx
        else:
            if not current_lines:
                start_line = idx
                heading_line = idx
            current_lines.append(line)
    flush(len(lines), current_heading)

    processed: List[Tuple[str, int, int, Optional[str]]] = []
    for text_chunk, s, e, heading in chunks:
        if len(text_chunk) <= 1500:
            processed.append((text_chunk, s, e, heading))
        else:
            # Split on H3 within this section
            sublines = text_chunk.splitlines()
            sub_chunks: List[Tuple[str, int, int]] = []
            cur: List[str] = []
            sub_start = s
            for off, line in enumerate(sublines):
                abs_line = s + off
                if line.startswith("### ") and cur:
                    sub_chunks.append(("\n".join(cur).rstrip(), sub_start, abs_line - 1))
                    cur = [line]
                    sub_start = abs_line
                else:
                    cur.append(line)
            if cur:
                sub_chunks.append(("\n".join(cur).rstrip(), sub_start, s + len(sublines) - 1))

            for sub_text, sub_s, sub_e in sub_chunks:
                if len(sub_text) <= 1500:
                    processed.append((sub_text, sub_s, sub_e, heading))
                else:
                    for t, ss, ee in _split_long(sub_text, sub_s, 1500):
                        processed.append((t, ss, ee, heading))

    return _merge_small_chunks(processed, min_chars=100)


PY_DEF_RE = re.compile(r"^(def|class)\s+\w+", re.MULTILINE)


def chunk_python(text: str) -> List[Tuple[str, int, int, Optional[str]]]:
    lines = text.splitlines()
    chunks: List[Tuple[str, int, int, Optional[str]]] = []
    current: List[str] = []
    start_line = 1
    file_docstring: Optional[str] = None

    # detect file-level docstring (very simple heuristic)
    if lines and lines[0].startswith(('"""', "'''")):
        doc_lines = [lines[0]]
        for i in range(1, len(lines)):
            doc_lines.append(lines[i])
            if lines[i].startswith(('"""', "'''")) and i != 0:
                file_docstring = "\n".join(doc_lines)
                break

    def flush(end_line: int) -> None:
        if not current:
            return
        body = "\n".join(current).rstrip()
        if file_docstring:
            text_chunk = file_docstring + "\n\n" + body
        else:
            text_chunk = body
        chunks.append((text_chunk, start_line, end_line, None))

    for idx, line in enumerate(lines, start=1):
        if re.match(r"^(def|class)\s+\w+", line) and current:
            flush(idx - 1)
            current = [line]
            start_line = idx
        else:
            if not current:
                start_line = idx
            current.append(line)
    flush(len(lines))

    # Split very long chunks by method if class
    processed: List[Tuple[str, int, int, Optional[str]]] = []
    for text_chunk, s, e, heading in chunks:
        if len(text_chunk) <= 3000:
            processed.append((text_chunk, s, e, heading))
        else:
            for t, ss, ee in _split_long(text_chunk, s, 3000):
                processed.append((t, ss, ee, heading))
    return _merge_small_chunks(processed, min_chars=50)


def chunk_php(text: str) -> List[Tuple[str, int, int, Optional[str]]]:
    lines = text.splitlines()
    chunks: List[Tuple[str, int, int, Optional[str]]] = []
    current: List[str] = []
    start_line = 1

    def flush(end_line: int) -> None:
        if not current:
            return
        chunks.append(("\n".join(current).rstrip(), start_line, end_line, None))

    for idx, line in enumerate(lines, start=1):
        if re.search(r"\b(class|function)\b", line) and current:
            flush(idx - 1)
            current = [line]
            start_line = idx
        else:
            if not current:
                start_line = idx
            current.append(line)
    flush(len(lines))

    processed: List[Tuple[str, int, int, Optional[str]]] = []
    for text_chunk, s, e, heading in chunks:
        if len(text_chunk) <= 3000:
            processed.append((text_chunk, s, e, heading))
        else:
            for t, ss, ee in _split_long(text_chunk, s, 3000):
                processed.append((t, ss, ee, heading))
    return _merge_small_chunks(processed, min_chars=50)


def chunk_ts_js(text: str) -> List[Tuple[str, int, int, Optional[str]]]:
    lines = text.splitlines()
    chunks: List[Tuple[str, int, int, Optional[str]]] = []
    current: List[str] = []
    start_line = 1

    def flush(end_line: int) -> None:
        if not current:
            return
        chunks.append(("\n".join(current).rstrip(), start_line, end_line, None))

    pattern = re.compile(
        r"^\s*(export\s+(function|class|const|let|var)|describe\s*\(|test\s*\()", re.MULTILINE
    )
    for idx, line in enumerate(lines, start=1):
        if pattern.match(line) and current:
            flush(idx - 1)
            current = [line]
            start_line = idx
        else:
            if not current:
                start_line = idx
            current.append(line)
    flush(len(lines))

    processed: List[Tuple[str, int, int, Optional[str]]] = []
    for text_chunk, s, e, heading in chunks:
        if len(text_chunk) <= 3000:
            processed.append((text_chunk, s, e, heading))
        else:
            for t, ss, ee in _split_long(text_chunk, s, 3000):
                processed.append((t, ss, ee, heading))
    return _merge_small_chunks(processed, min_chars=50)


def chunk_yaml(text: str) -> List[Tuple[str, int, int, Optional[str]]]:
    lines = text.splitlines()
    chunks: List[Tuple[str, int, int, Optional[str]]] = []
    current: List[str] = []
    start_line = 1
    current_key: Optional[str] = None

    def flush(end_line: int) -> None:
        if not current:
            return
        heading = current_key
        chunks.append(("\n".join(current).rstrip(), start_line, end_line, heading))

    top_key_re = re.compile(r"^[a-zA-Z0-9_\-]+:")
    for idx, line in enumerate(lines, start=1):
        if top_key_re.match(line) and current:
            flush(idx - 1)
            current = [line]
            start_line = idx
            current_key = line.strip()
        else:
            if not current:
                start_line = idx
                current_key = line.strip() if top_key_re.match(line) else None
            current.append(line)
    flush(len(lines))

    processed: List[Tuple[str, int, int, Optional[str]]] = []
    for text_chunk, s, e, heading in chunks:
        if len(text_chunk) <= 2000:
            processed.append((text_chunk, s, e, heading))
        else:
            for t, ss, ee in _split_long(text_chunk, s, 2000):
                processed.append((t, ss, ee, heading))
    return _merge_small_chunks(processed, min_chars=50)


def chunk_json(text: str) -> List[Tuple[str, int, int, Optional[str]]]:
    try:
        data = json.loads(text)
    except Exception:
        # fallback to generic
        return _merge_small_chunks(_split_long(text, 1, 2000), min_chars=50)  # type: ignore[arg-type]

    chunks: List[Tuple[str, int, int, Optional[str]]] = []
    if isinstance(data, dict):
        for key, value in data.items():
            pretty = json.dumps({key: value}, indent=2)
            lines = pretty.splitlines()
            chunks.append((pretty, 1, len(lines), str(key)))
    else:
        pretty = json.dumps(data, indent=2)
        lines = pretty.splitlines()
        chunks.append((pretty, 1, len(lines), None))
    return _merge_small_chunks(chunks, min_chars=50)


def chunk_shell(text: str) -> List[Tuple[str, int, int, Optional[str]]]:
    lines = text.splitlines()
    chunks: List[Tuple[str, int, int, Optional[str]]] = []
    current: List[str] = []
    start_line = 1
    current_head: Optional[str] = None

    def flush(end_line: int) -> None:
        if not current:
            return
        chunks.append(("\n".join(current).rstrip(), start_line, end_line, current_head))

    func_re = re.compile(r"^\s*(\w+)\s*\(\)\s*\{|^function\s+\w+")
    header_re = re.compile(r"^#[-\s]+|^##\s+")

    for idx, line in enumerate(lines, start=1):
        if (func_re.match(line) or header_re.match(line)) and current:
            flush(idx - 1)
            current = [line]
            start_line = idx
            current_head = line.strip()
        else:
            if not current:
                start_line = idx
                current_head = line.strip() if header_re.match(line) else None
            current.append(line)
    flush(len(lines))

    processed: List[Tuple[str, int, int, Optional[str]]] = []
    for text_chunk, s, e, heading in chunks:
        if len(text_chunk) <= 2000:
            processed.append((text_chunk, s, e, heading))
        else:
            for t, ss, ee in _split_long(text_chunk, s, 2000):
                processed.append((t, ss, ee, heading))
    return _merge_small_chunks(processed, min_chars=50)


def chunk_config(text: str) -> List[Tuple[str, int, int, Optional[str]]]:
    lines = text.splitlines()
    chunks: List[Tuple[str, int, int, Optional[str]]] = []
    current: List[str] = []
    start_line = 1
    current_head: Optional[str] = None

    def flush(end_line: int) -> None:
        if not current:
            return
        chunks.append(("\n".join(current).rstrip(), start_line, end_line, current_head))

    section_re = re.compile(r"^\[.+\]")
    header_re = re.compile(r"^#[-\s]+|^##\s+")

    for idx, line in enumerate(lines, start=1):
        if (section_re.match(line) or header_re.match(line)) and current:
            flush(idx - 1)
            current = [line]
            start_line = idx
            current_head = line.strip()
        else:
            if not current:
                start_line = idx
                current_head = line.strip() if (section_re.match(line) or header_re.match(line)) else None
            current.append(line)
    flush(len(lines))

    processed: List[Tuple[str, int, int, Optional[str]]] = []
    for text_chunk, s, e, heading in chunks:
        if len(text_chunk) <= 1000:
            processed.append((text_chunk, s, e, heading))
        else:
            for t, ss, ee in _split_long(text_chunk, s, 1000):
                processed.append((t, ss, ee, heading))
    return _merge_small_chunks(processed, min_chars=30)


def chunk_generic(text: str) -> List[Tuple[str, int, int, Optional[str]]]:
    chunks = _split_long(text, 1, 1000)
    return _merge_small_chunks(chunks, min_chars=50)  # type: ignore[arg-type]


def chunk_file_contents(path: str, text: str) -> List[Tuple[str, int, int, Optional[str]]]:
    lang = detect_language(path)
    if lang == "markdown":
        return chunk_markdown(text)
    if lang == "python":
        return chunk_python(text)
    if lang == "php":
        return chunk_php(text)
    if lang == "ts_js":
        return chunk_ts_js(text)
    if lang == "yaml":
        return chunk_yaml(text)
    if lang == "json":
        return chunk_json(text)
    if lang == "shell":
        return chunk_shell(text)
    if lang == "config":
        return chunk_config(text)
    return chunk_generic(text)


# -----------------------------
# Embedding via Ollama
# -----------------------------


def get_ollama_client(ollama_url: str) -> None:
    # `ollama` library uses OLLAMA_HOST env var
    os.environ.setdefault("OLLAMA_HOST", ollama_url)


def embed_texts(model: str, texts: Sequence[str], batch_size: int = 32) -> List[List[float]]:
    embeddings: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = ollama.embed(model=model, input=batch)
        batch_emb = resp.get("embeddings") or resp.get("embedding")  # type: ignore[assignment]
        if batch_emb is None:
            raise RuntimeError("Ollama embed response missing 'embeddings'")
        embeddings.extend(batch_emb)
    return embeddings


def embed_single(model: str, text: str) -> List[float]:
    resp = ollama.embed(model=model, input=[text])
    emb = resp.get("embeddings") or resp.get("embedding")
    if not emb:
        raise RuntimeError("Ollama embed response missing 'embeddings'")
    return emb[0]


# -----------------------------
# Vector Store (ChromaDB)
# -----------------------------


def get_chroma_client(state_dir: str):
    chroma_path = os.path.join(state_dir, "chroma")
    Path(chroma_path).mkdir(parents=True, exist_ok=True)
    client = chromadb.Client(
        Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=chroma_path,
        )
    )
    return client


def get_collection(client, name: str):
    return client.get_or_create_collection(name=name)


# -----------------------------
# Indexing
# -----------------------------


def deterministic_chunk_id(collection: str, rel_path: str, idx: int) -> str:
    return f"{collection}:{rel_path}:{idx}"


def index_collection(
    client,
    cfg: Dict[str, Any],
    index_state: IndexState,
    collection_name: str,
    reindex: bool = False,
) -> None:
    coll_cfg = cfg["collections"][collection_name]
    embedding_model = cfg["embedding_model"]
    state_dir = expand_path(cfg["state_dir"])
    get_ollama_client(cfg["ollama_url"])
    chroma_coll = get_collection(client, collection_name)

    files = discover_files(collection_name, cfg)
    if not files:
        console.print(f"[yellow]No files discovered for collection[/yellow] {collection_name}")
        return

    to_add: List[Tuple[str, str]] = []
    to_update: List[Tuple[str, str]] = []
    to_delete: List[str] = []

    # Determine changes
    existing_paths = set(index_state.files.keys())
    current_paths = set()

    for abs_path, rel_path in files:
        current_paths.add(abs_path)
        mtime = os.path.getmtime(abs_path)
        entry = index_state.files.get(abs_path)
        if reindex or entry is None:
            to_add.append((abs_path, rel_path))
        elif entry.mtime != mtime:
            to_update.append((abs_path, rel_path))

    # Deleted files
    for abs_path in list(existing_paths):
        if abs_path not in current_paths:
            entry = index_state.files.get(abs_path)
            if entry:
                to_delete.extend(entry.chunk_ids)
                del index_state.files[abs_path]

    if to_delete:
        chroma_coll.delete(ids=to_delete)

    def process_file(abs_path: str, rel_path: str) -> Tuple[List[str], List[str], List[Dict[str, Any]], List[ChunkMetadata]]:
        with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        chunks = chunk_file_contents(abs_path, text)
        mtime = os.path.getmtime(abs_path)
        lang = detect_language(abs_path)
        ids: List[str] = []
        docs: List[str] = []
        metas: List[Dict[str, Any]] = []
        meta_objs: List[ChunkMetadata] = []
        for idx, (chunk_text, start_line, end_line, heading) in enumerate(chunks):
            cid = deterministic_chunk_id(collection_name, rel_path, idx)
            ids.append(cid)
            docs.append(chunk_text)
            meta = ChunkMetadata(
                file_path=abs_path,
                file_name=os.path.basename(abs_path),
                collection=collection_name,
                language=lang,
                start_line=start_line,
                end_line=end_line,
                heading=heading,
                mtime=mtime,
                char_count=len(chunk_text),
            )
            metas.append(meta.to_dict())
            meta_objs.append(meta)
        return ids, docs, metas, meta_objs

    # Delete old chunks for updated files
    for abs_path, _ in to_update:
        entry = index_state.files.get(abs_path)
        if entry and entry.chunk_ids:
            chroma_coll.delete(ids=entry.chunk_ids)

    total_new = 0
    total_updated = 0

    # Process additions and updates
    for abs_path, rel_path in to_add + to_update:
        ids, docs, metas, _ = process_file(abs_path, rel_path)
        if not ids:
            continue
        embeddings = embed_texts(embedding_model, docs)
        chroma_coll.upsert(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)
        index_state.files[abs_path] = FileIndexEntry(
            mtime=os.path.getmtime(abs_path),
            chunk_ids=ids,
        )
        if (abs_path, rel_path) in to_add:
            total_new += len(ids)
        else:
            total_updated += len(ids)

    console.print(
        f"[green]Indexed collection[/green] {collection_name} "
        f"(new chunks: {total_new}, updated chunks: {total_updated}, deleted chunks: {len(to_delete)})"
    )


def index_all(
    cfg: Dict[str, Any],
    collection_filter: Optional[str],
    reindex: bool,
) -> int:
    state_dir = expand_path(cfg["state_dir"])
    index_meta_path = Path(state_dir) / "index_meta.json"
    index_state = IndexState.load(index_meta_path)
    client = get_chroma_client(state_dir)

    collections = [collection_filter] if collection_filter else list(cfg["collections"].keys())

    try:
        for name in collections:
            if name not in cfg["collections"]:
                console.print(f"[red]Unknown collection:[/red] {name}")
                return 3
            index_collection(client, cfg, index_state, name, reindex=reindex)
        index_state.save(index_meta_path)
    except Exception as exc:
        console.print(f"[red]Indexing failed:[/red] {exc}")
        return 2
    return 0


# -----------------------------
# Query Pipeline
# -----------------------------


SYSTEM_PROMPT = """You are a knowledgeable assistant that answers questions about a software development infrastructure and codebase.
Answer based ONLY on the provided context. If the context does not contain enough information to answer, say you don't know and do not guess.
Always cite the source file and line numbers for each claim you make."""


def build_context_snippets(
    results: Dict[str, Any],
    verbose: bool,
) -> Tuple[str, List[ChunkMetadata], List[str]]:
    metadatas: List[ChunkMetadata] = []
    chunk_texts: List[str] = []
    # results from Chroma: ids, documents, metadatas
    docs_list: List[str] = results.get("documents", [[]])[0]
    metas_list: List[Dict[str, Any]] = results.get("metadatas", [[]])[0]
    ids_list: List[str] = results.get("ids", [[]])[0]

    parts: List[str] = []
    for doc, meta, cid in zip(docs_list, metas_list, ids_list):
        cm = ChunkMetadata(
            file_path=meta.get("file_path", ""),
            file_name=meta.get("file_name", ""),
            collection=meta.get("collection", ""),
            language=meta.get("language", "text"),
            start_line=int(meta.get("start_line", 1)),
            end_line=int(meta.get("end_line", 1)),
            heading=meta.get("heading"),
            mtime=float(meta.get("mtime", 0.0)),
            char_count=int(meta.get("char_count", len(doc))),
        )
        metadatas.append(cm)
        chunk_texts.append(doc)
        header = f"[File: {cm.file_name} ({cm.collection}), lines {cm.start_line}-{cm.end_line}]"
        snippet = f"{header}\n{doc}"
        parts.append(snippet)

    context = "\n\n---\n\n".join(parts)
    if verbose:
        console.rule("Retrieved Chunks")
        for part in parts:
            console.print(Markdown(f"```text\n{part[:2000]}\n```"))
            console.print()
    return context, metadatas, chunk_texts


def ask_question(
    cfg: Dict[str, Any],
    question: str,
    collection_filter: Optional[str],
    generation_model_override: Optional[str],
    verbose: bool,
) -> int:
    state_dir = expand_path(cfg["state_dir"])
    get_ollama_client(cfg["ollama_url"])
    client = get_chroma_client(state_dir)
    top_k = int(cfg.get("top_k", 12))

    if collection_filter and collection_filter not in cfg["collections"]:
        console.print(f"[red]Unknown collection:[/red] {collection_filter}")
        return 3

    collections = [collection_filter] if collection_filter else list(cfg["collections"].keys())
    # For now, query each collection individually and merge results.
    all_docs: List[str] = []
    all_metas: List[Dict[str, Any]] = []
    all_ids: List[str] = []

    try:
        q_embedding = embed_single(cfg["embedding_model"], question)
    except Exception as exc:
        console.print(f"[red]Embedding failed:[/red] {exc}")
        return 1

    for name in collections:
        chroma_coll = get_collection(client, name)
        try:
            res = chroma_coll.query(query_embeddings=[q_embedding], n_results=top_k)
        except Exception:
            continue
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        ids = res.get("ids", [[]])[0]
        all_docs.extend(docs)
        all_metas.extend(metas)
        all_ids.extend(ids)

    if not all_docs:
        console.print("[yellow]No relevant documents found in the index.[/yellow]")
        return 1

    # Wrap combined results to match build_context_snippets input shape
    combined_results = {
        "documents": [all_docs],
        "metadatas": [all_metas],
        "ids": [all_ids],
    }
    context, metadatas, _ = build_context_snippets(combined_results, verbose=verbose)

    user_prompt = f"Context:\n---\n{context}\n\nQuestion: {question}"
    model = generation_model_override or cfg["generation_model"]

    try:
        resp = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": 0.1},
        )
    except Exception as exc:
        console.print(f"[red]Generation failed:[/red] {exc}")
        return 1

    content = resp["message"]["content"] if "message" in resp else str(resp)

    console.rule("[bold]Answer[/bold]")
    console.print(Markdown(content.strip()))
    console.print()
    console.print("[bold]Sources:[/bold]")
    seen = set()
    for meta in metadatas:
        key = (meta.file_name, meta.start_line, meta.end_line)
        if key in seen:
            continue
        seen.add(key)
        rel_path = meta.file_path
        console.print(
            f"  {rel_path}:{meta.start_line}-{meta.end_line} "
            f"({meta.collection}, {meta.language})"
        )
    return 0


# -----------------------------
# Status
# -----------------------------


def status_command(cfg: Dict[str, Any]) -> int:
    state_dir = expand_path(cfg["state_dir"])
    index_meta_path = Path(state_dir) / "index_meta.json"
    index_state = IndexState.load(index_meta_path)
    client = get_chroma_client(state_dir)

    table = Table(
        title="glean index status",
        box=box.SIMPLE_HEAVY,
        show_lines=False,
    )
    table.add_column("Collection")
    table.add_column("Files", justify="right")
    table.add_column("Chunks", justify="right")
    table.add_column("Last indexed")

    total_files = 0
    total_chunks = 0

    for name in cfg["collections"].keys():
        chroma_coll = get_collection(client, name)
        # Chroma does not expose per-collection stats directly; approximate via index_state
        files_in_coll = []
        chunks_in_coll = 0
        last_mtime = 0.0
        for fp, entry in index_state.files.items():
            # very rough heuristic: path under any of the collection roots
            for root in cfg["collections"][name]["paths"]:
                if fp.startswith(root):
                    files_in_coll.append(fp)
                    chunks_in_coll += len(entry.chunk_ids)
                    last_mtime = max(last_mtime, entry.mtime)
                    break
        total_files += len(files_in_coll)
        total_chunks += chunks_in_coll
        last_idx = (
            datetime.fromtimestamp(last_mtime).strftime("%Y-%m-%d %H:%M:%S")
            if last_mtime
            else "-"
        )
        table.add_row(name, str(len(files_in_coll)), str(chunks_in_coll), last_idx)

    table.add_row("-----------", "------", "------", "")
    table.add_row("Total", str(total_files), str(total_chunks), "")

    console.print(table)
    size_str = "-"
    try:
        if os.path.isdir(state_dir):
            total_bytes = 0
            for dirpath, _, filenames in os.walk(state_dir):
                for fn in filenames:
                    fp = os.path.join(dirpath, fn)
                    total_bytes += os.path.getsize(fp)
            size_mb = total_bytes / (1024 * 1024)
            size_str = f"{size_mb:.1f} MB"
    except Exception:
        pass

    console.print()
    console.print(
        f"Embedding model: {cfg['embedding_model']}\n"
        f"Generation model: {cfg['generation_model']}\n"
        f"State dir: {state_dir} ({size_str})"
    )
    return 0


# -----------------------------
# Interactive Mode
# -----------------------------


def interactive_mode(cfg: Dict[str, Any], collection: Optional[str], model: Optional[str], verbose: bool) -> int:
    current_collection = collection or "all"
    current_model = model or cfg["generation_model"]
    verbose_flag = verbose

    console.print("glean interactive mode (Ctrl+D to exit)")
    console.print(f"Collection: {current_collection} | Model: {current_model}")

    while True:
        try:
            line = Prompt.ask("> ")
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not line.strip():
            continue
        if line.startswith("/"):
            parts = line.strip().split(maxsplit=1)
            cmd = parts[0].lstrip("/")
            arg = parts[1].strip() if len(parts) > 1 else ""
            if cmd in {"collection", "c"}:
                if not arg or arg == "all":
                    current_collection = "all"
                else:
                    if arg not in cfg["collections"]:
                        console.print(f"[red]Unknown collection:[/red] {arg}")
                    else:
                        current_collection = arg
                console.print(f"Collection: {current_collection} | Model: {current_model}")
            elif cmd in {"model", "m"}:
                if not arg:
                    console.print("[yellow]Usage:[/yellow] /model MODEL_NAME")
                else:
                    current_model = arg
                    console.print(f"Collection: {current_collection} | Model: {current_model}")
            elif cmd in {"verbose", "v"}:
                verbose_flag = not verbose_flag
                console.print(f"Verbose: {verbose_flag}")
            elif cmd == "clear":
                console.clear()
                console.print("glean interactive mode (Ctrl+D to exit)")
                console.print(f"Collection: {current_collection} | Model: {current_model}")
            elif cmd == "status":
                status_command(cfg)
            elif cmd in {"quit", "q"}:
                break
            else:
                console.print(f"[yellow]Unknown command:[/yellow] {cmd}")
            continue

        coll_filter = None if current_collection == "all" else current_collection
        rc = ask_question(
            cfg,
            question=line,
            collection_filter=coll_filter,
            generation_model_override=current_model,
            verbose=verbose_flag,
        )
        if rc != 0:
            console.print(f"[red]Query failed with code {rc}[/red]")
    return 0


# -----------------------------
# CLI
# -----------------------------


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="glean",
        description="Local RAG for codebases and project documents using Ollama + ChromaDB.",
    )
    parser.add_argument("question", nargs="?", help="Question to ask")
    parser.add_argument(
        "-c",
        "--collection",
        dest="collection",
        help="Collection to restrict queries or indexing to",
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help="Incremental index of configured collections",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Full re-index (delete & rebuild) of configured collections",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show index status",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Interactive question-answering mode",
    )
    parser.add_argument(
        "-m",
        "--model",
        dest="model",
        help="Override generation model for this run",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output (show retrieved chunks)",
    )
    parser.add_argument(
        "--config",
        dest="config",
        help="Path to glean.yaml configuration file",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)

    if args.status:
        return status_command(cfg)

    if args.index or args.reindex:
        return index_all(cfg, collection_filter=args.collection, reindex=args.reindex)

    if args.interactive:
        return interactive_mode(cfg, collection=args.collection, model=args.model, verbose=args.verbose)

    if not args.question:
        console.print("[red]No question provided.[/red] Use --help for usage.")
        return 3

    return ask_question(
        cfg,
        question=args.question,
        collection_filter=args.collection,
        generation_model_override=args.model,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

