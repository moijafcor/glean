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
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import chromadb
import ollama
import yaml

try:
    from rank_bm25 import BM25Okapi as _BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False
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


# -----------------------------
# BM25 Corpus & RRF
# -----------------------------


@dataclass
class BM25Corpus:
    """Parallel keyword index stored as JSON alongside ChromaDB.

    The corpus maps chunk_id → {text, collection} and is updated in sync
    with every ChromaDB upsert/delete in index_collection.  At query time
    a BM25Okapi instance is built lazily and cached for the process lifetime.
    """

    entries: Dict[str, Dict[str, str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._bm25: Any = None
        self._bm25_ids: List[str] = []

    @classmethod
    def load(cls, path: Path) -> "BM25Corpus":
        if not path.exists():
            return cls()
        try:
            with path.open("r", encoding="utf-8") as f:
                entries = json.load(f)
        except Exception:
            return cls()
        return cls(entries=entries)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.entries, f, separators=(",", ":"))

    def add(self, chunk_id: str, text: str, collection: str) -> None:
        self.entries[chunk_id] = {"text": text, "collection": collection}
        self._bm25 = None  # invalidate cache

    def remove_many(self, chunk_ids: Sequence[str]) -> None:
        changed = any(cid in self.entries for cid in chunk_ids)
        for cid in chunk_ids:
            self.entries.pop(cid, None)
        if changed:
            self._bm25 = None

    def query(
        self,
        query_tokens: List[str],
        top_k: int,
        collection_filter: Optional[str],
    ) -> List[Tuple[str, float]]:
        """Return [(chunk_id, bm25_score), ...] sorted descending, up to top_k."""
        if not _BM25_AVAILABLE or not self.entries:
            return []
        if collection_filter:
            ids = [cid for cid, v in self.entries.items()
                   if v.get("collection") == collection_filter]
        else:
            ids = list(self.entries.keys())
        if not ids:
            return []
        # Rebuild BM25 when the ID set changes (new index run) or cache is cold
        if self._bm25 is None or self._bm25_ids != ids:
            tokenized = [_tokenize(self.entries[cid]["text"]) for cid in ids]
            self._bm25 = _BM25Okapi(tokenized)
            self._bm25_ids = ids
        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(zip(ids, scores.tolist()), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def query_rare_terms(
        self,
        query_tokens: List[str],
        top_k: int,
        collection_filter: Optional[str],
    ) -> List[Tuple[str, float]]:
        """Run BM25 using only the rarest query tokens (highest IDF).

        Prevents common words like 'public', 'address', 'function' from drowning
        out specific identifiers like 'rafael' in the ranking.  Selects up to 3
        tokens whose document-frequency is below 1% of the corpus.
        """
        if not _BM25_AVAILABLE or not self.entries:
            return []
        if not query_tokens:
            return []
        # Compute per-token document frequency
        n = len(self.entries)
        df: Dict[str, int] = {}
        for tok in set(query_tokens):
            df[tok] = sum(1 for e in self.entries.values() if tok in e["text"].lower())
        # Select rare tokens: df < 1% of corpus and > 0
        rare = [t for t in query_tokens if 0 < df.get(t, n) < n * 0.01]
        if not rare:
            # Fall back to least-common tokens among the query set
            rare = sorted(
                [t for t in query_tokens if df.get(t, 0) > 0],
                key=lambda t: df.get(t, n),
            )[:2]
        if not rare:
            return []
        # Combine rare token(s) with the rest of the query tokens so that
        # documents matching both "rafael" AND "ip address" score higher than
        # those matching only "rafael".  Rare tokens are repeated to give them
        # extra weight in the BM25 scoring.
        boosted = rare * 2 + [t for t in query_tokens if t not in rare]
        return self.query(boosted, top_k, collection_filter)


def _bm25_corpus_path(state_dir: str) -> Path:
    return Path(state_dir) / "bm25_corpus.json"


_WORD_RE = re.compile(r"\w+")

# Common English stopwords that add noise without discriminating power
_STOPWORDS = frozenset(
    "a an and are as at be been being by do does from has have he her his how "
    "i if in is it its me my no not of on or our s she so some that the their "
    "them there they this to up us was we were what when where which who will "
    "with you your".split()
)


def _tokenize(text: str) -> List[str]:
    """Word-boundary tokenizer: splits on non-alphanumeric boundaries and
    removes English stopwords.

    "ip: 198.55.58.201" → ["ip", "198", "55", "58", "201"]
    "rafael.pluio.net"  → ["rafael", "pluio", "net"]
    "What is rafael public IP?" → ["rafael", "public", "ip"]
    """
    return [w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS]


def rrf_merge(
    dense_ids: List[str],
    sparse_ids: List[str],
    k: int = 60,
) -> List[str]:
    """Reciprocal Rank Fusion over two ordered ID lists.

    Score = sum(1 / (k + rank)) across lists; higher is better.
    Returns IDs sorted by score descending (deduped).
    """
    scores: Dict[str, float] = {}
    for rank, cid in enumerate(dense_ids, start=1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    for rank, cid in enumerate(sparse_ids, start=1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda c: scores[c], reverse=True)


# -----------------------------
# Config Loading
# -----------------------------


DEFAULT_CONFIG_PATH = "glean.yaml"

# Files that are likely to contain secrets and should never be indexed
_SECRET_FILENAME_PATTERNS = {
    ".env", "*.env", ".env.*",
    "*.pem", "*.key", "*.p12", "*.pfx",
    "id_rsa", "id_ecdsa", "id_ed25519", "id_dsa",
    "credentials.json", "secrets.yaml", "secrets.yml",
}


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
    cfg.setdefault("max_context_chars", 24000)  # ~6000 tokens at ~4 chars/token
    cfg.setdefault("top_k", 30)
    cfg.setdefault("bm25_top_k", 50)
    cfg.setdefault("rrf_k", 60)
    cfg.setdefault("max_distance", None)  # None = no filtering; e.g. 0.65 to drop noise

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


def _is_secret_file(fname: str) -> bool:
    for pat in _SECRET_FILENAME_PATTERNS:
        if fnmatch.fnmatch(fname, pat):
            return True
    return False


def matches_any(path: str, patterns: Sequence[str]) -> bool:
    """Return True if *path* (relative, using forward slashes) matches any pattern."""
    for pat in patterns:
        if pat.endswith("/"):
            # directory exclusion: glob-match against every directory component
            dir_pat = pat.rstrip("/")
            parts = path.replace("\\", "/").split("/")
            if any(fnmatch.fnmatch(part, dir_pat) for part in parts[:-1]):
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
            # Prune excluded directories in-place
            pruned = []
            for d in list(dirnames):
                full = os.path.join(dirpath, d)
                rel = os.path.relpath(full, root_path)
                if matches_any(rel + "/", exclude):
                    continue
                pruned.append(d)
            dirnames[:] = pruned

            for fname in filenames:
                abs_path = os.path.join(dirpath, fname)
                rel = os.path.relpath(abs_path, root_path)
                try:
                    if os.path.getsize(abs_path) > 500 * 1024:
                        continue
                except OSError:
                    continue
                if _is_secret_file(fname):
                    continue
                if matches_any(rel, exclude):
                    continue
                if not any(fnmatch.fnmatch(fname, pat) for pat in include):
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


# Chunk tuple type: (text, start_line, end_line, heading)
Chunk = Tuple[str, int, int, Optional[str]]


def _merge_small_chunks(chunks: List[Chunk], min_chars: int) -> List[Chunk]:
    if not chunks:
        return []
    merged: List[Chunk] = []
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


def _split_long(text: str, start_line: int, max_chars: int) -> List[Chunk]:
    """Fallback: split long text on double newlines up to max_chars, preserving heading=None."""
    paragraphs = text.split("\n\n")
    chunks: List[Chunk] = []
    cur_lines: List[str] = []
    cur_len = 0
    cur_start = start_line
    line_no = start_line
    for para in paragraphs:
        add = para + "\n\n"
        if cur_len + len(add) > max_chars and cur_lines:
            chunk_text = "\n".join(cur_lines).rstrip()
            end_line = line_no - 1
            chunks.append((chunk_text, cur_start, end_line, None))
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
        chunks.append((chunk_text, cur_start, end_line, None))
    return chunks


def chunk_markdown(text: str, base_heading: Optional[str] = None) -> List[Chunk]:
    lines = text.splitlines()
    chunks: List[Chunk] = []
    current_heading = base_heading
    current_lines: List[str] = []
    start_line = 1

    def flush(end_line: int, heading: Optional[str]) -> None:
        if not current_lines:
            return
        section_text = "\n".join(current_lines).rstrip()
        chunks.append((section_text, start_line, end_line, heading))

    for idx, line in enumerate(lines, start=1):
        # Split on H1 and H2
        if line.startswith("# ") or line.startswith("## "):
            flush(idx - 1, current_heading)
            current_heading = line.strip()
            current_lines = [line]
            start_line = idx
        else:
            if not current_lines:
                start_line = idx
            current_lines.append(line)
    flush(len(lines), current_heading)

    processed: List[Chunk] = []
    for text_chunk, s, e, heading in chunks:
        if len(text_chunk) <= 1500:
            processed.append((text_chunk, s, e, heading))
        else:
            # Split on H3 within this section
            sublines = text_chunk.splitlines()
            sub_chunks: List[Chunk] = []
            cur: List[str] = []
            sub_start = s
            for off, line in enumerate(sublines):
                abs_line = s + off
                if line.startswith("### ") and cur:
                    sub_chunks.append(("\n".join(cur).rstrip(), sub_start, abs_line - 1, heading))
                    cur = [line]
                    sub_start = abs_line
                else:
                    cur.append(line)
            if cur:
                sub_chunks.append(("\n".join(cur).rstrip(), sub_start, s + len(sublines) - 1, heading))

            for sub_text, sub_s, sub_e, sub_h in sub_chunks:
                if len(sub_text) <= 1500:
                    processed.append((sub_text, sub_s, sub_e, sub_h))
                else:
                    processed.extend(_split_long(sub_text, sub_s, 1500))

    return _merge_small_chunks(processed, min_chars=100)


def _extract_file_docstring(lines: List[str]) -> Optional[str]:
    """Extract a module-level docstring from Python source lines."""
    # Skip shebang and blank lines
    start = 0
    while start < len(lines) and (lines[start].startswith("#") or not lines[start].strip()):
        start += 1
    if start >= len(lines):
        return None
    first = lines[start]
    for delim in ('"""', "'''"):
        if first.startswith(delim):
            # Check single-line: """..."""
            rest = first[len(delim):]
            if rest.endswith(delim) and len(rest) > len(delim):
                return first
            # Multi-line
            doc_lines = [first]
            for i in range(start + 1, len(lines)):
                doc_lines.append(lines[i])
                if lines[i].strip().endswith(delim):
                    return "\n".join(doc_lines)
            break
    return None


def chunk_python(text: str) -> List[Chunk]:
    lines = text.splitlines()
    chunks: List[Chunk] = []
    current: List[str] = []
    start_line = 1
    file_docstring = _extract_file_docstring(lines)

    def flush(end_line: int) -> None:
        if not current:
            return
        body = "\n".join(current).rstrip()
        text_chunk = (file_docstring + "\n\n" + body) if file_docstring else body
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

    processed: List[Chunk] = []
    for text_chunk, s, e, heading in chunks:
        if len(text_chunk) <= 3000:
            processed.append((text_chunk, s, e, heading))
        else:
            processed.extend(_split_long(text_chunk, s, 3000))
    return _merge_small_chunks(processed, min_chars=50)


def chunk_php(text: str) -> List[Chunk]:
    lines = text.splitlines()
    chunks: List[Chunk] = []
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

    processed: List[Chunk] = []
    for text_chunk, s, e, heading in chunks:
        if len(text_chunk) <= 3000:
            processed.append((text_chunk, s, e, heading))
        else:
            processed.extend(_split_long(text_chunk, s, 3000))
    return _merge_small_chunks(processed, min_chars=50)


def chunk_ts_js(text: str) -> List[Chunk]:
    lines = text.splitlines()
    chunks: List[Chunk] = []
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

    processed: List[Chunk] = []
    for text_chunk, s, e, heading in chunks:
        if len(text_chunk) <= 3000:
            processed.append((text_chunk, s, e, heading))
        else:
            processed.extend(_split_long(text_chunk, s, 3000))
    return _merge_small_chunks(processed, min_chars=50)


def chunk_yaml(text: str) -> List[Chunk]:
    lines = text.splitlines()
    chunks: List[Chunk] = []
    current: List[str] = []
    start_line = 1
    current_key: Optional[str] = None

    def flush(end_line: int) -> None:
        if not current:
            return
        chunks.append(("\n".join(current).rstrip(), start_line, end_line, current_key))

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

    processed: List[Chunk] = []
    for text_chunk, s, e, heading in chunks:
        if len(text_chunk) <= 2000:
            processed.append((text_chunk, s, e, heading))
        else:
            processed.extend(_split_long(text_chunk, s, 2000))
    return _merge_small_chunks(processed, min_chars=50)


# Matches host-entry keys indented 8–16 spaces that look like FQDNs or
# hostnames (contain at least one dot or are purely alphanumeric/hyphen).
_ANSIBLE_HOST_RE = re.compile(
    r"^( {8,16})([a-zA-Z0-9][a-zA-Z0-9_\-]*(?:\.[a-zA-Z0-9_\-]+)+)\s*:"
)


def _is_ansible_inventory(path: str) -> bool:
    """Return True if path looks like an Ansible inventory/hosts YAML file."""
    name = os.path.basename(path).lower()
    parts = path.replace("\\", "/").split("/")
    return (
        name in ("hosts", "hosts.yml", "hosts.yaml")
        or "inventory" in parts
        or "inventories" in parts
    )


def chunk_yaml_inventory(text: str) -> List[Chunk]:
    """Chunk Ansible inventory YAML at the host-entry level.

    Top-level keys (groups) are split normally; within each top-level block
    we further split at deeply-indented FQDN-like host keys so that each
    host gets its own chunk rather than being buried in a 50-host blob.
    """
    lines = text.splitlines()
    chunks: List[Chunk] = []
    current: List[str] = []
    start_line = 1
    current_key: Optional[str] = None

    def flush(end_line: int) -> None:
        if current:
            chunks.append(("\n".join(current).rstrip(), start_line, end_line, current_key))

    top_key_re = re.compile(r"^[a-zA-Z0-9_\-]+:")

    for idx, line in enumerate(lines, start=1):
        is_top = top_key_re.match(line)
        is_host = _ANSIBLE_HOST_RE.match(line)
        if (is_top or is_host) and current:
            flush(idx - 1)
            current = [line]
            start_line = idx
            current_key = line.strip().rstrip(":")
        else:
            if not current:
                start_line = idx
                current_key = line.strip().rstrip(":") if (is_top or is_host) else None
            current.append(line)
    flush(len(lines))

    processed: List[Chunk] = []
    for text_chunk, s, e, heading in chunks:
        if len(text_chunk) <= 2000:
            processed.append((text_chunk, s, e, heading))
        else:
            processed.extend(_split_long(text_chunk, s, 2000))
    return _merge_small_chunks(processed, min_chars=30)


def chunk_json(text: str) -> List[Chunk]:
    try:
        data = json.loads(text)
    except Exception:
        return _merge_small_chunks(_split_long(text, 1, 2000), min_chars=50)

    chunks: List[Chunk] = []
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


def chunk_shell(text: str) -> List[Chunk]:
    lines = text.splitlines()
    chunks: List[Chunk] = []
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

    processed: List[Chunk] = []
    for text_chunk, s, e, heading in chunks:
        if len(text_chunk) <= 2000:
            processed.append((text_chunk, s, e, heading))
        else:
            processed.extend(_split_long(text_chunk, s, 2000))
    return _merge_small_chunks(processed, min_chars=50)


def chunk_config(text: str) -> List[Chunk]:
    lines = text.splitlines()
    chunks: List[Chunk] = []
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

    processed: List[Chunk] = []
    for text_chunk, s, e, heading in chunks:
        if len(text_chunk) <= 1000:
            processed.append((text_chunk, s, e, heading))
        else:
            processed.extend(_split_long(text_chunk, s, 1000))
    return _merge_small_chunks(processed, min_chars=30)


def chunk_generic(text: str) -> List[Chunk]:
    return _merge_small_chunks(_split_long(text, 1, 1000), min_chars=50)


def chunk_file_contents(path: str, text: str) -> List[Chunk]:
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
        if _is_ansible_inventory(path):
            return chunk_yaml_inventory(text)
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


def _set_ollama_host(ollama_url: str) -> None:
    os.environ.setdefault("OLLAMA_HOST", ollama_url)


# nomic-embed-text caps at 8192 tokens. Content with many special characters
# (JSON, shell globs, regexes) can approach 1 token/char, so we cap well below
# the theoretical maximum. 3000 chars is safe for all observed content types.
_EMBED_MAX_CHARS = 3000


def embed_texts(model: str, texts: Sequence[str], batch_size: int = 32) -> List[List[float]]:
    """Embed a batch of document texts for indexing.

    Applies the ``search_document:`` task prefix required by nomic-embed-text
    for asymmetric retrieval (document side).  The prefix is stripped to within
    the char budget so a long prefix never eats into actual content.
    """
    embeddings: List[List[float]] = []
    prefix = "search_document: "
    budget = _EMBED_MAX_CHARS - len(prefix)
    for i in range(0, len(texts), batch_size):
        batch = [prefix + t[:budget] for t in texts[i : i + batch_size]]
        resp = ollama.embed(model=model, input=batch)
        # Support both typed response objects and plain dicts
        if hasattr(resp, "embeddings"):
            batch_emb = resp.embeddings
        else:
            batch_emb = resp.get("embeddings") or resp.get("embedding")
        if not batch_emb:
            raise RuntimeError("Ollama embed response missing 'embeddings'")
        embeddings.extend(batch_emb)
    return embeddings


def embed_single(model: str, text: str) -> List[float]:
    """Embed a single query string.

    Applies the ``search_query:`` task prefix required by nomic-embed-text for
    asymmetric retrieval (query side).
    """
    prefix = "search_query: "
    budget = _EMBED_MAX_CHARS - len(prefix)
    resp = ollama.embed(model=model, input=[prefix + text[:budget]])
    if hasattr(resp, "embeddings"):
        return resp.embeddings[0]
    raw = resp.get("embeddings") or resp.get("embedding")
    if not raw:
        raise RuntimeError("Ollama embed response missing 'embeddings'")
    return raw[0] if isinstance(raw[0], list) else raw


# -----------------------------
# Vector Store (ChromaDB)
# -----------------------------


def get_chroma_client(state_dir: str) -> chromadb.ClientAPI:
    chroma_path = os.path.join(state_dir, "chroma")
    Path(chroma_path).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=chroma_path)


def get_collection(client: chromadb.ClientAPI, name: str) -> chromadb.Collection:
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


# -----------------------------
# Indexing
# -----------------------------


def deterministic_chunk_id(collection: str, rel_path: str, idx: int) -> str:
    return f"{collection}:{rel_path}:{idx}"


def index_collection(
    client: chromadb.ClientAPI,
    cfg: Dict[str, Any],
    index_state: IndexState,
    corpus: BM25Corpus,
    collection_name: str,
    reindex: bool = False,
) -> None:
    embedding_model = cfg["embedding_model"]
    state_dir = expand_path(cfg["state_dir"])
    _set_ollama_host(cfg["ollama_url"])
    chroma_coll = get_collection(client, collection_name)

    files = discover_files(collection_name, cfg)
    if not files:
        console.print(f"[yellow]No files discovered for collection[/yellow] {collection_name}")
        return

    to_add: List[Tuple[str, str]] = []
    to_update: List[Tuple[str, str]] = []
    deleted_chunk_ids: List[str] = []

    # Scope existing_paths to only files that were indexed by THIS collection
    # (chunk IDs are prefixed with collection_name:).  Using both path-prefix
    # AND chunk-id-prefix prevents a _docs sibling collection (sharing the same
    # root paths but a disjoint include pattern) from treating the main
    # collection's files as deleted and wiping them from index_state/corpus.
    coll_roots = tuple(expand_path(p) for p in cfg["collections"][collection_name]["paths"])
    coll_prefix = collection_name + ":"
    existing_paths = {
        fp for fp, entry in index_state.files.items()
        if fp.startswith(coll_roots) and any(
            cid.startswith(coll_prefix) for cid in entry.chunk_ids
        )
    }
    current_paths: set[str] = set()

    for abs_path, rel_path in files:
        try:
            mtime = os.path.getmtime(abs_path)
        except OSError:
            continue
        current_paths.add(abs_path)
        entry = index_state.files.get(abs_path)
        if reindex or entry is None:
            to_add.append((abs_path, rel_path))
        elif entry.mtime != mtime:
            to_update.append((abs_path, rel_path))

    # Deleted files (only within this collection's roots)
    for abs_path in list(existing_paths):
        if abs_path not in current_paths:
            entry = index_state.files.pop(abs_path, None)
            if entry:
                deleted_chunk_ids.extend(entry.chunk_ids)

    if deleted_chunk_ids:
        chroma_coll.delete(ids=deleted_chunk_ids)
        corpus.remove_many(deleted_chunk_ids)

    # Delete old chunks for updated files and count them
    updated_old_chunk_ids: List[str] = []
    for abs_path, _ in to_update:
        entry = index_state.files.get(abs_path)
        if entry and entry.chunk_ids:
            updated_old_chunk_ids.extend(entry.chunk_ids)
    if updated_old_chunk_ids:
        chroma_coll.delete(ids=updated_old_chunk_ids)
        corpus.remove_many(updated_old_chunk_ids)

    to_add_set = {abs_path for abs_path, _ in to_add}
    total_new = 0
    total_updated = 0

    for abs_path, rel_path in to_add + to_update:
        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError as exc:
            console.print(f"[yellow]Skipping unreadable file:[/yellow] {abs_path} ({exc})")
            continue
        chunks = chunk_file_contents(abs_path, text)
        if not chunks:
            continue
        mtime = os.path.getmtime(abs_path)
        lang = detect_language(abs_path)
        ids: List[str] = []
        docs: List[str] = []
        metas: List[Dict[str, Any]] = []
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

        embeddings = embed_texts(embedding_model, docs)
        chroma_coll.upsert(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)
        for cid, doc_text in zip(ids, docs):
            corpus_text = f"# {rel_path}\n{doc_text}"
            corpus.add(cid, corpus_text, collection_name)
        index_state.files[abs_path] = FileIndexEntry(mtime=mtime, chunk_ids=ids)

        if abs_path in to_add_set:
            total_new += len(ids)
        else:
            total_updated += len(ids)

    console.print(
        f"[green]Indexed collection[/green] {collection_name} "
        f"(new chunks: {total_new}, updated chunks: {total_updated}, "
        f"deleted chunks: {len(deleted_chunk_ids) + len(updated_old_chunk_ids)})"
    )


def _bootstrap_corpus(
    client: chromadb.ClientAPI,
    cfg: Dict[str, Any],
    corpus: BM25Corpus,
    collection_filter: Optional[str],
) -> None:
    """Populate corpus from existing ChromaDB data without re-embedding.

    Called automatically when the corpus is detected to be stale (significantly
    fewer entries than chunks known to index_state).  Fetches documents in
    batches of 1000 per collection.
    """
    collections = [collection_filter] if collection_filter else list(cfg["collections"].keys())
    batch_size = 1000
    total = 0
    for name in collections:
        coll = get_collection(client, name)
        offset = 0
        while True:
            try:
                res = coll.get(
                    limit=batch_size,
                    offset=offset,
                    include=["documents", "metadatas"],
                )
            except Exception:
                break
            docs = res.get("documents") or []
            metas = res.get("metadatas") or []
            ids = res.get("ids") or []
            if not ids:
                break
            for cid, doc, meta in zip(ids, docs, metas):
                coll_name = meta.get("collection", name) if meta else name
                file_path = meta.get("file_path", "") if meta else ""
                if file_path:
                    # Reconstruct rel_path from absolute path using collection roots
                    rel = file_path
                    for root_path in cfg["collections"].get(coll_name, {}).get("paths", []):
                        expanded = expand_path(root_path)
                        if file_path.startswith(expanded + "/"):
                            rel = file_path[len(expanded) + 1:]
                            break
                    corpus_text = f"# {rel}\n{doc}"
                else:
                    corpus_text = doc
                corpus.add(cid, corpus_text, coll_name)
            total += len(ids)
            if len(ids) < batch_size:
                break
            offset += batch_size
    console.print(f"[dim]BM25 corpus bootstrapped from ChromaDB: {total} chunks[/dim]")


def index_all(
    cfg: Dict[str, Any],
    collection_filter: Optional[str],
    reindex: bool,
) -> int:
    state_dir = expand_path(cfg["state_dir"])
    index_meta_path = Path(state_dir) / "index_meta.json"
    corpus_path = _bm25_corpus_path(state_dir)
    index_state = IndexState.load(index_meta_path)
    corpus = BM25Corpus.load(corpus_path)
    client = get_chroma_client(state_dir)

    # Auto-bootstrap corpus if it's empty or far behind the known chunk count
    total_known_chunks = sum(
        len(e.chunk_ids) for e in index_state.files.values()
    )
    if _BM25_AVAILABLE and len(corpus.entries) < total_known_chunks * 0.5:
        _bootstrap_corpus(client, cfg, corpus, collection_filter)

    collections = [collection_filter] if collection_filter else list(cfg["collections"].keys())

    for name in collections:
        if name not in cfg["collections"]:
            console.print(f"[red]Unknown collection:[/red] {name}")
            return 3
        try:
            index_collection(client, cfg, index_state, corpus, name, reindex=reindex)
        except Exception as exc:
            console.print(f"[red]Indexing failed for '{name}':[/red] {exc}")
            corpus.save(corpus_path)
            index_state.save(index_meta_path)
            return 2

    corpus.save(corpus_path)
    index_state.save(index_meta_path)
    return 0


# -----------------------------
# Query Pipeline
# -----------------------------


SYSTEM_PROMPT = """You are a knowledgeable assistant that answers questions about software, infrastructure, and project documentation.
Answer based ONLY on the provided context snippets. Read ALL context snippets carefully before answering — the answer may appear in any snippet.
If the context contains the answer, state it directly and cite the source file and line numbers.
If the context does not contain enough information to answer, say so and do not guess."""


def _truncate_context(parts: List[str], max_chars: int) -> List[str]:
    """Return the largest prefix of parts whose joined length fits within max_chars."""
    kept: List[str] = []
    total = 0
    sep = "\n\n---\n\n"
    sep_len = len(sep)
    for part in parts:
        needed = len(part) + (sep_len if kept else 0)
        if total + needed > max_chars:
            break
        kept.append(part)
        total += needed
    return kept


def build_context_snippets(
    results: Dict[str, Any],
    max_chars: int,
    verbose: bool,
) -> Tuple[str, List[ChunkMetadata], List[str]]:
    metadatas: List[ChunkMetadata] = []
    chunk_texts: List[str] = []
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
        parts.append(f"{header}\n{doc}")

    parts = _truncate_context(parts, max_chars)
    # Trim metadatas and chunk_texts to match kept parts
    metadatas = metadatas[: len(parts)]
    chunk_texts = chunk_texts[: len(parts)]

    context = "\n\n---\n\n".join(parts)
    if verbose:
        console.rule("Retrieved Chunks")
        for part in parts:
            console.print(Markdown(f"```text\n{part[:2000]}\n```"))
            console.print()
    return context, metadatas, chunk_texts


def retrieve_chunks(
    cfg: Dict[str, Any],
    question: str,
    q_embedding: List[float],
    collection_filter: Optional[str],
    client: chromadb.ClientAPI,
    corpus: BM25Corpus,
) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
    """Hybrid (dense + BM25 sparse) retrieval with RRF merge.

    Returns (docs, metas, ids) ordered by relevance.  BM25-only hits that are
    not in the dense result set are fetched from ChromaDB in a single batch
    call per collection so their metadatas are available.
    """
    top_k = int(cfg.get("top_k", 20))
    bm25_top_k = int(cfg.get("bm25_top_k", top_k))
    rrf_k = int(cfg.get("rrf_k", 60))
    max_distance: Optional[float] = cfg.get("max_distance")
    if max_distance is not None:
        max_distance = float(max_distance)

    collections = [collection_filter] if collection_filter else list(cfg["collections"].keys())

    # --- Dense retrieval ---
    dense_ids_ranked: List[str] = []
    dense_map: Dict[str, Tuple[str, Dict]] = {}  # chunk_id -> (doc, meta)

    for name in collections:
        chroma_coll = get_collection(client, name)
        try:
            res = chroma_coll.query(
                query_embeddings=[q_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            console.print(f"[dim]Collection {name} query error: {exc}[/dim]")
            continue
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        raw_ids = res.get("ids", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for doc, meta, cid, dist in zip(docs, metas, raw_ids, dists):
            if max_distance is None or dist <= max_distance:
                dense_ids_ranked.append(cid)
                dense_map[cid] = (doc, meta)

    # --- Sparse (BM25) retrieval ---
    query_tokens = _tokenize(question)
    sparse_results = corpus.query(query_tokens, bm25_top_k, collection_filter)
    sparse_ids_ranked = [cid for cid, _ in sparse_results]

    # --- Sparse rare-term pass (boosts exact-match on specific identifiers) ---
    rare_results = corpus.query_rare_terms(query_tokens, bm25_top_k, collection_filter)
    rare_ids_ranked = [cid for cid, _ in rare_results]

    # --- RRF merge (dense + full BM25 + rare-term BM25) ---
    if sparse_ids_ranked or rare_ids_ranked:
        scores: Dict[str, float] = {}
        for rank, cid in enumerate(dense_ids_ranked, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
        for rank, cid in enumerate(sparse_ids_ranked, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
        for rank, cid in enumerate(rare_ids_ranked, start=1):
            # Rare-term list gets double weight — it's the most targeted signal
            scores[cid] = scores.get(cid, 0.0) + 2.0 / (rrf_k + rank)
        merged_ids = sorted(scores, key=lambda c: scores[c], reverse=True)
    else:
        merged_ids = dense_ids_ranked

    # Fetch metadata for BM25-only hits (not in dense results) via batch get
    bm25_only_ids = [cid for cid in merged_ids if cid not in dense_map]
    if bm25_only_ids:
        # Group by collection for efficient batch fetching
        by_collection: Dict[str, List[str]] = {}
        for cid in bm25_only_ids:
            entry = corpus.entries.get(cid)
            if entry:
                cname = entry.get("collection", "")
                by_collection.setdefault(cname, []).append(cid)
        for cname, cids in by_collection.items():
            chroma_coll = get_collection(client, cname)
            try:
                got = chroma_coll.get(ids=cids, include=["documents", "metadatas"])
                for doc, meta, cid in zip(
                    got.get("documents") or [],
                    got.get("metadatas") or [],
                    got.get("ids") or [],
                ):
                    dense_map[cid] = (doc, meta)
            except Exception:
                pass

    # Reconstruct in merged order
    all_docs: List[str] = []
    all_metas: List[Dict[str, Any]] = []
    all_ids: List[str] = []
    for cid in merged_ids:
        if cid in dense_map:
            doc, meta = dense_map[cid]
            all_docs.append(doc)
            all_metas.append(meta)
            all_ids.append(cid)

    return all_docs, all_metas, all_ids


def ask_question(
    cfg: Dict[str, Any],
    question: str,
    collection_filter: Optional[str],
    generation_model_override: Optional[str],
    verbose: bool,
    client: Optional[chromadb.ClientAPI] = None,
) -> int:
    state_dir = expand_path(cfg["state_dir"])
    _set_ollama_host(cfg["ollama_url"])
    if client is None:
        client = get_chroma_client(state_dir)
    max_chars = int(cfg.get("max_context_chars", 24000))

    if collection_filter and collection_filter not in cfg["collections"]:
        console.print(f"[red]Unknown collection:[/red] {collection_filter}")
        return 3

    try:
        q_embedding = embed_single(cfg["embedding_model"], question)
    except Exception as exc:
        console.print(f"[red]Embedding failed:[/red] {exc}")
        return 1

    corpus = BM25Corpus.load(_bm25_corpus_path(state_dir))
    all_docs, all_metas, all_ids = retrieve_chunks(
        cfg, question, q_embedding, collection_filter, client, corpus
    )

    if not all_docs:
        console.print("[yellow]No relevant documents found in the index.[/yellow]")
        return 1

    combined_results = {
        "documents": [all_docs],
        "metadatas": [all_metas],
        "ids": [all_ids],
    }
    context, metadatas, _ = build_context_snippets(combined_results, max_chars=max_chars, verbose=verbose)

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

    # Support both typed response objects and plain dicts
    if hasattr(resp, "message"):
        content = resp.message.content
    else:
        content = resp["message"]["content"] if "message" in resp else str(resp)

    console.rule("[bold]Answer[/bold]")
    console.print(Markdown(content.strip()))
    console.print()
    console.print("[bold]Sources:[/bold]")
    seen: set[Tuple[str, int, int]] = set()
    for meta in metadatas:
        key = (meta.file_name, meta.start_line, meta.end_line)
        if key in seen:
            continue
        seen.add(key)
        console.print(
            f"  {meta.file_path}:{meta.start_line}-{meta.end_line} "
            f"({meta.collection}, {meta.language})"
        )
    return 0


# -----------------------------
# Status
# -----------------------------


def status_command(cfg: Dict[str, Any], client: Optional[chromadb.ClientAPI] = None) -> int:
    state_dir = expand_path(cfg["state_dir"])
    index_meta_path = Path(state_dir) / "index_meta.json"
    index_state = IndexState.load(index_meta_path)
    if client is None:
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
        files_in_coll: List[str] = []
        chunks_in_coll = 0
        last_mtime = 0.0
        for fp, entry in index_state.files.items():
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
            total_bytes = sum(
                os.path.getsize(os.path.join(dirpath, fn))
                for dirpath, _, filenames in os.walk(state_dir)
                for fn in filenames
            )
            size_str = f"{total_bytes / (1024 * 1024):.1f} MB"
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
    state_dir = expand_path(cfg["state_dir"])
    # Reuse a single ChromaDB client for the session
    client = get_chroma_client(state_dir)

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
                status_command(cfg, client=client)
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
            client=client,
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
