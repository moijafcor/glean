"""
glean Web GUI — FastAPI + HTMX. Wraps glean.py as a local web app.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from queue import Queue

import ollama
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.concurrency import run_in_threadpool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glean as glean_lib

app = FastAPI(title="glean")

# Static and templates relative to this file
BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

CONFIG_PATH = os.environ.get("GLEAN_CONFIG", str(BASE_DIR / "glean.yaml"))
cfg = glean_lib.load_config(CONFIG_PATH)
chroma_client = glean_lib.get_chroma_client(glean_lib.expand_path(cfg["state_dir"]))

glean_lib._set_ollama_host(cfg["ollama_url"])
try:
    _models_resp = ollama.list()
    _model_list = _models_resp.get("models") if isinstance(_models_resp, dict) else getattr(_models_resp, "models", [])
    AVAILABLE_MODELS: list[str] = sorted(
        (m.get("name") if isinstance(m, dict) else getattr(m, "model", None) or getattr(m, "name", None))
        for m in (_model_list or [])
        if (m.get("name") if isinstance(m, dict) else getattr(m, "model", None) or getattr(m, "name", None))
    )
except Exception:
    AVAILABLE_MODELS = []

if not AVAILABLE_MODELS:
    AVAILABLE_MODELS = [cfg.get("generation_model", "qwen2.5-coder:7b")]


def _run_ollama_stream(model: str, system: str, user_content: str, token_queue: Queue) -> None:
    """Run blocking ollama stream in a thread; push tokens into token_queue."""
    glean_lib._set_ollama_host(cfg["ollama_url"])
    try:
        stream = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            options={"temperature": 0.1},
            stream=True,
        )
        for chunk in stream:
            msg = chunk.get("message") if isinstance(chunk, dict) else getattr(chunk, "message", None)
            if msg:
                content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
                if content:
                    token_queue.put(content)
    except Exception as e:
        token_queue.put({"__error__": str(e)})
    finally:
        token_queue.put(None)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    collections = list(cfg["collections"].keys())
    return templates.TemplateResponse(
        "base.html",
        {
            "request": request,
            "collections": collections,
            "generation_model": cfg.get("generation_model", "qwen2.5-coder:7b"),
            "available_models": AVAILABLE_MODELS,
        },
    )


@app.post("/ask")
async def ask(request: Request):
    form = await request.form()
    question = (form.get("question") or "").strip()
    if not question:
        async def _empty():
            yield "event: done\ndata: \n\n"
        return StreamingResponse(_empty(), media_type="text/event-stream")
    collection = form.get("collection") or None
    model = form.get("model") or cfg.get("generation_model", "qwen2.5-coder:7b")

    async def event_stream():
        glean_lib._set_ollama_host(cfg["ollama_url"])
        # 1. Embed + retrieve + build context (blocking — run in thread pool)
        def do_retrieve():
            q_embedding = glean_lib.embed_single(cfg["embedding_model"], question)
            top_k = int(cfg.get("top_k", 20))
            max_distance = cfg.get("max_distance")
            if max_distance is not None:
                max_distance = float(max_distance)
            all_docs, all_metas, all_ids = [], [], []
            collections = [collection] if collection else list(cfg["collections"].keys())
            for name in collections:
                coll = glean_lib.get_collection(chroma_client, name)
                try:
                    res = coll.query(
                        query_embeddings=[q_embedding],
                        n_results=top_k,
                        include=["documents", "metadatas", "distances"],
                    )
                except Exception:
                    continue
                docs = res.get("documents", [[]])[0]
                metas = res.get("metadatas", [[]])[0]
                ids = res.get("ids", [[]])[0]
                dists = res.get("distances", [[]])[0]
                for doc, meta, cid, dist in zip(docs, metas, ids, dists):
                    if max_distance is None or dist <= max_distance:
                        all_docs.append(doc)
                        all_metas.append(meta)
                        all_ids.append(cid)
            if not all_docs:
                return None, None, None
            combined = {"documents": [all_docs], "metadatas": [all_metas], "ids": [all_ids]}
            max_chars = int(cfg.get("max_context_chars", 24000))
            context, metadatas, _ = glean_lib.build_context_snippets(
                combined, max_chars=max_chars, verbose=False
            )
            user_prompt = f"Context:\n---\n{context}\n\nQuestion: {question}"
            return context, metadatas, user_prompt

        try:
            context, metadatas, user_prompt = await run_in_threadpool(do_retrieve)
        except Exception as e:
            yield f"data: {json.dumps({'token': f'Error: {e}'})}\n\n"
            yield "event: done\ndata: \n\n"
            return

        if context is None:
            yield "data: No relevant documents found.\n\n"
            yield "event: done\ndata: \n\n"
            return

        # 2. Stream LLM tokens from a background thread via queue
        token_queue = Queue()
        loop = asyncio.get_running_loop()
        fut = loop.run_in_executor(
            None,
            _run_ollama_stream,
            model,
            glean_lib.SYSTEM_PROMPT,
            user_prompt,
            token_queue,
        )

        while True:
            try:
                token = await asyncio.wait_for(
                    loop.run_in_executor(None, token_queue.get),
                    timeout=120.0,
                )
            except asyncio.TimeoutError:
                break
            if token is None:
                break
            if isinstance(token, dict) and token.get("__error__"):
                err_msg = "Error: " + token["__error__"]
                yield f"data: {json.dumps({'token': err_msg})}\n\n"
                break
            yield f"data: {json.dumps({'token': token})}\n\n"

        await fut

        # 3. Send sources
        sources = []
        seen = set()
        for meta in (metadatas or []):
            key = (meta.file_name, meta.start_line, meta.end_line)
            if key in seen:
                continue
            seen.add(key)
            sources.append({
                "file": meta.file_name,
                "path": meta.file_path,
                "start": meta.start_line,
                "end": meta.end_line,
                "collection": meta.collection,
                "language": meta.language,
            })
        yield f"event: sources\ndata: {json.dumps(sources)}\n\n"
        yield "event: done\ndata: \n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/status")
async def status_route(request: Request):
    state_dir = glean_lib.expand_path(cfg["state_dir"])
    meta_path = Path(state_dir) / "index_meta.json"
    index_state = glean_lib.IndexState.load(meta_path)

    rows = []
    for name, coll_cfg in cfg["collections"].items():
        roots = tuple(coll_cfg["paths"])
        files = [
            fp for fp in index_state.files
            if any(fp == r or fp.startswith(r + "/") for r in roots)
        ]
        chunks = sum(len(index_state.files[fp].chunk_ids) for fp in files)
        mtimes = [index_state.files[fp].mtime for fp in files]
        last = max(mtimes) if mtimes else None
        rows.append({
            "name": name,
            "files": len(files),
            "chunks": chunks,
            "last_indexed": datetime.fromtimestamp(last).strftime("%Y-%m-%d %H:%M") if last else "—",
        })

    total_chunks = sum(r["chunks"] for r in rows)
    return templates.TemplateResponse(
        "status.html",
        {
            "request": request,
            "rows": rows,
            "total_chunks": total_chunks,
            "embedding_model": cfg.get("embedding_model", ""),
            "generation_model": cfg.get("generation_model", ""),
        },
    )


@app.post("/index")
async def index_route(request: Request):
    form = await request.form()
    collection_filter = form.get("collection") or None
    await run_in_threadpool(glean_lib.index_all, cfg, collection_filter, False)
    return await status_route(request)


@app.post("/reindex")
async def reindex_route(request: Request):
    form = await request.form()
    collection_filter = form.get("collection") or None
    await run_in_threadpool(glean_lib.index_all, cfg, collection_filter, True)
    return await status_route(request)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("GLEAN_PORT", 7777))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
