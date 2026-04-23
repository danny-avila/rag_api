"""Tests for the optional multimodal-RAG visual ingest pipeline.

The pipeline is feature-flagged via VISUAL_EMBED_URL. Tests reload the
config module with env vars set so module-level constants pick up.

We never call real pdftoppm or real clip-embed-service here — both are
mocked. The pgvector pool is replaced by a FakePool that records calls.
"""

from __future__ import annotations

import asyncio
import importlib
import json
from pathlib import Path

import pytest


VISUAL_URL = "http://clip.test/embed/image"
VISUAL_TEXT_URL = "http://clip.test/embed/text"


class FakeConn:
    def __init__(self, parent: "FakePool"):
        self.parent = parent

    async def execute(self, sql, *args):
        self.parent.executed.append({"sql": sql, "args": args})
        return "INSERT 0 1"

    async def fetch(self, sql, *args):
        self.parent.fetched.append({"sql": sql, "args": args})
        return self.parent.fetch_result


class _Ctx:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, *exc):
        return False


class FakePool:
    def __init__(self):
        self.executed: list = []
        self.fetched: list = []
        self.fetch_result: list = []

    def acquire(self):
        return _Ctx(FakeConn(self))


@pytest.fixture
def enabled_visual(monkeypatch, tmp_path):
    storage = tmp_path / "rag-visual"
    monkeypatch.setenv("VISUAL_EMBED_URL", VISUAL_URL)
    monkeypatch.setenv("VISUAL_TEXT_EMBED_URL", VISUAL_TEXT_URL)
    monkeypatch.setenv("VISUAL_PAGE_DPI", "100")
    monkeypatch.setenv("VISUAL_STORAGE_ROOT", str(storage))
    monkeypatch.setenv("VISUAL_SCORE_THRESHOLD", "0.25")

    import app.config as config_module
    import app.utils.visual_embed as visual_module

    importlib.reload(config_module)
    importlib.reload(visual_module)
    yield visual_module

    monkeypatch.delenv("VISUAL_EMBED_URL", raising=False)
    importlib.reload(config_module)
    importlib.reload(visual_module)


@pytest.fixture
def disabled_visual(monkeypatch):
    monkeypatch.delenv("VISUAL_EMBED_URL", raising=False)
    import app.config as config_module
    import app.utils.visual_embed as visual_module

    importlib.reload(config_module)
    importlib.reload(visual_module)
    yield visual_module


def _async(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_disabled_feature_is_noop(disabled_visual, tmp_path):
    pdf = tmp_path / "any.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    async def run():
        return await disabled_visual.maybe_embed_visuals(
            file_path=str(pdf),
            file_id="file-1",
            file_ext="pdf",
            user_id="u1",
            executor=None,
        )

    result = asyncio.run(run())
    assert result == 0


def test_non_pdf_is_noop(enabled_visual, tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("hi")

    async def run():
        return await enabled_visual.maybe_embed_visuals(
            file_path=str(f),
            file_id="file-2",
            file_ext="txt",
            user_id="u1",
            executor=None,
        )

    assert asyncio.run(run()) == 0


def test_vector_literal_format(enabled_visual):
    assert enabled_visual._vector_literal([0.1, 0.2]) == "[0.100000,0.200000]"


def test_page_number_from_path(enabled_visual):
    assert enabled_visual._page_number_from_path(Path("/x/page-3.png")) == 3
    assert enabled_visual._page_number_from_path(Path("/x/page-042.png")) == 42
    assert enabled_visual._page_number_from_path(Path("/x/bad.png")) is None


def test_render_pdf_pages_raises_on_fitz_failure(
    enabled_visual, tmp_path, monkeypatch
):
    """If PyMuPDF.open throws (corrupt PDF, missing file, etc.) the
    helper must re-raise RuntimeError so the pipeline can soft-fail
    cleanly."""
    import sys
    import types

    fake_fitz = types.ModuleType("fitz")

    def _raise(*a, **kw):
        raise FileNotFoundError("no such file")

    fake_fitz.open = _raise
    fake_fitz.Matrix = lambda *a, **kw: None
    monkeypatch.setitem(sys.modules, "fitz", fake_fitz)

    out = tmp_path / "pages"
    with pytest.raises(RuntimeError, match="PyMuPDF"):
        enabled_visual.render_pdf_pages("/non/existent.pdf", out, 100)


def test_embed_image_calls_sidecar(enabled_visual, tmp_path, monkeypatch):
    png = tmp_path / "p.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")

    captured = {}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"embedding": [0.1] * 768, "dim": 768, "model": "nomic"}

    def fake_post(url, **kw):
        captured["url"] = url
        captured["timeout"] = kw.get("timeout")
        captured["files"] = kw.get("files")
        return FakeResp()

    import requests

    monkeypatch.setattr(requests, "post", fake_post)

    emb = enabled_visual.embed_image(png)
    assert emb == [0.1] * 768
    assert captured["url"] == VISUAL_URL
    assert captured["files"]["file"][0] == "p.png"


def test_embed_text_query_calls_text_endpoint(enabled_visual, monkeypatch):
    captured = {}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"embedding": [0.2] * 768}

    def fake_post(url, **kw):
        captured["url"] = url
        captured["json"] = kw.get("json")
        return FakeResp()

    import requests

    monkeypatch.setattr(requests, "post", fake_post)

    emb = enabled_visual.embed_text_query("layout seite 2")
    assert emb == [0.2] * 768
    assert captured["url"] == VISUAL_TEXT_URL
    assert captured["json"] == {"text": "layout seite 2"}


def test_persist_visual_chunk_builds_pg_insert(enabled_visual):
    pool = FakePool()
    asyncio.run(
        enabled_visual.persist_visual_chunk(
            pool=pool,
            file_id="f1",
            page_number=2,
            image_path="/var/rag-visual/f1/page-2.png",
            embedding=[0.3] * 768,
            cmetadata={"user_id": "u1", "source": "x.pdf"},
        )
    )
    assert len(pool.executed) == 1
    sql = pool.executed[0]["sql"]
    assert "INSERT INTO visual_chunks" in sql
    assert "ON CONFLICT (file_id, page_number) DO UPDATE" in sql
    args = pool.executed[0]["args"]
    assert args[0] == "f1"
    assert args[1] == 2
    assert args[2] == "/var/rag-visual/f1/page-2.png"
    assert args[3].startswith("[0.3000") and args[3].endswith("]")
    assert json.loads(args[4]) == {"user_id": "u1", "source": "x.pdf"}


def test_similarity_search_visual_drops_low_scores(enabled_visual):
    pool = FakePool()
    pool.fetch_result = [
        {"file_id": "f1", "page_number": 1, "image_path": "/x/p1.png", "score": 0.9},
        {"file_id": "f1", "page_number": 2, "image_path": "/x/p2.png", "score": 0.1},
        {"file_id": "f2", "page_number": 5, "image_path": "/x/p5.png", "score": 0.3},
    ]
    results = asyncio.run(
        enabled_visual.similarity_search_visual(
            pool=pool, query_embedding=[0.1] * 768, file_ids=["f1", "f2"], k=10
        )
    )
    assert [r["page_number"] for r in results] == [1, 5]  # 0.1 dropped by threshold
    assert results[0]["score"] == pytest.approx(0.9)


def test_fetch_visual_chunks_for_pages_returns_only_requested_pages(enabled_visual):
    """Direct lookup by (file_id, page_number). DB returns subset; helper
    passes them through unchanged (order is SQL-imposed — ORDER BY page_number)."""
    pool = FakePool()
    pool.fetch_result = [
        {"file_id": "f1", "page_number": 5, "image_path": "/x/f1/p-05.png"},
        {"file_id": "f1", "page_number": 8, "image_path": "/x/f1/p-08.png"},
        {"file_id": "f1", "page_number": 12, "image_path": "/x/f1/p-12.png"},
    ]
    results = asyncio.run(
        enabled_visual.fetch_visual_chunks_for_pages(
            pool=pool, file_id="f1", page_numbers=[5, 8, 12]
        )
    )
    assert [r["page_number"] for r in results] == [5, 8, 12]
    assert all(r["file_id"] == "f1" for r in results)
    assert "score" not in results[0]  # helper does not synthesize a score

    # Verify the SQL params: file_id + int[] of page numbers.
    assert len(pool.fetched) == 1
    args = pool.fetched[0]["args"]
    assert args[0] == "f1"
    assert list(args[1]) == [5, 8, 12]


def test_fetch_visual_chunks_for_pages_returns_empty_for_missing_pages(enabled_visual):
    """If the DB has no rows for the requested pages (e.g. visual ingest
    soft-failed for that file), the helper returns an empty list — the
    caller treats that as 'no visuals for these pages' and moves on."""
    pool = FakePool()
    pool.fetch_result = []
    results = asyncio.run(
        enabled_visual.fetch_visual_chunks_for_pages(
            pool=pool, file_id="f-missing", page_numbers=[1, 2, 3]
        )
    )
    assert results == []

    # Empty input short-circuits — no DB call at all.
    pool2 = FakePool()
    results2 = asyncio.run(
        enabled_visual.fetch_visual_chunks_for_pages(
            pool=pool2, file_id="f1", page_numbers=[]
        )
    )
    assert results2 == []
    assert pool2.fetched == []


def test_similarity_search_visual_empty_files_returns_empty(enabled_visual):
    pool = FakePool()
    results = asyncio.run(
        enabled_visual.similarity_search_visual(
            pool=pool, query_embedding=[0.1] * 768, file_ids=[], k=10
        )
    )
    assert results == []
    assert pool.fetched == []


def test_full_pipeline_happy_path(enabled_visual, tmp_path, monkeypatch):
    """End-to-end flow (pdftoppm + HTTP + DB are all mocked)."""
    pdf = tmp_path / "flyer.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    import app.config as cfg

    out_dir = Path(cfg.VISUAL_STORAGE_ROOT) / "file-7"

    # Mock pdftoppm: instead of rendering, drop two fake PNGs.
    def fake_render(pdf_path, out, dpi):
        out.mkdir(parents=True, exist_ok=True)
        p1 = out / "page-1.png"
        p2 = out / "page-2.png"
        p1.write_bytes(b"\x89PNG1")
        p2.write_bytes(b"\x89PNG2")
        return [p1, p2]

    monkeypatch.setattr(enabled_visual, "render_pdf_pages", fake_render)

    # Mock embed_image
    def fake_embed(image_path):
        return [0.5] * 768

    monkeypatch.setattr(enabled_visual, "embed_image", fake_embed)

    # Mock DB pool
    pool = FakePool()

    class _DB:
        @classmethod
        async def get_pool(cls):
            return pool

    # Inject fake PSQLDatabase
    import app.services.database as db_mod

    monkeypatch.setattr(db_mod, "PSQLDatabase", _DB)

    async def run():
        return await enabled_visual.maybe_embed_visuals(
            file_path=str(pdf),
            file_id="file-7",
            file_ext="pdf",
            user_id="u1",
            executor=None,
        )

    persisted = asyncio.run(run())
    assert persisted == 2
    assert len(pool.executed) == 2
    assert pool.executed[0]["args"][0] == "file-7"
    assert pool.executed[0]["args"][1] in (1, 2)


def test_full_pipeline_continues_when_single_page_embed_fails(
    enabled_visual, tmp_path, monkeypatch
):
    pdf = tmp_path / "flyer.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    def fake_render(pdf_path, out, dpi):
        out.mkdir(parents=True, exist_ok=True)
        p1 = out / "page-1.png"
        p2 = out / "page-2.png"
        p1.write_bytes(b"\x89PNG1")
        p2.write_bytes(b"\x89PNG2")
        return [p1, p2]

    monkeypatch.setattr(enabled_visual, "render_pdf_pages", fake_render)

    calls = {"n": 0}

    def flaky_embed(image_path):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("sidecar down for page 1")
        return [0.5] * 768

    monkeypatch.setattr(enabled_visual, "embed_image", flaky_embed)

    pool = FakePool()

    class _DB:
        @classmethod
        async def get_pool(cls):
            return pool

    import app.services.database as db_mod

    monkeypatch.setattr(db_mod, "PSQLDatabase", _DB)

    async def run():
        return await enabled_visual.maybe_embed_visuals(
            file_path=str(pdf),
            file_id="file-8",
            file_ext="pdf",
            user_id="u1",
            executor=None,
        )

    persisted = asyncio.run(run())
    # Page 1 failed, page 2 succeeded
    assert persisted == 1
    assert len(pool.executed) == 1
