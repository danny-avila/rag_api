import os
import jwt
import datetime
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from concurrent.futures import ThreadPoolExecutor

from main import app
from app.routes import document_routes

client = TestClient(app)


@pytest.fixture
def auth_headers():
    jwt_secret = "testsecret"
    os.environ["JWT_SECRET"] = jwt_secret
    payload = {
        "id": "testuser",
        "exp": datetime.datetime.now(datetime.timezone.utc)
        + datetime.timedelta(hours=1),
    }
    token = jwt.encode(payload, jwt_secret, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(autouse=True)
def override_vector_store(monkeypatch):
    from app.config import vector_store
    from app.services.vector_store.async_pg_vector import AsyncPgVector
    from app.routes import document_routes

    # Clear the LRU cache and patch the cached function to return dummy embeddings
    document_routes.get_cached_query_embedding.cache_clear()

    def dummy_get_cached_query_embedding(query):
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr(
        document_routes, "get_cached_query_embedding", dummy_get_cached_query_embedding
    )

    # Initialize thread pool for tests since TestClient doesn't run lifespan
    if not hasattr(app.state, "thread_pool") or app.state.thread_pool is None:
        app.state.thread_pool = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="test-worker"
        )

    # Override get_all_ids as an async function - patch at CLASS level to bypass run_in_executor
    async def dummy_get_all_ids(self, executor=None):
        return ["testid1", "testid2"]

    monkeypatch.setattr(AsyncPgVector, "get_all_ids", dummy_get_all_ids)

    # Override get_filtered_ids as an async function.
    async def dummy_get_filtered_ids(self, ids, executor=None):
        dummy_ids = ["testid1", "testid2"]
        return [id for id in dummy_ids if id in ids]

    monkeypatch.setattr(AsyncPgVector, "get_filtered_ids", dummy_get_filtered_ids)

    # Override get_documents_by_ids as an async function.
    async def dummy_get_documents_by_ids(self, ids, executor=None):
        return [
            Document(page_content="Test content", metadata={"file_id": id})
            for id in ids
        ]

    monkeypatch.setattr(
        AsyncPgVector, "get_documents_by_ids", dummy_get_documents_by_ids
    )

    # Override embedding_function with a dummy that doesn't call OpenAI
    class DummyEmbedding:
        def embed_query(self, query):
            return [0.1, 0.2, 0.3]

    vector_store.embedding_function = DummyEmbedding()

    # Override similarity search to return a tuple (Document, score).
    def dummy_similarity_search_with_score_by_vector(self, embedding, k, filter):
        doc = Document(
            page_content="Queried content",
            metadata={
                "file_id": filter.get("file_id", "testid1"),
                "user_id": "testuser",
            },
        )
        return [(doc, 0.9)]

    async def dummy_asimilarity_search_with_score_by_vector(
        self, embedding, k, filter=None, executor=None
    ):
        doc = Document(
            page_content="Queried content",
            metadata={
                "file_id": filter.get("file_id", "testid1") if filter else "testid1",
                "user_id": "testuser",
            },
        )
        return [(doc, 0.9)]

    monkeypatch.setattr(
        AsyncPgVector,
        "similarity_search_with_score_by_vector",
        dummy_similarity_search_with_score_by_vector,
    )
    monkeypatch.setattr(
        AsyncPgVector,
        "asimilarity_search_with_score_by_vector",
        dummy_asimilarity_search_with_score_by_vector,
    )

    # Override document addition functions.
    def dummy_add_documents(self, docs, ids):
        return ids

    async def dummy_aadd_documents(self, docs, ids=None, executor=None):
        return ids

    monkeypatch.setattr(AsyncPgVector, "add_documents", dummy_add_documents)
    monkeypatch.setattr(AsyncPgVector, "aadd_documents", dummy_aadd_documents)

    # Override delete function.
    async def dummy_delete(self, ids=None, collection_only=False, executor=None):
        return None

    monkeypatch.setattr(AsyncPgVector, "delete", dummy_delete)


def test_get_all_ids(auth_headers):
    response = client.get("/ids", headers=auth_headers)
    assert response.status_code == 200
    json_data = response.json()
    assert isinstance(json_data, list)
    assert "testid1" in json_data


def test_get_documents_by_ids(auth_headers):
    response = client.get(
        "/documents", params={"ids": ["testid1"]}, headers=auth_headers
    )
    assert response.status_code == 200
    json_data = response.json()
    assert isinstance(json_data, list)
    assert json_data[0]["page_content"] == "Test content"
    assert json_data[0]["metadata"]["file_id"] == "testid1"


def test_delete_documents(auth_headers):
    response = client.request(
        "DELETE", "/documents", json=["testid1"], headers=auth_headers
    )
    assert response.status_code == 200
    json_data = response.json()
    assert "Documents for" in json_data["message"]


def test_query_embeddings_by_file_id(auth_headers):
    data = {
        "query": "Test query",
        "file_id": "testid1",
        "k": 4,
        "entity_id": "testuser",
    }
    response = client.post("/query", json=data, headers=auth_headers)
    assert response.status_code == 200
    json_data = response.json()
    assert isinstance(json_data, list)
    if json_data:
        doc = json_data[0][0]
        assert doc["page_content"] == "Queried content"


def test_query_include_visual_wraps_response(auth_headers, monkeypatch):
    """include_visual=True returns {chunks, visual_matches} and calls the visual helper."""

    async def fake_visual(request, query, file_ids, text_documents=None):
        assert file_ids == ["testid1"]
        return [
            {
                "file_id": "testid1",
                "page_number": 2,
                "image_path": "/var/rag-visual/testid1/page-2.png",
                "score": 0.87,
            }
        ]

    monkeypatch.setattr(
        document_routes, "_fetch_visual_matches_for_file_ids", fake_visual
    )

    data = {
        "query": "Wie ist das Layout auf Seite 2?",
        "file_id": "testid1",
        "k": 4,
        "entity_id": "testuser",
        "include_visual": True,
    }
    response = client.post("/query", json=data, headers=auth_headers)
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, dict)
    assert "chunks" in body and "visual_matches" in body
    assert body["visual_matches"][0]["page_number"] == 2
    assert body["visual_matches"][0]["score"] == 0.87


def test_query_include_visual_when_text_empty(auth_headers, monkeypatch):
    """When the text index returns nothing but include_visual=True,
    the response still wraps and ships the visual hits (no 404)."""
    from app.services.vector_store.async_pg_vector import AsyncPgVector

    async def no_docs(self, embedding, k, filter=None, executor=None):
        return []

    monkeypatch.setattr(
        AsyncPgVector, "asimilarity_search_with_score_by_vector", no_docs
    )

    async def fake_visual(request, query, file_ids, text_documents=None):
        return [
            {
                "file_id": "testid1",
                "page_number": 1,
                "image_path": "/x/p1.png",
                "score": 0.42,
            }
        ]

    monkeypatch.setattr(
        document_routes, "_fetch_visual_matches_for_file_ids", fake_visual
    )

    response = client.post(
        "/query",
        json={
            "query": "q",
            "file_id": "testid1",
            "k": 4,
            "entity_id": "testuser",
            "include_visual": True,
        },
        headers=auth_headers,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["chunks"] == []
    assert body["visual_matches"][0]["score"] == 0.42


def _patch_visual_merge_deps(
    monkeypatch,
    *,
    text_pages_for_file: dict,
    clip_hits: list,
    coupled_visuals: dict = None,
    text_coupled: bool = True,
    max_pages: int = 4,
):
    """Helper: wire the dependencies of ``_fetch_visual_matches_for_file_ids``
    for the merge-logic tests below.

    ``text_pages_for_file``: {file_id: [page, …]} — what text-search returns
    as (Document, score) with page_number metadata.
    ``coupled_visuals``: {file_id: [{page_number, image_path}, …]} — what the
    visual_chunks DB has for those pages. If None, derive from text_pages.
    ``clip_hits``: list of CLIP-retrieved dicts (pages, scores).
    """
    from app.services.vector_store.async_pg_vector import AsyncPgVector
    from app.routes import document_routes
    from app.utils import visual_embed

    if coupled_visuals is None:
        coupled_visuals = {
            fid: [
                {"file_id": fid, "page_number": p, "image_path": f"/x/{fid}/p-{p}.png"}
                for p in pages
            ]
            for fid, pages in text_pages_for_file.items()
        }

    async def fake_text_search(self, embedding, k, filter=None, executor=None):
        wanted = None
        if filter:
            eq = filter.get("file_id", {})
            if isinstance(eq, dict):
                wanted = eq.get("$eq") or eq.get("$in")
            else:
                wanted = eq
        file_ids = [wanted] if isinstance(wanted, str) else (wanted or [])
        out = []
        for fid in file_ids:
            for page in text_pages_for_file.get(fid, []):
                out.append(
                    (
                        Document(
                            page_content=f"chunk from {fid} p{page}",
                            metadata={
                                "file_id": fid,
                                "page_number": page,
                                "user_id": "testuser",
                            },
                        ),
                        0.9,
                    )
                )
        return out

    monkeypatch.setattr(
        AsyncPgVector, "asimilarity_search_with_score_by_vector", fake_text_search
    )
    monkeypatch.setattr(document_routes, "VISUAL_EMBED_URL", "http://clip.test/embed/image")
    monkeypatch.setattr(document_routes, "VISUAL_TEXT_COUPLED", text_coupled)
    monkeypatch.setattr(document_routes, "VISUAL_TEXT_COUPLED_MAX_PAGES", max_pages)

    class _FakePool:
        pass

    class _DB:
        @classmethod
        async def get_pool(cls):
            return _FakePool()

    import app.services.database as db_mod

    monkeypatch.setattr(db_mod, "PSQLDatabase", _DB)

    async def fake_fetch_coupled(pool, file_id, page_numbers):
        rows = coupled_visuals.get(file_id, [])
        wanted = set(page_numbers)
        return [r for r in rows if r["page_number"] in wanted]

    monkeypatch.setattr(document_routes, "fetch_visual_chunks_for_pages", fake_fetch_coupled)
    monkeypatch.setattr(visual_embed, "fetch_visual_chunks_for_pages", fake_fetch_coupled)

    def fake_embed_text(query):
        return [0.1] * 768

    monkeypatch.setattr(document_routes, "embed_text_query", fake_embed_text)

    async def fake_clip(pool, query_embedding, file_ids, k):
        return [h for h in clip_hits if h["file_id"] in file_ids][:k]

    monkeypatch.setattr(document_routes, "similarity_search_visual", fake_clip)


def test_pages_by_file_from_text_docs_accepts_page_or_page_number():
    """PyPDFLoader writes ``page``, custom loaders write ``page_number``.
    The helper must accept both (prefers page_number), preserve rank order
    within a file, dedupe, and drop chunks without either key."""
    from app.routes.document_routes import _pages_by_file_from_text_docs

    docs = [
        (Document(page_content="a", metadata={"file_id": "f1", "page": 75}), 0.9),
        (Document(page_content="b", metadata={"file_id": "f1", "page": 76}), 0.8),
        (Document(page_content="c", metadata={"file_id": "f1", "page": 75}), 0.7),  # dup
        (Document(page_content="d", metadata={"file_id": "f2", "page_number": 3}), 0.6),
        (Document(page_content="e", metadata={"file_id": "f2"}), 0.5),  # no page → skip
        (Document(page_content="f", metadata={"file_id": "f3", "page_number": "not-int"}), 0.4),
    ]
    out = _pages_by_file_from_text_docs(docs)
    assert out == {"f1": [75, 76], "f2": [3]}


def test_query_with_text_coupling_returns_text_pages_first(auth_headers, monkeypatch):
    """Text pipeline finds pages [5, 8, 12]. visual_chunks also has 50, 51
    (CLIP top hits). Expect visual_matches to contain {5, 8, 12} with
    source='text_coupled' regardless of CLIP score, plus the CLIP hits that
    don't collide, deduped."""
    _patch_visual_merge_deps(
        monkeypatch,
        text_pages_for_file={"testid1": [5, 8, 12]},
        coupled_visuals={
            "testid1": [
                {"file_id": "testid1", "page_number": p, "image_path": f"/x/p-{p}.png"}
                for p in (5, 8, 12, 50, 51)
            ]
        },
        clip_hits=[
            {"file_id": "testid1", "page_number": 50, "image_path": "/x/p-50.png", "score": 0.42},
            {"file_id": "testid1", "page_number": 51, "image_path": "/x/p-51.png", "score": 0.40},
            # This one collides with a text-coupled page — must be deduped.
            {"file_id": "testid1", "page_number": 5, "image_path": "/x/p-5.png", "score": 0.38},
        ],
    )

    response = client.post(
        "/query",
        json={
            "query": "Vietnam group photo",
            "file_id": "testid1",
            "k": 4,
            "entity_id": "testuser",
            "include_visual": True,
        },
        headers=auth_headers,
    )
    assert response.status_code == 200
    body = response.json()
    matches = body["visual_matches"]
    pages = [m["page_number"] for m in matches]
    # Text-coupled first (in text-rank order), then CLIP, with no dups.
    assert pages[:3] == [5, 8, 12]
    assert set(pages) == {5, 8, 12, 50, 51}
    assert all(m["source"] == "text_coupled" for m in matches if m["page_number"] in (5, 8, 12))
    assert all(m["source"] == "clip" for m in matches if m["page_number"] in (50, 51))
    assert all(m["score"] == 1.0 for m in matches if m["source"] == "text_coupled")


def test_query_with_text_coupling_disabled_falls_back_to_clip(auth_headers, monkeypatch):
    """With VISUAL_TEXT_COUPLED=False, behaviour is Phase-3: only CLIP hits
    are returned, no synthetic 1.0 scores, no 'text_coupled' source."""
    _patch_visual_merge_deps(
        monkeypatch,
        text_pages_for_file={"testid1": [5, 8, 12]},
        clip_hits=[
            {"file_id": "testid1", "page_number": 50, "image_path": "/x/p-50.png", "score": 0.42},
            {"file_id": "testid1", "page_number": 51, "image_path": "/x/p-51.png", "score": 0.40},
        ],
        text_coupled=False,
    )

    response = client.post(
        "/query",
        json={
            "query": "anything",
            "file_id": "testid1",
            "k": 4,
            "entity_id": "testuser",
            "include_visual": True,
        },
        headers=auth_headers,
    )
    assert response.status_code == 200
    body = response.json()
    pages = [m["page_number"] for m in body["visual_matches"]]
    assert pages == [50, 51]
    assert all(m["source"] == "clip" for m in body["visual_matches"])


def test_query_with_text_coupling_caps_at_max_pages(auth_headers, monkeypatch):
    """Text pipeline finds 10 pages, but VISUAL_TEXT_COUPLED_MAX_PAGES=4
    caps the text-coupled signal at 4. The remaining pages (ranks 5..10)
    are NOT attached just because the text found them — but CLIP hits for
    pages outside the capped set are still included, deduped."""
    text_pages = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    all_visuals = [
        {"file_id": "testid1", "page_number": p, "image_path": f"/x/p-{p}.png"}
        for p in text_pages
    ]
    _patch_visual_merge_deps(
        monkeypatch,
        text_pages_for_file={"testid1": text_pages},
        coupled_visuals={"testid1": all_visuals},
        clip_hits=[
            # CLIP returns a page OUTSIDE the capped-4 text set — must survive.
            {"file_id": "testid1", "page_number": 77, "image_path": "/x/p-77.png", "score": 0.31},
            # And a page INSIDE the capped set — must be deduped away (primary wins).
            {"file_id": "testid1", "page_number": 10, "image_path": "/x/p-10.png", "score": 0.29},
        ],
        max_pages=4,
    )

    response = client.post(
        "/query",
        json={
            "query": "q",
            "file_id": "testid1",
            "k": 4,
            "entity_id": "testuser",
            "include_visual": True,
        },
        headers=auth_headers,
    )
    assert response.status_code == 200
    body = response.json()
    matches = body["visual_matches"]
    text_coupled_pages = [m["page_number"] for m in matches if m["source"] == "text_coupled"]
    clip_pages = [m["page_number"] for m in matches if m["source"] == "clip"]
    assert text_coupled_pages == [10, 11, 12, 13]  # capped at 4, rank order preserved
    assert 77 in clip_pages
    assert 10 not in clip_pages  # deduped — text-coupled won


def test_embed_local_file(tmp_path, auth_headers, monkeypatch):
    # Monkeypatch RAG_UPLOAD_DIR so the file is within the allowed directory.
    monkeypatch.setattr(document_routes, "RAG_UPLOAD_DIR", str(tmp_path))

    # Create a temporary file inside the patched upload dir.
    test_file = tmp_path / "test.txt"
    test_file.write_text("This is a test document.")

    data = {
        "filepath": "test.txt",
        "filename": "test.txt",
        "file_content_type": "text/plain",
        "file_id": "testid1",
    }
    response = client.post("/local/embed", json=data, headers=auth_headers)
    assert response.status_code == 200, f"Response: {response.text}"
    json_data = response.json()
    assert json_data["status"] is True
    assert json_data["file_id"] == "testid1"


def test_embed_file(tmp_path, auth_headers):
    file_content = "This is a test file for the embed endpoint."
    test_file = tmp_path / "test_embed.txt"
    test_file.write_text(file_content)
    with test_file.open("rb") as f:
        response = client.post(
            "/embed",
            data={"file_id": "testid1", "entity_id": "testuser"},
            files={"file": ("test_embed.txt", f, "text/plain")},
            headers=auth_headers,
        )
    assert response.status_code == 200, f"Response: {response.text}"
    json_data = response.json()
    assert json_data["status"] is True
    assert json_data["file_id"] == "testid1"


def test_load_document_context(auth_headers):
    response = client.get("/documents/testid1/context", headers=auth_headers)
    assert response.status_code == 200, f"Response: {response.text}"
    content = response.text
    assert "testid1" in content or "Test content" in content


def test_embed_file_upload(tmp_path, auth_headers, monkeypatch):
    file_content = "Test content for embed upload."
    test_file = tmp_path / "upload_test.txt"
    test_file.write_text(file_content)

    with test_file.open("rb") as f:
        response = client.post(
            "/embed-upload",
            data={"file_id": "testid1", "entity_id": "testuser"},
            files={"uploaded_file": ("upload_test.txt", f, "text/plain")},
            headers=auth_headers,
        )
    assert response.status_code == 200, f"Response: {response.text}"
    json_data = response.json()
    assert json_data["status"] is True
    assert json_data["file_id"] == "testid1"


def test_query_multiple(auth_headers):
    data = {
        "query": "Test query multiple",
        "file_ids": ["testid1", "testid2"],
        "k": 4,
    }
    response = client.post("/query_multiple", json=data, headers=auth_headers)
    assert response.status_code == 200, f"Response: {response.text}"
    json_data = response.json()
    assert isinstance(json_data, list)
    if json_data:
        doc = json_data[0][0]
        assert doc["page_content"] == "Queried content"


def test_extract_text_from_file(tmp_path, auth_headers):
    """Test the /text endpoint for text extraction without embeddings."""
    file_content = "This is a test file for text extraction.\nIt has multiple lines.\nAnd should be extracted properly."
    test_file = tmp_path / "test_text_extraction.txt"
    test_file.write_text(file_content)

    with test_file.open("rb") as f:
        response = client.post(
            "/text",
            data={"file_id": "test_text_123", "entity_id": "testuser"},
            files={"file": ("test_text_extraction.txt", f, "text/plain")},
            headers=auth_headers,
        )

    assert response.status_code == 200, f"Response: {response.text}"
    json_data = response.json()

    # Check response structure
    assert "text" in json_data
    assert "file_id" in json_data
    assert "filename" in json_data
    assert "known_type" in json_data

    # Check response content
    assert json_data["text"] == file_content
    assert json_data["file_id"] == "test_text_123"
    assert json_data["filename"] == "test_text_extraction.txt"
    assert json_data["known_type"] is True  # text files are known types
