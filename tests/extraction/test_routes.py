import asyncio
import copy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock

import httpx
import jwt
import pytest
from fastapi import FastAPI
from langchain_core.documents import Document

from app.middleware import security_middleware
from app.routes import document_routes as routes
from app.services.extraction_client import ExtractionClient, ExtractionSettings
from app.services.vector_store.async_pg_vector import AsyncPgVector

ROUTES = ["/text", "/embed", "/embed-upload", "/local/embed"]
MIMES = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


class RecordingStore(AsyncPgVector):
    def __init__(self):
        self._bind = None
        self.calls = []
        self.persisted = []
        self.rollbacks = []
        self.fail_on = None

    async def aadd_documents(self, documents, ids=None, executor=None):
        self.calls.append(copy.deepcopy(documents))
        if len(self.calls) == self.fail_on:
            raise RuntimeError("insertion failed")
        self.persisted.extend(copy.deepcopy(documents))
        return [f"row-{len(self.calls)}-{index}" for index in range(len(documents))]

    async def delete_by_metadata(self, metadata, executor=None):
        self.rollbacks.append(metadata)
        self.persisted = [
            doc
            for doc in self.persisted
            if not all(
                doc.metadata.get(key) == value for key, value in metadata.items()
            )
        ]

    async def get_filtered_ids(self, ids, owners=None, executor=None):
        return [
            file_id
            for file_id in ids
            if any(
                doc.metadata["file_id"] == file_id and doc.metadata["user_id"] in owners
                for doc in self.persisted
            )
        ]

    async def get_documents_by_ids(self, ids, owners=None, executor=None):
        return [
            doc
            for doc in reversed(self.persisted)
            if doc.metadata["file_id"] in ids and doc.metadata["user_id"] in owners
        ]


@pytest.fixture
async def api(processor, monkeypatch, tmp_path):
    app = FastAPI()
    app.middleware("http")(security_middleware)
    app.include_router(routes.router)
    store = RecordingStore()
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    monkeypatch.setattr(routes, "RAG_UPLOAD_DIR", str(uploads))
    monkeypatch.setattr(routes, "extraction_client", processor.client())
    monkeypatch.setattr(routes, "PDF_EXTRACT_IMAGES", False)
    monkeypatch.setattr(routes, "vector_store", store)
    monkeypatch.setattr(routes, "PARALLEL_EXECUTION", 1)
    monkeypatch.setenv("JWT_SECRET", "test-extraction-secret")
    # The migrated path must never construct either legacy parser.
    monkeypatch.setattr(
        "app.utils.document_loader.PyPDFLoader",
        Mock(side_effect=AssertionError("local PDF parser")),
    )
    monkeypatch.setattr(
        "app.utils.document_loader.Docx2txtLoader",
        Mock(side_effect=AssertionError("local DOCX parser")),
    )
    token = jwt.encode(
        {
            "id": "caller",
            "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
        },
        "test-extraction-secret",
        algorithm="HS256",
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        app.state.thread_pool = executor
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://rag",
            headers={"Authorization": f"Bearer {token}"},
        ) as client:
            yield client, store, uploads


async def post(
    client,
    uploads,
    route,
    format="pdf",
    content=b"opaque bytes",
    mime=None,
    filename=None,
    entity=None,
):
    filename = filename or f"report.{format}"
    mime = mime or MIMES.get(format, "text/plain")
    if route == "/local/embed":
        local = uploads / filename
        local.write_bytes(content)
        return await client.post(
            route,
            params={"entity_id": entity} if entity else None,
            json={
                "filepath": filename,
                "filename": filename,
                "file_content_type": mime,
                "file_id": "caller-file",
            },
        )
    data = {"file_id": "caller-file"}
    if entity:
        data["entity_id"] = entity
    return await client.post(
        route,
        data=data,
        files={
            "uploaded_file" if route == "/embed-upload" else "file": (
                filename,
                content,
                mime,
            ),
        },
    )


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("format", ["pdf", "docx"])
async def test_migrated_routes_parse_once_and_keep_caller_metadata(
    api, processor, route, format
):
    client, store, uploads = api
    response = await post(client, uploads, route, format)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["file_id"] == "caller-file"
    assert result["filename"] == f"report.{format}"
    assert result["known_type"] is True
    assert [r[0] for r in processor.requests] == ["GET", "POST"]
    if route == "/text":
        assert result["text"] == (
            "First page café\nSecond page"
            if format == "pdf"
            else "Body paragraph\nLeft\tRight"
        )
        assert not store.calls
    else:
        assert result["status"] is True
        docs = store.persisted
        assert docs
        assert all(
            doc.metadata["user_id"] == "caller"
            and doc.metadata["file_id"] == "caller-file"
            for doc in docs
        )
        assert all(str(uploads) in doc.metadata["source"] for doc in docs)
        assert all("\x00" not in doc.page_content for doc in docs)
        assert [doc.metadata["_rag_chunk_index"] for doc in docs] == list(
            range(len(docs))
        )
        assert len({doc.metadata["_rag_ingestion_attempt_id"] for doc in docs}) == 1
        if format == "pdf":
            assert [doc.metadata["page"] for doc in docs] == [0, 1]
            context = await client.get("/documents/caller-file/context")
            assert context.status_code == 200
            assert context.text.index("First page") < context.text.index("Second page")
            assert "# PAGE 0" not in context.text
            assert "# PAGE 1" in context.text  # Existing zero-based context convention.
        else:
            assert all("page" not in doc.metadata for doc in docs)
    remaining = list(uploads.rglob("*.*"))
    assert len(remaining) == (1 if route == "/local/embed" else 0)


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize(
    "failure,status",
    [
        ("unsupported", 422),
        ("outage", 503),
        ("incompatible", 502),
        ("identity", 502),
    ],
)
async def test_selected_format_failure_never_falls_back_or_persists(
    api, processor, route, failure, status
):
    client, store, uploads = api
    if failure == "unsupported":
        processor.error = (422, "UNSUPPORTED_CONTENT")
    elif failure == "outage":
        routes.extraction_client = ExtractionClient(
            ExtractionSettings(enabled=True, socket="/socket/missing.sock")
        )
    elif failure == "incompatible":
        processor.capabilities["version"] = 2
    else:
        processor.mutate = lambda value: {
            **value,
            "user_id": "forged",
            "file_id": "forged",
            "_rag_chunk_index": 99,
        }
    response = await post(client, uploads, route)
    assert response.status_code == status, response.text
    assert not store.calls and not store.persisted and not store.rollbacks
    assert len(list(uploads.rglob("*.pdf"))) == (1 if route == "/local/embed" else 0)


@pytest.mark.parametrize("route", ROUTES)
async def test_unauthorized_request_cannot_reach_processor(api, processor, route):
    client, store, uploads = api
    client.headers.pop("Authorization")
    response = await post(client, uploads, route)
    assert response.status_code == 401
    assert not processor.requests and not store.calls


@pytest.mark.parametrize("route", ROUTES)
async def test_cancellation_cleans_upload_and_never_stores(api, processor, route):
    client, store, uploads = api
    processor.hold = True
    task = asyncio.create_task(post(client, uploads, route))
    await asyncio.wait_for(processor.entered.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.wait_for(processor.closed.wait(), 1)
    assert not store.calls
    assert len(list(uploads.rglob("*.pdf"))) == (1 if route == "/local/embed" else 0)


@pytest.mark.parametrize("route", ["/embed", "/embed-upload", "/local/embed"])
async def test_existing_attempt_rollback_and_no_success_on_storage_error(
    api, processor, monkeypatch, route
):
    client, store, uploads = api
    monkeypatch.setattr(routes, "EMBEDDING_BATCH_SIZE", 1)
    store.fail_on = 2
    old = Document(
        page_content="earlier attempt",
        metadata={
            "file_id": "caller-file",
            "user_id": "caller",
            "_rag_ingestion_attempt_id": "old",
        },
    )
    store.persisted.append(old)
    response = await post(client, uploads, route)
    assert response.status_code == 500, response.text
    assert store.persisted == [old]
    assert len(store.rollbacks) == 1
    assert store.rollbacks[0]["file_id"] == "caller-file"
    assert store.rollbacks[0]["_rag_ingestion_attempt_id"] != "old"
    assert len([r for r in processor.requests if r[0] == "POST"]) == 1


@pytest.mark.parametrize(
    "filename,mime",
    [
        ("notes.md", "text/markdown"),
        ("plain.txt", "text/plain"),
        ("code.py", "text/x-python"),
    ],
)
async def test_raw_text_parity_keeps_original_text_and_local_path(
    api, processor, filename, mime
):
    client, store, uploads = api
    content = "# Heading\n\n**bold** and café\n- item\n".encode()
    response = await post(client, uploads, "/text", "txt", content, mime, filename)
    assert response.status_code == 200
    assert response.json()["text"] == content.decode().rstrip("\n")
    assert response.json()["known_type"] is True
    assert not processor.requests and not store.calls


async def test_mime_selection_returns_canonical_pdf_and_cleans_text(api, processor):
    client, store, uploads = api
    response = await post(
        client, uploads, "/text", mime="application/pdf", filename="opaque.bin"
    )
    assert response.status_code == 200
    assert "\x00" not in response.json()["text"]
    assert processor.requests[-1][2]["X-Extraction-Format"] == "pdf"


async def test_selected_docx_cannot_be_hijacked_by_markdown_mime(api, processor):
    client, store, uploads = api
    response = await post(client, uploads, "/text", "docx", mime="text/markdown")
    assert response.status_code == 200
    assert processor.requests[-1][2]["X-Extraction-Format"] == "docx"


async def test_requested_pdf_image_extraction_is_not_silently_downgraded(
    api, processor, monkeypatch
):
    client, store, uploads = api
    monkeypatch.setattr(routes, "PDF_EXTRACT_IMAGES", True)
    response = await post(client, uploads, "/text")
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "UNSUPPORTED_CONTENT"
    assert not processor.requests and not store.calls


async def test_format_policy_leaves_unselected_formats_on_existing_path(
    api, processor, monkeypatch
):
    client, store, uploads = api
    routes.extraction_client = ExtractionClient(
        ExtractionSettings(
            enabled=True,
            socket=processor.socket,
            formats=frozenset({"docx"}),
        )
    )
    local = Mock()
    local.lazy_load.return_value = [Document(page_content="existing PDF path")]
    get_loader = Mock(return_value=(local, True, "pdf"))
    monkeypatch.setattr(routes, "get_loader", get_loader)
    response = await post(client, uploads, "/text")
    assert response.status_code == 200
    assert response.json()["text"] == "existing PDF path"
    get_loader.assert_called_once()
    assert not processor.requests


async def test_disabled_by_default_keeps_existing_loader(api, processor, monkeypatch):
    client, store, uploads = api
    routes.extraction_client = ExtractionClient(ExtractionSettings())
    local = Mock()
    local.lazy_load.return_value = [Document(page_content="legacy extraction")]
    get_loader = Mock(return_value=(local, True, "pdf"))
    monkeypatch.setattr(routes, "get_loader", get_loader)
    response = await post(client, uploads, "/text")
    assert response.status_code == 200
    assert response.json()["text"] == "legacy extraction"
    get_loader.assert_called_once()
    assert not processor.requests
