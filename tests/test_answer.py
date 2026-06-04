"""Tests for the optional grounded-answer layer (app/answer).

The pure orchestration in pipeline.run_answer is tested directly with fake
retriever / generator / compressor callables — no FastAPI, no database, no
language model. The /answer endpoint is tested with the vector store,
embedding, and generator monkeypatched, following the existing rag_api style.
"""
import os
import datetime
import hashlib
import types
from concurrent.futures import ThreadPoolExecutor

import jwt
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from main import app
from app.answer import generator, router
from app.answer.schema import AnswerGeneration, UsedExcerpt
from app.answer.pipeline import run_answer

client = TestClient(app)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def make_gen(answerable, *, cite=(), skip=(), answer="", router_feedback="", missing=""):
    used = [UsedExcerpt(excerpt_index=i, action="CITE") for i in cite]
    used += [UsedExcerpt(excerpt_index=i, action="SKIP") for i in skip]
    return AnswerGeneration(
        used_excerpts=used,
        answerable=answerable,
        evidence_assessment="evidence",
        missing=missing,
        answer=answer,
        router_feedback=router_feedback,
    )


def doc(text, metadata=None):
    return {"page_content": text, "metadata": metadata or {}}


# --------------------------------------------------------------------------
# Unit tests: router.next_query (mechanism 2 — routing decision)
# --------------------------------------------------------------------------
def test_router_stops_when_answerable():
    gen = make_gen(True, cite=[0], answer="x")
    assert router.next_query(gen, iteration=1, max_iterations=3) is None


def test_router_retries_with_refined_query():
    gen = make_gen(False, router_feedback="find the date")
    assert router.next_query(gen, iteration=1, max_iterations=3) == "find the date"


def test_router_stops_on_last_iteration():
    gen = make_gen(False, router_feedback="still missing")
    assert router.next_query(gen, iteration=3, max_iterations=3) is None


def test_router_stops_when_no_feedback():
    gen = make_gen(False, router_feedback="", missing="a gap with no query")
    assert router.next_query(gen, iteration=1, max_iterations=3) is None


# --------------------------------------------------------------------------
# Unit tests: router compression helpers (pure)
# --------------------------------------------------------------------------
def test_cite_summary_uses_digest():
    d = {"page_content": "AAAA", "metadata": {"digest": "xyz"}}
    assert router.cite_summary([d]) == "id=xyz AAAA"


def test_apply_rewrites_replaces_and_drops():
    a = {"page_content": "AAAA", "metadata": {}}
    b = {"page_content": "BBBB", "metadata": {}}
    da = hashlib.md5(b"AAAA").hexdigest()
    out = router.apply_rewrites([a, b], [{"id": da, "text": "A!"}])
    # Only the rewritten chunk survives; the omitted one is dropped.
    assert len(out) == 1
    assert out[0]["page_content"] == "A!"
    assert out[0]["metadata"]["compressed"] is True


# --------------------------------------------------------------------------
# Unit tests: pipeline.run_answer (mechanism 1 — context assembly schema)
# --------------------------------------------------------------------------
def test_answerable_first_iteration():
    docs = [doc("A"), doc("B")]

    def retrieve(query, k, exclude):
        return docs[:k]

    def generate(query, excerpts, **kwargs):
        return make_gen(True, cite=[0], answer="hello")

    resp = run_answer(
        "q", retrieve, generate, max_iterations=1, top_k=4, generate_answer=True
    )

    assert resp.answerable is True
    assert resp.answer == "hello"
    assert resp.iterations == 1
    # CITE-выбор отключён: used_excerpts — CoT-самопроверка, не фильтр контекста,
    # поэтому cited = весь контекст, хотя сгенерён CITE только на [0].
    assert [c["page_content"] for c in resp.cited] == ["A", "B"]


def test_schema_only_leaves_answer_empty():
    """Default (mode=schema): schema is filled but answer stays empty."""
    docs = [doc("A")]

    def retrieve(query, k, exclude):
        return docs

    def generate(query, excerpts, **kwargs):
        return make_gen(True, cite=[0], answer="should be dropped")

    resp = run_answer("q", retrieve, generate, max_iterations=1, top_k=4)

    assert resp.answerable is True
    assert resp.answer == ""  # upstream client generates the text
    assert [c["page_content"] for c in resp.cited] == ["A"]


def test_skip_does_not_filter_cited_on_current_iteration():
    """CITE-выбор отключён: used_excerpts — CoT-самопроверка, не фильтр. На текущей
    итерации cited = весь контекст (SKIP влияет только на перенос в варианте 3)."""
    docs = [doc("A"), doc("B")]

    def retrieve(query, k, exclude):
        return docs

    def generate(query, excerpts, **kwargs):
        return make_gen(True, cite=[0], skip=[1], answer="x")

    resp = run_answer("q", retrieve, generate, max_iterations=1, top_k=4)

    assert [c["page_content"] for c in resp.cited] == ["A", "B"]


def test_explicit_refusal_single_iteration():
    def retrieve(query, k, exclude):
        return [doc("A")]

    def generate(query, excerpts, **kwargs):
        return make_gen(False, router_feedback="need more", missing="need more")

    resp = run_answer("q", retrieve, generate, max_iterations=1, top_k=4)

    assert resp.answerable is False
    assert resp.answer == ""
    assert resp.iterations == 1
    assert resp.refusal_reason  # non-empty explicit reason


def test_all_cite_mode_keeps_every_retrieved_doc():
    docs = [doc("A"), doc("B"), doc("C")]

    def retrieve(query, k, exclude):
        return docs

    def generate(query, excerpts, **kwargs):
        # All-CITE: the generator marks every excerpt CITE, none SKIP.
        return make_gen(True, cite=[0, 1, 2], answer="x")

    resp = run_answer("q", retrieve, generate, max_iterations=1, top_k=4)

    assert [c["page_content"] for c in resp.cited] == ["A", "B", "C"]


# --------------------------------------------------------------------------
# Unit tests: pipeline.run_answer (mechanism 2 — router retry + evidence)
# --------------------------------------------------------------------------
def test_router_refines_query_and_excludes_seen():
    retrieve_calls = []
    gen_calls = []

    def retrieve(query, k, exclude):
        retrieve_calls.append((query, set(exclude)))
        return [doc("A")] if not exclude else [doc("B")]

    def generate(query, excerpts, **kwargs):
        gen_calls.append(list(excerpts))
        if len(gen_calls) == 1:
            return make_gen(False, cite=[0], router_feedback="find B", missing="B")
        return make_gen(True, cite=[0, 1], answer="AB")

    resp = run_answer(
        "q", retrieve, generate, max_iterations=2, top_k=4, generate_answer=True
    )

    assert resp.iterations == 2
    assert resp.answer == "AB"
    assert retrieve_calls[0][0] == "q"
    assert retrieve_calls[1][0] == "find B"
    assert len(retrieve_calls[1][1]) == 1  # dedup excluded the first digest
    assert gen_calls[1] == ["A", "B"]  # accumulated context across iterations
    assert [c["page_content"] for c in resp.cited] == ["A", "B"]


def test_evidence_and_missing_carried_forward():
    seen_kwargs = []

    def retrieve(query, k, exclude):
        return [doc("A")] if not exclude else [doc("B")]

    def generate(query, excerpts, *, prior_evidence="", prior_missing=""):
        seen_kwargs.append((prior_evidence, prior_missing))
        if len(seen_kwargs) == 1:
            return make_gen(False, cite=[0], router_feedback="more", missing="gap1")
        return make_gen(True, cite=[0, 1], answer="done")

    run_answer("q", retrieve, generate, max_iterations=2, top_k=4, mode="answer")

    assert seen_kwargs[0] == ("", "")
    # Iteration 2 receives the prior evidence and the prior gap.
    assert seen_kwargs[1] == ("evidence", "gap1")


def test_missing_populates_refusal():
    def retrieve(query, k, exclude):
        return [doc("A")]

    def generate(query, excerpts, **kwargs):
        return make_gen(False, cite=[0], missing="need the 2023 figure")

    resp = run_answer("q", retrieve, generate, max_iterations=1, top_k=4, mode="answer")

    assert resp.missing == "need the 2023 figure"
    assert resp.refusal_reason == "need the 2023 figure"


def test_compression_shrinks_accumulated_context():
    gen_calls = []
    full_text = "AAAA full long passage"
    digest_a = hashlib.md5(full_text.encode()).hexdigest()

    def retrieve(query, k, exclude):
        return [doc(full_text)] if not exclude else [doc("B")]

    def generate(query, excerpts, **kwargs):
        gen_calls.append(list(excerpts))
        if len(gen_calls) == 1:
            return make_gen(False, cite=[0], router_feedback="find B", missing="B")
        return make_gen(True, cite=list(range(len(excerpts))), answer="done")

    def compress_fn(query, cite_summary):
        return [{"id": digest_a, "text": "A!"}]

    resp = run_answer(
        "q", retrieve, generate, max_iterations=2, top_k=4,
        mode="answer", compress_fn=compress_fn,
    )

    # Iteration 2 sees the compressed span, not the full passage.
    assert gen_calls[1][0] == "A!"
    assert resp.compressed is True
    # Raw retrieval is preserved before compression (for honest recall).
    raw_texts = [c["page_content"] for c in resp.retrieved]
    assert full_text in raw_texts
    assert "A!" not in raw_texts


def test_compress_fn_not_called_when_answerable():
    calls = []

    def retrieve(query, k, exclude):
        return [doc("A")]

    def generate(query, excerpts, **kwargs):
        return make_gen(True, cite=[0], answer="x")

    def compress_fn(query, cite_summary):
        calls.append(1)
        return []

    run_answer(
        "q", retrieve, generate, max_iterations=3, top_k=4,
        mode="answer", compress_fn=compress_fn,
    )

    assert calls == []  # no retry => no compression


def test_refusal_after_exhausting_iterations():
    retrieve_calls = []

    def retrieve(query, k, exclude):
        retrieve_calls.append(query)
        return [doc(f"chunk-{len(retrieve_calls)}")]

    def generate(query, excerpts, **kwargs):
        return make_gen(False, router_feedback="still missing", missing="x")

    resp = run_answer("q", retrieve, generate, max_iterations=3, top_k=4)

    assert resp.answerable is False
    assert resp.iterations == 3
    assert len(retrieve_calls) == 3
    assert resp.refusal_reason


def test_single_iteration_does_not_retry_even_if_not_answerable():
    retrieve_calls = []

    def retrieve(query, k, exclude):
        retrieve_calls.append(query)
        return [doc("A")]

    def generate(query, excerpts, **kwargs):
        return make_gen(False, router_feedback="more", missing="more")

    run_answer("q", retrieve, generate, max_iterations=1, top_k=4)

    assert len(retrieve_calls) == 1  # no retry when max_iterations == 1


# --------------------------------------------------------------------------
# Unit test: generator free mode (raw text, no schema)
# --------------------------------------------------------------------------
def test_free_mode_returns_raw_text_all_cite(monkeypatch):
    class FakeModel:
        def invoke(self, messages):
            return types.SimpleNamespace(content="raw answer text")

    monkeypatch.setattr(generator, "get_chat_model", lambda: FakeModel())

    gen = generator.generate("q", ["x", "y"], mode="free")

    assert gen.answer == "raw answer text"
    assert gen.answerable is True
    assert [u.action for u in gen.used_excerpts] == ["CITE", "CITE"]


# --------------------------------------------------------------------------
# Endpoint tests: /answer
# --------------------------------------------------------------------------
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


@pytest.fixture
def ensure_thread_pool():
    if not hasattr(app.state, "thread_pool") or app.state.thread_pool is None:
        app.state.thread_pool = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="test-worker"
        )


def test_answer_disabled_returns_404(auth_headers, ensure_thread_pool, monkeypatch):
    from app.answer import config as answer_config

    monkeypatch.setattr(answer_config, "RAG_ANSWER_ENABLED", False)

    resp = client.post(
        "/answer",
        json={"query": "q", "file_id": "testid1"},
        headers=auth_headers,
    )
    assert resp.status_code == 404


def _enable(monkeypatch, answer_config, mode):
    monkeypatch.setattr(answer_config, "RAG_ANSWER_ENABLED", True)
    monkeypatch.setattr(answer_config, "RAG_ANSWER_MODE", mode)
    monkeypatch.setattr(answer_config, "RAG_ANSWER_COMPRESS", False)
    monkeypatch.setattr(answer_config, "RAG_ANSWER_MODEL", "test-model")
    monkeypatch.setattr(answer_config, "RAG_ANSWER_BASE_URL", "http://localhost")
    monkeypatch.setattr(answer_config, "RAG_ANSWER_API_KEY", "test-key")
    monkeypatch.setattr(answer_config, "RAG_ANSWER_MAX_ITERATIONS", 1)
    monkeypatch.setattr(answer_config, "RAG_ANSWER_TOP_K", 4)


def test_answer_enabled_returns_grounded_answer(
    auth_headers, ensure_thread_pool, monkeypatch
):
    from app.config import vector_store
    from app.answer import config as answer_config
    from app.answer import generator as answer_generator

    _enable(monkeypatch, answer_config, mode="answer")

    monkeypatch.setattr(
        vector_store,
        "embedding_function",
        types.SimpleNamespace(embed_query=lambda q: [0.1, 0.2, 0.3]),
    )

    def fake_search(embedding, k, filter):
        return [(Document(page_content="Revenue grew 12%.", metadata={}), 0.1)]

    monkeypatch.setattr(
        vector_store, "similarity_search_with_score_by_vector", fake_search
    )

    def fake_generate(query, excerpts, **kwargs):
        return make_gen(True, cite=[0], answer="Revenue grew 12 percent.")

    monkeypatch.setattr(answer_generator, "generate", fake_generate)

    resp = client.post(
        "/answer",
        json={"query": "How did revenue change?", "file_id": "testid1"},
        headers=auth_headers,
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["answerable"] is True
    assert data["answer"] == "Revenue grew 12 percent."
    assert data["iterations"] == 1
    assert len(data["cited"]) == 1


def test_answer_schema_only_leaves_answer_empty(
    auth_headers, ensure_thread_pool, monkeypatch
):
    """Schema mode: endpoint returns the filled schema, answer empty."""
    from app.config import vector_store
    from app.answer import config as answer_config
    from app.answer import generator as answer_generator

    _enable(monkeypatch, answer_config, mode="schema")

    monkeypatch.setattr(
        vector_store,
        "embedding_function",
        types.SimpleNamespace(embed_query=lambda q: [0.1, 0.2, 0.3]),
    )

    def fake_search(embedding, k, filter):
        return [(Document(page_content="Revenue grew 12%.", metadata={}), 0.1)]

    monkeypatch.setattr(
        vector_store, "similarity_search_with_score_by_vector", fake_search
    )

    def fake_generate(query, excerpts, **kwargs):
        return make_gen(True, cite=[0], answer="dropped in schema-only")

    monkeypatch.setattr(answer_generator, "generate", fake_generate)

    resp = client.post(
        "/answer",
        json={"query": "How did revenue change?", "file_id": "testid1"},
        headers=auth_headers,
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["answerable"] is True
    assert data["answer"] == ""  # LibreChat generates from cited
    assert len(data["cited"]) == 1
