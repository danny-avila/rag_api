# app/answer/routes.py
"""Маршрут FastAPI для дополнительного эндпоинта обоснованного ответа.

Самодостаточный роутер, подключается из main.py одной строкой. Использует vector_store
оригинального rag_api (только чтение) и не меняет его файлы, поэтому `git pull` rag_api
проходит без конфликтов.
"""
import asyncio
import hashlib
import traceback

from fastapi import APIRouter, HTTPException, Request

from app.config import logger, vector_store
from app.answer import config as answer_config
from app.answer import generator, pipeline
from app.answer.schema import AnswerRequestBody

router = APIRouter()


@router.post("/answer")
async def answer_query(body: AnswerRequestBody, request: Request):
    """Обоснованный ответ по извлечённым отрывкам (опционально, по умолчанию выключено).

    Включается через RAG_ANSWER_ENABLED. Запускает схему сборки контекста (CITE/SKIP +
    answerable + явный отказ), а при RAG_ANSWER_MAX_ITERATIONS > 1 — цикл маршрутизатора,
    который уточняет запрос и исключает уже виденные отрывки на ретрае. Чистый поиск
    (/query) не затрагивается.
    """
    if not answer_config.RAG_ANSWER_ENABLED:
        raise HTTPException(
            status_code=404,
            detail="Answer generation is disabled. Set RAG_ANSWER_ENABLED=true.",
        )
    answer_config.require_answer_config()

    if not hasattr(request.state, "user"):
        user_authorized = body.entity_id if body.entity_id else "public"
    else:
        user_authorized = (
            body.entity_id if body.entity_id else request.state.user.get("id")
        )

    top_k = body.k or answer_config.RAG_ANSWER_TOP_K
    max_iterations = body.max_iterations or answer_config.RAG_ANSWER_MAX_ITERATIONS
    mode = body.mode or answer_config.RAG_ANSWER_MODE

    # Сжатие на стороне маршрутизатора — вызов модели, внедряемый здесь, чтобы модуль
    # router оставался без модели, а оригинальные файлы rag_api не менялись.
    compress_fn = None
    if answer_config.RAG_ANSWER_COMPRESS:

        def compress_fn(query: str, cite_summary: str):
            return generator.compress(query, cite_summary)

    # Маршрутизатор-агент (один вызов модели), если включён флагом. Тогда pipeline
    # использует его вместо детерминированного next_query. Оригинальные файлы не трогаем.
    route_fn = None
    if answer_config.RAG_ANSWER_ROUTER_AGENT:

        def route_fn(main_query, evidence, cite_summary, prev_subqueries, iteration, max_it):
            return generator.route(
                main_query, evidence, cite_summary, prev_subqueries, iteration, max_it
            )

    def _retrieve(query: str, k: int, seen_chunk_ids):
        """Синхронный поиск, исключающий уже виденные отрывки (для маршрутизатора)."""
        embedding = vector_store.embedding_function.embed_query(query)
        fetch_k = k + len(seen_chunk_ids)
        documents = vector_store.similarity_search_with_score_by_vector(
            embedding, k=fetch_k, filter={"file_id": {"$eq": body.file_id}}
        )
        fresh = []
        for doc, _score in documents:
            doc_user_id = doc.metadata.get("user_id")
            if doc_user_id is not None and doc_user_id != user_authorized:
                continue
            chunk_id = hashlib.md5(doc.page_content.encode()).hexdigest()
            if chunk_id in seen_chunk_ids:
                continue
            fresh.append({"page_content": doc.page_content, "metadata": doc.metadata})
            if len(fresh) >= k:
                break
        return fresh

    def _generate(query, excerpts, *, prior_evidence="", prior_missing=""):
        return generator.generate(
            query,
            excerpts,
            mode=mode,
            prior_evidence=prior_evidence,
            prior_missing=prior_missing,
        )

    def _run():
        return pipeline.run_answer(
            body.query,
            _retrieve,
            _generate,
            max_iterations=max_iterations,
            top_k=top_k,
            mode=mode,
            compress_fn=compress_fn,
            route_fn=route_fn,
        )

    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(request.app.state.thread_pool, _run)
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(
            "Error in answer | File ID: %s | Query: %s | Error: %s | Traceback: %s",
            body.file_id,
            body.query,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail=str(e))
