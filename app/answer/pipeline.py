# app/answer/pipeline.py
"""Оркестрация ответа: retrieve -> генератор -> маршрутизатор.

Чистая логика: retrieve_fn / generate_fn / compress_fn / route_fn передаются
извне, поэтому цикл тестируется без БД и без живой модели.
"""
import hashlib
from typing import Callable, Dict, List, Optional, Set

from app.answer import router
from app.answer.schema import AnswerGeneration, AnswerResponse, RouterDecision

# (запрос, top_k, seen_chunk_ids) -> список {page_content, metadata, score}
RetrieveFn = Callable[[str, int, Set[str]], List[Dict]]
# (запрос, отрывки, *, prior_evidence, prior_missing) -> AnswerGeneration
GenerateFn = Callable[..., AnswerGeneration]
# (запрос, cite_summary) -> список {id, text}
CompressFn = Callable[[str, str], List[Dict]]
# (main_query, evidence, cite_summary, subqueries, итерация, max) -> RouterDecision
RouteFn = Callable[..., RouterDecision]


def _chunk_id(page_content: str) -> str:
    """Идентификатор отрывка по содержимому (md5)."""
    return hashlib.md5(page_content.encode()).hexdigest()


def _dedup_docs(docs: List[Dict]) -> List[Dict]:
    seen: Set[str] = set()
    out: List[Dict] = []
    for d in docs:
        cid = _chunk_id(d["page_content"])
        if cid not in seen:
            seen.add(cid)
            out.append(d)
    return out


def _retrieve_fresh(retrieve_fn, query, k, seen_chunk_ids, retrieved_raw):
    """retrieve + отбросить уже виденные чанки, обновить seen и сырой снимок."""
    fresh: List[Dict] = []
    for doc in retrieve_fn(query, k, seen_chunk_ids):
        cid = _chunk_id(doc["page_content"])
        if cid in seen_chunk_ids:
            continue
        seen_chunk_ids.add(cid)
        fresh.append(doc)
    retrieved_raw.extend(fresh)
    return fresh


def _answered(gen, cited, retrieved_raw, compressed, iteration, trace, mode):
    """Ответ найден: финальный AnswerResponse с answerable=True."""
    return AnswerResponse(
        answer=gen.answer if mode in ("answer", "free") else "",
        answerable=True,
        evidence_assessment=gen.evidence_assessment,
        missing="",
        used_excerpts=gen.used_excerpts,
        cited=_as_payload(cited),
        retrieved=_as_payload(retrieved_raw),
        compressed=compressed,
        iterations=iteration,
        trace=trace,
    )


def _refused(last_gen, cited, retrieved_raw, compressed, iteration, trace):
    """Ответа нет: отказ берём из answer генератора, не выдумываем."""
    return AnswerResponse(
        answer=last_gen.answer,
        answerable=False,
        evidence_assessment=last_gen.evidence_assessment,
        missing=last_gen.missing,
        used_excerpts=last_gen.used_excerpts,
        cited=_as_payload(cited),
        retrieved=_as_payload(retrieved_raw),
        compressed=compressed,
        iterations=iteration or 1,
        trace=trace,
        refusal_reason=last_gen.missing
        or last_gen.evidence_assessment
        or "The retrieved context does not support an answer.",
    )


def run_answer(
    query: str,
    retrieve_fn: RetrieveFn,
    generate_fn: GenerateFn,
    *,
    max_iterations: int,
    top_k: int,
    generate_answer: bool = False,
    mode: Optional[str] = None,
    compress_fn: Optional[CompressFn] = None,
    route_fn: Optional[RouteFn] = None,
) -> AnswerResponse:
    """Прогон цикла ответа, возврат финального результата.

    mode in {"free","schema","answer"} (если None — из generate_answer). При
    "schema" финальный answer пуст (заполняется снаружи), при "answer"/"free" —
    возвращается текст.

    Два способа ретрая:
    - без маршрутизатора-агента (route_fn=None): генератор сам помечает пробел,
      retrieve добирает по уточнённому запросу, compress_fn сжимает контекст.
    - с маршрутизатором-агентом (route_fn задан): генератор отвечает на главный
      запрос и останавливается, как только ответ полон. Иначе маршрутизатор называет
      пробел, формулирует подзапрос для добора и сжимает процитированные отрывки.
    """
    if mode is None:
        mode = "answer" if generate_answer else "schema"

    if route_fn is not None:
        return _run_answer_agentic(
            query, retrieve_fn, generate_fn, route_fn,
            max_iterations=max_iterations, top_k=top_k, mode=mode,
        )

    seen_chunk_ids: Set[str] = set()
    cited_docs: List[Dict] = []
    retrieved_raw: List[Dict] = []  # все свежие чанки до компрессии (для recall)
    current_query = query
    last_gen = AnswerGeneration(answerable=False)
    prior_evidence = ""
    prior_missing = ""
    compressed_flag = False
    trace: List[Dict] = []
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        fresh = _retrieve_fresh(retrieve_fn, current_query, top_k, seen_chunk_ids, retrieved_raw)
        context_docs = cited_docs + fresh
        excerpts = [doc["page_content"] for doc in context_docs]

        gen = generate_fn(
            current_query,
            excerpts,
            prior_evidence=prior_evidence,
            prior_missing=prior_missing,
        )
        last_gen = gen

        # CITE-выбор отключён: used_excerpts — шаг рассуждения (CoT), не фильтр.
        # cited = весь контекст, иначе модель сужает его до 2-3 фрагментов и теряет данные.
        cited_docs = _dedup_docs(list(context_docs))

        step = {
            "iteration": iteration,
            "in_query": current_query,
            "in_context_chunks": len(context_docs),
            "in_fresh_retrieved": len(fresh),
            "in_prior_evidence": prior_evidence,
            "in_prior_missing": prior_missing,
            "out_answerable": gen.answerable,
            "out_cite": len([u for u in gen.used_excerpts if u.action == "CITE"]),
            "out_evidence": gen.evidence_assessment,
            "out_missing": gen.missing,
            "out_router_feedback": gen.router_feedback,
            "out_answer": gen.answer,
            "action": None,
        }
        trace.append(step)

        if gen.answerable:
            step["action"] = "answer"
            return _answered(gen, cited_docs, retrieved_raw, compressed_flag, iteration, trace, mode)

        # Не answerable: переносим evidence и пробел в следующую генерацию.
        prior_evidence = gen.evidence_assessment
        prior_missing = gen.missing

        refined_query = router.next_query(gen, iteration, max_iterations)
        if refined_query is None:
            step["action"] = "stop"
            break

        # Сжать накопленные cited перед следующим кругом (работа маршрутизатора),
        # чтобы многократный ретрай не раздувал контекст.
        if compress_fn is not None and cited_docs:
            rewritten = compress_fn(current_query, router.cite_summary(cited_docs))
            shrunk = router.apply_rewrites(cited_docs, rewritten)
            if shrunk:  # если компрессор вернул пусто — оставляем прежние cited
                cited_docs = shrunk
                compressed_flag = True
                step["action"] = "retry+compress"
            else:
                step["action"] = "retry"
        else:
            step["action"] = "retry"

        current_query = refined_query

    # Итерации исчерпаны: отказ берём из answer генератора (см. _refused).
    return _refused(last_gen, cited_docs, retrieved_raw, compressed_flag, iteration, trace)


def _run_answer_agentic(
    query: str,
    retrieve_fn: RetrieveFn,
    generate_fn: GenerateFn,
    route_fn: RouteFn,
    *,
    max_iterations: int,
    top_k: int,
    mode: str,
) -> AnswerResponse:
    """Ретрай с маршрутизатором-агентом. Генератор всегда отвечает на главный запрос,
    цикл останавливается, как только ответ полон (answerable и пустой missing). Иначе
    маршрутизатор (один вызов модели) оценивает собранное, называет пробел,
    формулирует подзапрос и сжимает процитированные отрывки. На следующую итерацию
    переносятся только сжатые, поэтому свежих = top_k - |cited|.
    """
    seen_chunk_ids: Set[str] = set()
    cited_docs: List[Dict] = []
    retrieved_raw: List[Dict] = []
    prior_evidence = ""
    subqueries: List[str] = []
    retrieve_query = query
    last_gen = AnswerGeneration(answerable=False)
    context_docs: List[Dict] = []
    trace: List[Dict] = []
    iteration = 0
    compressed_flag = False

    for iteration in range(1, max_iterations + 1):
        budget = max(1, top_k - len(cited_docs))
        fresh = _retrieve_fresh(retrieve_fn, retrieve_query, budget, seen_chunk_ids, retrieved_raw)
        context_docs = cited_docs + fresh

        # Генератор ВСЕГДА отвечает на главный запрос (subquery — только для retrieve).
        gen = generate_fn(
            query,
            [doc["page_content"] for doc in context_docs],
            prior_evidence=prior_evidence,
            prior_missing="",
        )
        last_gen = gen
        step = {
            "iteration": iteration,
            "in_query": retrieve_query,
            "in_context_chunks": len(context_docs),
            "in_fresh_retrieved": len(fresh),
            "out_answerable": gen.answerable,
            "out_missing": gen.missing,
            "out_answer": gen.answer,
            "action": None,
        }
        trace.append(step)
        prior_evidence = gen.evidence_assessment

        # Стоп по уверенности генератора: он сам сообщил о полном ответе. Маршрутизатор
        # не переоценивает (он склонен завышать пробел и ломать готовый ответ).
        if gen.answerable and not (gen.missing or "").strip():
            step["action"] = "answer"
            return _answered(gen, context_docs, retrieved_raw, compressed_flag, iteration, trace, mode)

        if iteration >= max_iterations:
            step["action"] = "stop"
            break

        # Маршрутизатор-агент: видит весь контекст с метками CITE/SKIP и решает сам.
        decision = route_fn(
            query,
            prior_evidence,
            router.cite_summary(context_docs, gen.used_excerpts),
            subqueries,
            iteration,
            max_iterations,
        )
        if decision.done or not (decision.subquery or "").strip():
            step["action"] = "stop"
            cited_docs = context_docs
            break
        # Переносим только сжатое маршрутизатором (SKIP-дубли он исключил).
        rewritten = [rc.model_dump() for rc in decision.compressed_chunks]
        carried = router.apply_rewrites(context_docs, rewritten)
        cited_docs = carried if carried else context_docs
        compressed_flag = compressed_flag or bool(carried)
        subqueries.append(decision.subquery)
        retrieve_query = decision.subquery
        step["action"] = "retry"

    return _refused(last_gen, cited_docs or context_docs, retrieved_raw, compressed_flag, iteration, trace)


def _as_payload(docs: List[Dict]) -> List[Dict]:
    return [
        {
            "page_content": d["page_content"],
            "metadata": d.get("metadata", {}),
            "score": d.get("score"),
        }
        for d in docs
    ]
