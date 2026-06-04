# app/answer/router.py
"""Маршрутизатор: дополняет контекст, когда его недостаточно.

Все функции здесь чистые (без вызова модели). Вызов модели для сжатия внедряется в
pipeline как callable из routes.py, поэтому модуль тестируется без БД и без модели.

Назначение:
- next_query — читает answerable / missing, решает, делать ли ретрай и с каким
  уточнённым запросом.
- cite_summary — собирает процитированные отрывки для промпта сжатия.
- apply_rewrites — применяет сжатые спаны обратно на процитированные отрывки.
"""
import hashlib
from typing import Dict, List, Optional

from app.answer.schema import AnswerGeneration


def _digest(page_content: str) -> str:
    """Идентификатор по содержимому, совпадает с pipeline._chunk_id и digest при ingest."""
    return hashlib.md5(page_content.encode()).hexdigest()


def next_query(
    generation: AnswerGeneration, iteration: int, max_iterations: int
) -> Optional[str]:
    """Вернуть уточнённый запрос для ретрая или None, чтобы остановиться.

    Маршрутизатор дополняет контекст только когда генератор сообщил о пробеле. Читает
    `answerable` и `missing` из схемы и берёт `router_feedback` как поисковый запрос
    на этот пробел.
    """
    if generation.answerable:
        return None
    if iteration >= max_iterations:
        # Последний круг достигнут, новый поиск уже не пригодится для ответа.
        return None
    refined = generation.router_feedback.strip()
    if not refined:
        # Пробел есть (generation.missing), но конкретного запроса нет, ретрай лишь
        # повторил бы тот же поиск. Останавливаемся.
        return None
    return refined


def cite_summary(cited_docs: List[Dict], used_excerpts=None) -> str:
    """Собрать процитированные отрывки в строки 'id=<digest> <text>' для сжатия.

    Если передан used_excerpts, к каждому отрывку добавляется метка [CITE]/[SKIP] от
    генератора — маршрутизатор видит дубли (SKIP) и сам не переносит их в compressed_chunks.
    id — это digest, чтобы apply_rewrites сопоставлял по нему.
    """
    action_by_idx = {ue.excerpt_index: ue.action for ue in (used_excerpts or [])}
    lines = []
    for i, d in enumerate(cited_docs):
        digest = (d.get("metadata") or {}).get("digest") or _digest(d["page_content"])
        tag = f"[{action_by_idx.get(i, 'CITE')}] " if used_excerpts is not None else ""
        lines.append(f"id={digest} {tag}{d['page_content']}")
    return "\n".join(lines)


def apply_rewrites(cited_docs: List[Dict], rewritten: List[Dict]) -> List[Dict]:
    """Заменить текст процитированных отрывков сжатыми спанами по digest-id.

    Отрывки, которые компрессор опустил, отбрасываются (сочтены нерелевантными).
    Возвращает новый список, у оставшихся metadata.compressed=True. Чистая функция.
    """
    by_id = {
        (d.get("metadata") or {}).get("digest") or _digest(d["page_content"]): d
        for d in cited_docs
    }
    out: List[Dict] = []
    for item in rewritten:
        d = by_id.get(item.get("id"))
        if d is None or "text" not in item:
            continue
        md = dict(d.get("metadata") or {})
        md["compressed"] = True
        out.append({"page_content": item["text"], "metadata": md})
    return out
