# app/answer/generator.py
"""Генератор: строит ответ только из извлечённых отрывков.

Три режима:
- "free":   сырой текст, без схемы (baseline-ячейка).
- "schema": размечает CITE, заполняет evidence_assessment/answerable/missing,
            answer оставляет пустым (заполняется снаружи).
- "answer": та же схема, плюс пишет обоснованный ответ.

route() — вызов маршрутизатора-агента (добор по подзапросу). compress() — отдельное
сжатие процитированных отрывков для простого ретрая.
"""
from pathlib import Path
from typing import Dict, List, Optional

from langchain_openai import ChatOpenAI

from app.answer import config
from app.answer.schema import (
    AnswerGeneration,
    CompressionResult,
    RouterDecision,
    UsedExcerpt,
)

_EXCERPT_TEMPLATE = "[{index}] {text}"
_PROMPT_DIR = Path(__file__).parent / "prompts"
_PROMPT_BY_MODE = {
    "free": "free_system.txt",
    "schema": "schema_system.txt",
    "answer": "answer_system.txt",
}


def _load_prompt(name: str) -> str:
    base = Path(config.RAG_ANSWER_PROMPT_DIR) if config.RAG_ANSWER_PROMPT_DIR else _PROMPT_DIR
    return (base / name).read_text(encoding="utf-8")


def _build_user_message(
    query: str,
    excerpts: List[str],
    prior_evidence: str = "",
    prior_missing: str = "",
) -> str:
    numbered = "\n\n".join(
        _EXCERPT_TEMPLATE.format(index=i, text=text)
        for i, text in enumerate(excerpts)
    )
    parts = [f"Вопрос:\n{query}"]
    if prior_evidence:
        parts.append(f"Собранные доказательства:\n{prior_evidence}")
    if prior_missing:
        parts.append(f"Чего не хватало в прошлых раундах:\n{prior_missing}")
    parts.append(f"Отрывки:\n{numbered}")
    return "\n\n".join(parts)


def get_chat_model() -> ChatOpenAI:
    """Собрать OpenAI-совместимую модель из настроек окружения."""
    return ChatOpenAI(
        model=config.RAG_ANSWER_MODEL,
        base_url=config.RAG_ANSWER_BASE_URL,
        api_key=config.RAG_ANSWER_API_KEY,
        temperature=config.RAG_ANSWER_TEMPERATURE,
        max_tokens=config.RAG_ANSWER_MAX_TOKENS,
    )


def _structured(system_prompt: str, user_msg: str, model_cls):
    """Один структурный вызов модели: system + human -> экземпляр model_cls."""
    model = get_chat_model().with_structured_output(model_cls)
    return model.invoke([("system", system_prompt), ("human", user_msg)])


def generate(
    query: str,
    excerpts: List[str],
    *,
    mode: str = "schema",
    generate_answer: Optional[bool] = None,
    prior_evidence: str = "",
    prior_missing: str = "",
) -> AnswerGeneration:
    """Одна генерация по отрывкам в заданном режиме.

    generate_answer — для обратной совместимости: True -> mode="answer",
    False -> mode="schema". Явный mode имеет приоритет.
    """
    if generate_answer is not None:
        mode = "answer" if generate_answer else "schema"
    if mode not in _PROMPT_BY_MODE:
        raise ValueError(f"unknown generation mode: {mode!r}")

    system_prompt = _load_prompt(_PROMPT_BY_MODE[mode])
    user_msg = _build_user_message(query, excerpts, prior_evidence, prior_missing)

    if mode == "free":
        # Без структуры: сырой текст, все отрывки помечаются CITE (схема сборки
        # контекста в baseline-ячейке не используется).
        raw = get_chat_model().invoke([("system", system_prompt), ("human", user_msg)])
        text = getattr(raw, "content", raw) or ""
        return AnswerGeneration(
            used_excerpts=[
                UsedExcerpt(excerpt_index=i, action="CITE")
                for i in range(len(excerpts))
            ],
            answerable=True,
            answer=str(text).strip(),
        )

    return _structured(system_prompt, user_msg, AnswerGeneration)


def compress(query: str, cite_summary: str) -> List[Dict]:
    """Переписать процитированные отрывки в компактные спаны по вопросу.

    Возвращает список {"id", "text"}, маршрутизатор применяет их по id.
    """
    user_msg = f"Вопрос:\n{query}\n\nПроцитированные отрывки:\n{cite_summary}"
    result = _structured(_load_prompt("router_compress.txt"), user_msg, CompressionResult)
    return [rc.model_dump() for rc in result.rewritten_chunks]


def route(
    main_query: str,
    evidence: str,
    cite_summary: str,
    prev_subqueries: List[str],
    iteration: int,
    max_iterations: int,
) -> RouterDecision:
    """Маршрутизатор-агент: один вызов модели в порядке chain-of-thought — оценить
    собранное, назвать пробел (missing), сформулировать подзапрос, решить `done` и
    сжать CITED (ПОСЛЕДНИМ). Видит метки CITE/SKIP и сам исключает SKIP-дубли из
    compressed_chunks.
    """
    prev = "\n".join(f"- {s}" for s in prev_subqueries) or "(пока нет)"
    user_msg = (
        f"Основной вопрос:\n{main_query}\n\n"
        f"Собранные доказательства:\n{evidence}\n\n"
        f"Прошлые подзапросы:\n{prev}\n\n"
        f"Процитированные отрывки (id [CITE/SKIP] text):\n{cite_summary}\n\n"
        f"Сейчас итерация {iteration} из {max_iterations}."
    )
    return _structured(_load_prompt("router_system.txt"), user_msg, RouterDecision)
