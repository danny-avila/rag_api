# app/answer/schema.py
"""Pydantic-модели слоя ответа (самодостаточный модуль).

Лежат внутри app/answer/, а не в app/models.py, чтобы вся функция была в одном месте
и оригинальные файлы rag_api не менялись (чистый git pull).

Схема сборки контекста: генератор метит каждый отрывок CITE или SKIP, объявляет
answerable и при невозможности ответа предлагает уточнённый запрос.
"""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class AnswerRequestBody(BaseModel):
    query: str
    file_id: str
    k: Optional[int] = None
    entity_id: Optional[str] = None
    max_iterations: Optional[int] = None
    mode: Optional[str] = None  # free | schema | answer; переопределяет RAG_ANSWER_MODE на запрос


class UsedExcerpt(BaseModel):
    """Один извлечённый отрывок с решением оставить или отбросить."""

    excerpt_index: int = Field(
        ge=0, description="Индекс отрывка (с нуля) в списке, показанном модели."
    )
    action: Literal["CITE", "SKIP"] = Field(
        description="CITE, если отрывок поддерживает ответ, иначе SKIP."
    )
    reason: str = Field(default="", description="Краткое обоснование решения.")


class AnswerGeneration(BaseModel):
    """Структурированный вывод одной итерации. Порядок полей — это цепочка рассуждения.
    Сначала дословно процитировать доказательства, затем решить об ответимости, затем
    ответ, и ПОСЛЕДНИМ отметить использованные отрывки (CITE как план обоснования после
    рассуждения, а не до)."""

    evidence_assessment: str = Field(
        default="",
        description="Накопленные доказательства ДОСЛОВНЫМИ цитатами. Повтори прошлые "
        "доказательства и добавь точные строки источника (числа, названия, единицы "
        "дословно, с подписью строки). Цитируй строку таблицы, не пересказывай.",
    )
    answerable: bool = Field(
        description="True, если хотя бы часть вопроса подтверждается доказательствами."
    )
    missing: str = Field(
        default="",
        description="Чего ещё НЕ хватает для полного ответа (пусто, если покрыто полностью).",
    )
    answer: str = Field(
        default="",
        description="Ответ на основе процитированных доказательств. Частичный допустим, "
        "явный отказ, если ничего релевантного не найдено.",
    )
    router_feedback: str = Field(
        default="",
        description="Уточнённый поисковый запрос на `missing` (пусто, если ничего не недостаёт).",
    )
    used_excerpts: List[UsedExcerpt] = Field(
        default_factory=list,
        description="Самопроверка как цепочка рассуждения (НЕ фильтр контекста), "
        "заполняется ПОСЛЕДНИМ. Пройди по каждому отрывку и обоснуй. По умолчанию все CITE, "
        "SKIP метит только дубль уже процитированного факта. По Yelmagambetov, разметка "
        "каждого отрывка заставляет не-рассуждающую модель думать и улучшает обоснованность.",
    )


class AnswerResponse(BaseModel):
    """Итоговый ответ эндпоинта /answer."""

    answer: str
    answerable: bool
    evidence_assessment: str = ""
    used_excerpts: List[UsedExcerpt] = Field(default_factory=list)
    cited: List[dict] = Field(
        default_factory=list,
        description="Отрывки, помеченные CITE: {page_content, metadata}.",
    )
    retrieved: List[dict] = Field(
        default_factory=list,
        description="Сырые извлечённые отрывки по всем итерациям, ДО сжатия "
        "маршрутизатором. Для recall используй это, а не `cited` (он переписан).",
    )
    trace: List[dict] = Field(
        default_factory=list,
        description="Вход/выход компонентов по итерациям: запрос, число извлечённых, "
        "поля генератора (answerable/evidence/missing/cite), действие маршрутизатора.",
    )
    iterations: int = Field(
        ge=1, description="Сколько кругов retrieve+генерация выполнено."
    )
    missing: str = Field(
        default="", description="Итоговые пробелы, когда answerable=false."
    )
    compressed: bool = Field(
        default=False,
        description="True, если маршрутизатор переписал процитированные спаны в сжатый текст.",
    )
    refusal_reason: str = Field(
        default="",
        description="Почему в ответе отказано, когда answerable=false.",
    )


class RewrittenChunk(BaseModel):
    """Один процитированный отрывок, сжатый до релевантного вопросу спана."""

    id: str = Field(description="Digest-id переписываемого процитированного отрывка.")
    text: str = Field(description="Сжатый текст: релевантные факты и минимальный контекст.")


class CompressionResult(BaseModel):
    """Структурированный вывод вызова сжатия маршрутизатора."""

    rewritten_chunks: List[RewrittenChunk] = Field(default_factory=list)


class RouterDecision(BaseModel):
    """Решение маршрутизатора-агента. Порядок полей — это цепочка рассуждения. Сначала
    оценить собранное, затем назвать пробел, сформулировать подзапрос, решить об
    остановке и ПОСЛЕДНИМ сжать процитированные отрывки (действие после рассуждения)."""

    evidence_review: str = Field(
        default="",
        description="Что собранные доказательства уже покрывают по ГЛАВНОМУ вопросу "
        "(рассуждение первым). Опираться только на переданное, ничего не выдумывать.",
    )
    missing: str = Field(
        default="",
        description="Чего ещё НЕ хватает для полного ответа на главный вопрос "
        "(пусто, если покрыто полностью).",
    )
    subquery: str = Field(
        default="",
        description="Новый поисковый подзапрос на `missing`, сформулированный иначе, "
        "чем прошлые подзапросы и главный вопрос (пусто, если ничего не недостаёт).",
    )
    done: bool = Field(
        description="True, если главный вопрос полностью покрыт и поиск останавливается."
    )
    compressed_chunks: List[RewrittenChunk] = Field(
        default_factory=list,
        description="Процитированные отрывки, переписанные в компактные спаны, "
        "заполняется ПОСЛЕДНИМ. Числа, единицы и названия строк — дословно.",
    )
