# app/answer/config.py
"""Самодостаточный конфиг слоя ответа.

Читает переменные RAG_ANSWER_* (через get_env_variable оригинального rag_api).
Лежит внутри app/answer/, чтобы функция владела своими настройками, а оригинальный
app/config.py не менялся. Выключено, пока RAG_ANSWER_ENABLED не истинно.
"""
from app.config import get_env_variable

RAG_ANSWER_ENABLED = get_env_variable("RAG_ANSWER_ENABLED", "False").lower() == "true"

# Обязательны при включённой функции (проверяется в require_answer_config()).
RAG_ANSWER_MODEL = get_env_variable("RAG_ANSWER_MODEL", None)
RAG_ANSWER_BASE_URL = get_env_variable("RAG_ANSWER_BASE_URL", None)
RAG_ANSWER_API_KEY = get_env_variable("RAG_ANSWER_API_KEY", None)

RAG_ANSWER_TEMPERATURE = float(get_env_variable("RAG_ANSWER_TEMPERATURE", "0.0"))
RAG_ANSWER_MAX_TOKENS = int(get_env_variable("RAG_ANSWER_MAX_TOKENS", "1024"))
RAG_ANSWER_MAX_ITERATIONS = int(get_env_variable("RAG_ANSWER_MAX_ITERATIONS", "1"))
RAG_ANSWER_GENERATE = get_env_variable("RAG_ANSWER_GENERATE", "False").lower() == "true"
# Режим генерации: "free" (сырой текст, без схемы), "schema" (CITE + answerable, без
# ответа), "answer" (схема + пишет ответ). По умолчанию из RAG_ANSWER_GENERATE.
RAG_ANSWER_MODE = get_env_variable(
    "RAG_ANSWER_MODE", "answer" if RAG_ANSWER_GENERATE else "schema"
).lower()
# Сжатие на стороне маршрутизатора: переписать процитированные отрывки в компактные
# факты перед накоплением между итерациями, чтобы контекст не раздувался.
RAG_ANSWER_COMPRESS = get_env_variable("RAG_ANSWER_COMPRESS", "False").lower() == "true"
# Маршрутизатор-агент: вместо детерминированного router.next_query маршрутизатор
# становится вызовом модели (chain-of-thought). Он называет пробел (missing),
# формулирует подзапрос и сжимает процитированные отрывки. Цикл также
# останавливается, как только генератор уверен в полноте ответа.
RAG_ANSWER_ROUTER_AGENT = (
    get_env_variable("RAG_ANSWER_ROUTER_AGENT", "False").lower() == "true"
)
RAG_ANSWER_TOP_K = int(get_env_variable("RAG_ANSWER_TOP_K", "4"))
# None -> генератор использует свою папку prompts/.
RAG_ANSWER_PROMPT_DIR = get_env_variable("RAG_ANSWER_PROMPT_DIR", None)


def require_answer_config() -> None:
    """Ошибка, если функция включена, но обязательные настройки не заданы."""
    missing = [
        name
        for name, value in (
            ("RAG_ANSWER_MODEL", RAG_ANSWER_MODEL),
            ("RAG_ANSWER_BASE_URL", RAG_ANSWER_BASE_URL),
            ("RAG_ANSWER_API_KEY", RAG_ANSWER_API_KEY),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            "RAG_ANSWER_ENABLED истинно, но не заданы переменные: "
            + ", ".join(missing)
        )
