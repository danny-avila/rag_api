# app/services/summary_store.py
from app.config import logger
from app.services.database import PSQLDatabase


async def upsert_file_summary(
    file_id: str, user_id: str, summary: str, chunk_count: int
) -> None:
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO file_summaries (file_id, user_id, summary, chunk_count)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (file_id, user_id)
            DO UPDATE SET summary = EXCLUDED.summary,
                         chunk_count = EXCLUDED.chunk_count,
                         updated_at = now()
            """,
            file_id,
            user_id,
            summary,
            chunk_count,
        )


async def get_summaries_by_user(user_id: str) -> list[dict]:
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT file_id, summary, chunk_count
            FROM file_summaries
            WHERE user_id = $1
            """,
            user_id,
        )
        return [dict(row) for row in rows]


async def delete_summaries_by_file_ids(
    file_ids: list[str], user_id: str = None
) -> None:
    if not file_ids:
        return
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        if user_id:
            await conn.execute(
                """
                DELETE FROM file_summaries
                WHERE file_id = ANY($1::text[]) AND user_id = $2
                """,
                file_ids,
                user_id,
            )
        else:
            await conn.execute(
                """
                DELETE FROM file_summaries
                WHERE file_id = ANY($1::text[])
                """,
                file_ids,
            )
