# app/routes/pgvector_routes.py
from fastapi import APIRouter, HTTPException
from app.services.database import PSQLDatabase

router = APIRouter()


async def check_index_exists(table_name: str, column_name: str) -> bool:
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        result = await conn.fetch(
            """
            SELECT EXISTS (
                SELECT 1
                FROM pg_indexes
                WHERE tablename = $1 
                AND indexdef LIKE '%' || $2 || '%'
            );
            """,
            table_name,
            column_name,
        )
    return result[0]['exists']


@router.get("/test/check_index")
async def check_file_id_index(table_name: str, column_name: str):
    if await check_index_exists(table_name, column_name):
        return {"message": f"Index on {column_name} exists in the table {table_name}."}
    else:
        return HTTPException(status_code=404, detail=f"No index on {column_name} found in the table {table_name}.")


@router.get("/db/tables")
async def get_table_names(schema: str = "public"):
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        table_names = await conn.fetch(
            """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = $1
            """,
            schema,
        )
    # Extract table names from records
    tables = [record['table_name'] for record in table_names]
    return {"schema": schema, "tables": tables}


@router.get("/db/tables/columns")
async def get_table_columns(table_name: str, schema: str = "public"):
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        columns = await conn.fetch(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = $1 AND table_name = $2
            ORDER BY ordinal_position;
            """,
            schema, table_name,
        )
    column_names = [col['column_name'] for col in columns]
    return {"table_name": table_name, "columns": column_names}


@router.get("/records/all")
async def get_all_records(table_name: str):
    # Validate that the table name is one of the expected ones to prevent SQL injection
    if table_name not in ["langchain_pg_collection", "langchain_pg_embedding"]:
        raise HTTPException(status_code=400, detail="Invalid table name")

    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        # Use SQLAlchemy core or raw SQL queries to fetch all records
        records = await conn.fetch(f"SELECT * FROM {table_name};")

    # Convert records to JSON serializable format, assuming records can be directly serialized
    records_json = [dict(record) for record in records]

    return records_json


@router.get("/records")
async def get_records_filtered_by_custom_id(custom_id: str, table_name: str = "langchain_pg_embedding"):
    # Validate that the table name is one of the expected ones to prevent SQL injection
    if table_name not in ["langchain_pg_collection", "langchain_pg_embedding"]:
        raise HTTPException(status_code=400, detail="Invalid table name")

    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        # Use parameterized queries to prevent SQL Injection
        query = f"SELECT * FROM {table_name} WHERE custom_id=$1;"
        records = await conn.fetch(query, custom_id)

    # Convert records to JSON serializable format, assuming the Record class has a dict method.
    records_json = [dict(record) for record in records]

    return records_json