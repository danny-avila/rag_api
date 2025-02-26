from fastapi import APIRouter, HTTPException
from app.db.psql import PSQLDatabase

router = APIRouter()

@router.get("/test/check_index")
async def check_file_id_index(table_name: str, column_name: str):
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT 1
                FROM pg_indexes
                WHERE tablename = $1 
                AND indexdef LIKE '%' || $2 || '%'
            );
        """, table_name, column_name)
    if exists:
        return {"message": f"Index on {column_name} exists in the table {table_name}."}
    else:
        raise HTTPException(status_code=404, detail=f"No index on {column_name} found in the table {table_name}.")

@router.get("/db/tables")
async def get_table_names(schema: str = "public"):
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        table_names = await conn.fetch("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = $1
        """, schema)
    tables = [record['table_name'] for record in table_names]
    return {"schema": schema, "tables": tables}

@router.get("/db/tables/columns")
async def get_table_columns(table_name: str, schema: str = "public"):
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        columns = await conn.fetch("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = $1 AND table_name = $2
            ORDER BY ordinal_position;
        """, schema, table_name)
    column_names = [col['column_name'] for col in columns]
    return {"table_name": table_name, "columns": column_names}

@router.get("/records/all")
async def get_all_records(table_name: str):
    if table_name not in ["langchain_pg_collection", "langchain_pg_embedding"]:
        raise HTTPException(status_code=400, detail="Invalid table name")
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        records = await conn.fetch(f"SELECT * FROM {table_name};")
    records_json = [dict(record) for record in records]
    return records_json

@router.get("/records")
async def get_records_filtered_by_custom_id(custom_id: str, table_name: str = "langchain_pg_embedding"):
    if table_name not in ["langchain_pg_collection", "langchain_pg_embedding"]:
        raise HTTPException(status_code=400, detail="Invalid table name")
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        query = f"SELECT * FROM {table_name} WHERE custom_id=$1;"
        records = await conn.fetch(query, custom_id)
    records_json = [dict(record) for record in records]
    return records_json