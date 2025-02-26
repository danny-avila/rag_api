import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import (
    debug_mode,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    PDF_EXTRACT_IMAGES,
    RAG_HOST,
    RAG_PORT,
    VECTOR_DB_TYPE,
)
from app.db.psql import PSQLDatabase, ensure_custom_id_index_on_embedding, ensure_jsonb_metadata
from app.middleware import security_middleware, LogMiddleware

# Load environment variables
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic: initialize the DB pool and run migrations
    if VECTOR_DB_TYPE.value == "pgvector":
        await PSQLDatabase.get_pool()
        # Run the index creation and metadata migration
        await ensure_custom_id_index_on_embedding()
        await ensure_jsonb_metadata()
    yield

app = FastAPI(lifespan=lifespan, debug=debug_mode)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middlewares
app.add_middleware(LogMiddleware)
app.middleware("http")(security_middleware)

# Set state variables (e.g. for text splitting)
app.state.CHUNK_SIZE = CHUNK_SIZE
app.state.CHUNK_OVERLAP = CHUNK_OVERLAP
app.state.PDF_EXTRACT_IMAGES = PDF_EXTRACT_IMAGES

# Include API routes
from app.routes import documents, embed, query, pgvector, health
app.include_router(health.router)
app.include_router(documents.router)
app.include_router(embed.router)
app.include_router(query.router)
app.include_router(pgvector.router)

if __name__ == "__main__":
    uvicorn.run(app, host=RAG_HOST, port=RAG_PORT, log_config=None)