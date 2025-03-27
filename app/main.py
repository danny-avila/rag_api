# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.config import debug_mode, RAG_HOST, RAG_PORT, CHUNK_SIZE, CHUNK_OVERLAP, PDF_EXTRACT_IMAGES, VECTOR_DB_TYPE
from app.middleware import security_middleware
from app.services.database import PSQLDatabase, ensure_custom_id_index_on_embedding
from app.routes import document_routes, pgvector_routes

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic goes here
    if VECTOR_DB_TYPE == "pgvector":
        await PSQLDatabase.get_pool()  # Initialize the pool
        await ensure_custom_id_index_on_embedding()

    yield

app = FastAPI(lifespan=lifespan, debug=debug_mode)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.middleware("http")(security_middleware)

# Set state variables for use in routes
app.state.CHUNK_SIZE = CHUNK_SIZE
app.state.CHUNK_OVERLAP = CHUNK_OVERLAP
app.state.PDF_EXTRACT_IMAGES = PDF_EXTRACT_IMAGES

# Include routers
app.include_router(document_routes.router)
app.include_router(pgvector_routes.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=RAG_HOST, port=RAG_PORT, log_config=None)
