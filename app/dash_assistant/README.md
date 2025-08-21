# Dash Assistant Setup Guide

## Quick Start

This guide provides step-by-step instructions to set up and run the Dash Assistant system.

### Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for local development)
- PostgreSQL with pgvector extension

### Step-by-Step Setup

#### 1. Start Database

Start the PostgreSQL database with pgvector support:

```bash
# Start only the database
docker compose -f db-compose.yaml up -d

# Or start the full stack (database + API)
docker compose up -d db
```

Wait for the database to be ready (health check will confirm):
```bash
# Check database status
docker compose ps
```

#### 2. Run Migrations

Apply database schema and create necessary tables:

```bash
# Run migrations
python -m app.dash_assistant.migrate

# Optional: Check migration status
python -m app.dash_assistant.migrate --list

# Optional: Dry run to see what would be executed
python -m app.dash_assistant.migrate --dry-run
```

#### 3. Data Ingestion

Load your dashboard and chart data:

```bash
# Example ingestion using test fixtures
python -c "
import asyncio
from app.dash_assistant.ingestion.index_jobs import IndexJob

async def main():
    job = IndexJob()
    await job.ingest_from_files(
        dashboards_csv='tests/fixtures/superset/dashboards.csv',
        charts_csv='tests/fixtures/superset/charts.csv',
        enrichment_yaml='tests/fixtures/superset/enrichment.yaml',
        md_dir='tests/fixtures/superset/md/'
    )

asyncio.run(main())
"
```

Or use the REST API:
```bash
curl -X POST "http://localhost:8000/dash/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "dashboards_csv": "tests/fixtures/superset/dashboards.csv",
    "charts_csv": "tests/fixtures/superset/charts.csv", 
    "enrichment_yaml": "tests/fixtures/superset/enrichment.yaml",
    "md_dir": "tests/fixtures/superset/md/"
  }'
```

#### 4. Generate Embeddings

Create vector embeddings for semantic search:

```bash
# Generate embeddings for all content
python -c "
import asyncio
from app.dash_assistant.ingestion.index_jobs import IndexJob

async def main():
    job = IndexJob()
    await job.create_embeddings_for_all()

asyncio.run(main())
"
```

#### 5. Optimize Database (Important!)

After mass loading, optimize the database for better performance:

```bash
# Connect to database and run optimization
psql -h localhost -U postgres -d rag_api -c "SELECT optimize_after_mass_loading();"

# Or using Python
python -c "
import asyncio
from app.dash_assistant.db import DashAssistantDB

async def main():
    await DashAssistantDB.execute_query('SELECT optimize_after_mass_loading()')
    await DashAssistantDB.close_pool()

asyncio.run(main())
"
```

#### 6. Start API Server

Start the FastAPI server:

```bash
# Local development
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Or using Docker
docker compose up fastapi
```

### Verification

Test that everything is working:

```bash
# Health check
curl http://localhost:8000/dash/health

# Statistics
curl http://localhost:8000/dash/stats

# Test query
curl -X POST "http://localhost:8000/dash/query" \
  -H "Content-Type: application/json" \
  -d '{
    "q": "user retention",
    "top_k": 5
  }'
```

## Configuration

### Environment Variables

The system uses these key environment variables (see `dash_assistant.env.example`):

```bash
# Embeddings Configuration
EMBEDDINGS_PROVIDER=MOCK          # MOCK, OPENAI, HUGGINGFACE
EMBEDDINGS_DIMENSION=3072         # Vector dimension

# Retrieval Configuration  
RRF_K=60                         # Reciprocal Rank Fusion parameter
DEFAULT_TOPK=5                   # Default number of results

# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=rag_api
POSTGRES_USER=postgres
POSTGRES_PASSWORD=password

# API Keys (if using OpenAI)
OPENAI_API_KEY=your-api-key-here
```

### Development vs Production

**Development (Default):**
- Uses MOCK embeddings provider for deterministic testing
- 3072-dimensional vectors for comprehensive testing
- Local PostgreSQL database

**Production:**
- Set `EMBEDDINGS_PROVIDER=OPENAI` 
- Set `EMBEDDINGS_DIMENSION=1536` (OpenAI standard)
- Provide `OPENAI_API_KEY`
- Configure production database credentials

## Troubleshooting

### Common Issues

1. **Database Connection Failed**
   ```bash
   # Check if database is running
   docker compose ps
   
   # Check database logs
   docker compose logs db
   ```

2. **Migration Fails**
   ```bash
   # Check migration status
   python -m app.dash_assistant.migrate --list
   
   # Rollback last migration if needed
   python -m app.dash_assistant.migrate --rollback
   ```

3. **Embeddings Generation Slow**
   ```bash
   # Use batch processing for large datasets
   python -c "
   import asyncio
   from app.dash_assistant.ingestion.index_jobs import IndexJob
   
   async def main():
       job = IndexJob()
       await job.create_embeddings_for_all(batch_size=100, only_missing=True)
   
   asyncio.run(main())
   "
   ```

4. **Poor Query Performance**
   ```bash
   # Run database optimization
   psql -h localhost -U postgres -d rag_api -c "SELECT optimize_after_mass_loading();"
   ```

### Performance Optimization

After loading large datasets:

1. **Always run ANALYZE** - Critical for PostgreSQL query planner
2. **Reindex vector indices** - Improves HNSW/IVFFlat performance  
3. **Monitor query performance** - Use `EXPLAIN ANALYZE` for slow queries
4. **Consider index tuning** - Adjust HNSW parameters for your data size

### Monitoring

```bash
# Check system health
curl http://localhost:8000/dash/health

# View database statistics
curl http://localhost:8000/dash/stats

# Monitor query performance in logs
tail -f logs/dash_assistant.log
```

## Architecture

- **Ingestion Pipeline**: CSV → Enrichment → Chunking → Embeddings
- **Search System**: Multi-signal retrieval (FTS + Vector + Trigram) with RRF
- **Database**: PostgreSQL with pgvector, optimized indices
- **API**: FastAPI with async support, structured responses

For more details, see the main project documentation.