# ID-based RAG FastAPI

## Overview
This project integrates Langchain with FastAPI in an Asynchronous, Scalable manner, providing a framework for document indexing and retrieval, using PostgreSQL/pgvector.

Files are organized into embeddings by `file_id`. The primary use case is for integration with [LibreChat](https://librechat.ai), but this simple API can be used for any ID-based use case.

The main reason to use the ID approach is to work with embeddings on a file-level. This makes for targeted queries when combined with file metadata stored in a database, such as is done by LibreChat.

The API will evolve over time to employ different querying/re-ranking methods, embedding models, and vector stores.

## Features
- **Document Management**: Methods for adding, retrieving, and deleting documents.
- **Vector Store**: Utilizes Langchain's vector store for efficient document retrieval.
- **Asynchronous Support**: Offers async operations for enhanced performance.

## Setup

### Getting Started

- **Configure `.env` file based on [section below](#environment-variables)**
- **Setup pgvector database:**
  - Run an existing PSQL/PGVector setup, or,
  - Docker: `docker compose up` (also starts RAG API)
    - or, use docker just for DB: `docker compose -f ./db-compose.yaml up`
- **Run API**:
  - Docker: `docker compose up` (also starts PSQL/pgvector)
    - or, use docker just for RAG API: `docker compose -f ./api-compose.yaml up`
  - Local:
    - Make sure to setup `DB_HOST` to the correct database hostname
    - Run the following commands (preferably in a [virtual environment](https://realpython.com/python-virtual-environments-a-primer/))
```bash
pip install -r requirements.txt
uvicorn main:app
```

### Clean Install (Local Development)

To do a clean reinstall of all dependencies (e.g., after updating `requirements.txt`):

```bash
# Remove existing virtual environment and recreate it
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

For the lite version (without sentence_transformers/huggingface):

```bash
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.lite.txt
```

For Docker, rebuild without cache:

```bash
docker compose build --no-cache
```

### Environment Variables

Copy `.env.example` to `.env` and replace every `REPLACE_ME_*` placeholder. The
database credentials ship **no fallback defaults** — the service refuses to
start without them, and the compose files use `${VAR:?}` so `docker compose up`
fails loudly rather than bringing up a database whose credentials are public in
this repository:

- `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`: **required** when
  `VECTOR_DB_TYPE=pgvector`. `POSTGRES_PASSWORD` may be empty only when
  `POSTGRES_USE_UNIX_SOCKET=True`, where peer authentication carries no password.

The following environment variables are required to run the application:

- `RAG_OPENAI_API_KEY`: The API key for OpenAI API Embeddings (if using default settings).
    - Note: `OPENAI_API_KEY` will work but `RAG_OPENAI_API_KEY` will override it in order to not conflict with LibreChat setting.
- `RAG_OPENAI_BASEURL`: (Optional) The base URL for your OpenAI API Embeddings
- `RAG_OPENAI_PROXY`: (Optional) Proxy for OpenAI API Embeddings
    - Note: When using with LibreChat, you can also set `HTTP_PROXY` and `HTTPS_PROXY` environment variables in the `docker-compose.override.yml` file (see [Proxy Configuration](#proxy-configuration) section below)
- `VECTOR_DB_TYPE`: (Optional) select vector database type, default to `pgvector`.
- `POSTGRES_USE_UNIX_SOCKET`: (Optional) Set to "True" when connecting to the PostgreSQL database server with Unix Socket.
- `POSTGRES_DB`: The name of the PostgreSQL database, used when `VECTOR_DB_TYPE=pgvector`. Required, no default.
- `POSTGRES_USER`: The username for connecting to the PostgreSQL database. Required, no default.
- `POSTGRES_PASSWORD`: The password for connecting to the PostgreSQL database. Required, no default (may be empty only under `POSTGRES_USE_UNIX_SOCKET=True`).
- `DB_HOST`: (Optional) The hostname or IP address of the PostgreSQL database server.
- `DB_PORT`: (Optional) The port number of the PostgreSQL database server.
- `PGVECTOR_CREATE_EXTENSION`: (Optional) Set to "False" to skip the `CREATE EXTENSION IF NOT EXISTS vector` call on startup. Default is "True". Use this when the `vector` extension is already installed on a managed Postgres (e.g. RDS, Azure Database for PostgreSQL) and the application user is not a superuser.
- `PG_POOL_PRE_PING`: (Optional) Set to "False" to disable SQLAlchemy's pre-ping check. Default is "True". When enabled, the connection pool issues a lightweight `SELECT 1` before handing out a pooled connection, so stale connections dropped by a remote server or middlebox idle timeout are transparently replaced instead of surfacing as query errors. Recommended for any deployment that connects to a remote PostgreSQL instance (managed Postgres, connections that traverse a load balancer, etc.).
- `PG_POOL_RECYCLE`: (Optional) Maximum age in seconds of a pooled connection before it is recycled. Default is "-1" (disabled). Set to a positive value when the server enforces a hard idle or max-lifetime limit (e.g. "1800" for a 30-minute cap).
- `POSTGRES_SCHEMA`: (Optional) Prepend this schema to the Postgres `search_path` so langchain's pgvector tables live in (and are read from) it. Unset by default (uses the user's default schema, typically `public`). Useful when sharing a database with other services — create the schema out-of-band first (`CREATE SCHEMA IF NOT EXISTS <name>; GRANT USAGE, CREATE ON SCHEMA <name> TO <app_user>;`); the RAG API will not create it for you and fails fast at startup if the schema is missing. `public` is always appended to the resulting search path so the `vector` data type stays resolvable when the extension was installed there (the common case). Multiple schemas may be supplied as a comma-separated list (e.g. `myapp,extensions`) when the `vector` extension lives in a non-`public` schema.
- `PGVECTOR_CREATE_LEGACY_INDEXES`: (Optional) Set to "True" to create the legacy `custom_id` and `cmetadata->>'file_id'` indexes on startup. Default is "False".
- `PGVECTOR_MIGRATE_CMETADATA_JSONB`: (Optional) Set to "True" to migrate `langchain_pg_embedding.cmetadata` from JSON to JSONB on startup. Default is "False".
- `PGVECTOR_CREATE_CMETADATA_GIN_INDEX`: (Optional) Set to "True" to create the `cmetadata` JSONB GIN index on startup. Default is "False". The index is created only when `cmetadata` is already JSONB; for a legacy JSON column, also enable `PGVECTOR_MIGRATE_CMETADATA_JSONB` or the index step is skipped.
- `RAG_HOST`: (Optional) The hostname or IP address where the API server will run. Defaults to "0.0.0.0"
- `RAG_PORT`: (Optional) The port number where the API server will run. Defaults to port 8000.
- `JWT_SECRET`: (Optional) The secret key used for verifying JWT tokens for requests.
  - The secret is only used for verification. This basic approach assumes a signed JWT from elsewhere.
  - Omit to run API without requiring authentication

- `COLLECTION_NAME`: (Optional) The name of the collection in the vector store. Default value is "testcollection".
- `CHUNK_SIZE`: (Optional) The size of the chunks for text processing. Default value is "1500".
- `CHUNK_OVERLAP`: (Optional) The overlap between chunks during text processing. Default value is "100".
- `EMBEDDING_BATCH_SIZE`: (Optional) Number of document chunks to process per batch. Defaults to `500`; set to `0` to disable batching. Recommended value is `750` for `text-embedding-3-small`.
- `EMBEDDING_MAX_QUEUE_SIZE`: (Optional) Maximum number of batches to buffer in memory during async processing. Default value is "3".
- `PARALLEL_EXECUTION`: (Optional) Maximum number of async embedding/database insertion consumers to run per file when batching is enabled. Default value is "2".
- `RAG_DISTANCE_THRESHOLD`: (Optional, `VECTOR_DB_TYPE=pgvector` only) Drop results whose vector distance is greater than this value, after the top-`k` search. Unset by default (no filtering). Lower distance = more similar, so e.g. `0.5` keeps only hits with distance ≤ 0.5 and discards weaker matches. Useful for reducing downstream LLM token cost when the top-`k` call returns loosely-related chunks. Appropriate values depend on the embedding model and distance strategy — inspect your actual scores before choosing one. Ignored (with a startup warning) under `VECTOR_DB_TYPE=atlas-mongo`, because Atlas returns a similarity score (higher = better) with inverted semantics.
- `RAG_UPLOAD_DIR`: (Optional) The directory where uploaded files are stored. Default value is "./uploads/".
- `PDF_EXTRACT_IMAGES`: (Optional) A boolean value indicating whether to extract images from PDF files. Default value is "False".
- `DEBUG_RAG_API`: (Optional) Set to "True" to show more verbose logging output in the server console, and to enable postgresql database routes
- `DEBUG_PGVECTOR_QUERIES`: (Optional) Set to "True" to enable detailed PostgreSQL query logging for pgvector operations. Useful for debugging performance issues with vector database queries.
- `CONSOLE_JSON`: (Optional) Set to "True" to log as json for Cloud Logging aggregations
- `EMBEDDINGS_PROVIDER`: (Optional) either "openai", "bedrock", "azure", "huggingface", "huggingfacetei", "google_genai", "vertexai", or "ollama", where "huggingface" uses sentence_transformers; defaults to "openai"
- `EMBEDDINGS_MODEL`: (Optional) Set a valid embeddings model to use from the configured provider.
    - **Defaults**
    - openai: "text-embedding-3-small"
    - azure: "text-embedding-3-small" (will be used as your Azure Deployment)
    - huggingface: "sentence-transformers/all-MiniLM-L6-v2"
    - huggingfacetei: "http://huggingfacetei:3000". Hugging Face TEI uses model defined on TEI service launch.
    - vertexai: "gemini-embedding-001"
    - ollama: "nomic-embed-text"
    - bedrock: "amazon.titan-embed-text-v1"
    - google_genai: "gemini-embedding-001"
- `EMBEDDINGS_CHUNK_SIZE`: (Optional) The chunk size used by the OpenAI and Azure embeddings clients to limit the number of inputs per request. Default value is `200`.
- `EMBEDDINGS_DIMENSIONS`: (Optional) Output vector size to request from the embedding model. Only honored by the `openai` and `azure` providers, and only supported by `text-embedding-3-*` models. Leave unset to use the model's native dimensionality (1536 for `text-embedding-3-small`, 3072 for `text-embedding-3-large`). Setting a smaller value (e.g. `512`, `1024`) trades some retrieval quality for lower storage cost and faster similarity search. Note: do not change this on an existing collection — all vectors in a `pgvector` column must share the same dimensionality.
- `RAG_AZURE_OPENAI_API_VERSION`: (Optional) Default is `2023-05-15`. The version of the Azure OpenAI API.
- `RAG_AZURE_OPENAI_API_KEY`: (Optional) The API key for Azure OpenAI service.
    - Note: `AZURE_OPENAI_API_KEY` will work but `RAG_AZURE_OPENAI_API_KEY` will override it in order to not conflict with LibreChat setting.
- `RAG_AZURE_OPENAI_ENDPOINT`: (Optional) The endpoint URL for Azure OpenAI service, including the resource.
    - Example: `https://YOUR_RESOURCE_NAME.openai.azure.com`.
    - Note: `AZURE_OPENAI_ENDPOINT` will work but `RAG_AZURE_OPENAI_ENDPOINT` will override it in order to not conflict with LibreChat setting.
- `HF_TOKEN`: (Optional) if needed for `huggingface` option.
- `OLLAMA_BASE_URL`: (Optional) defaults to `http://ollama:11434`.
- `ATLAS_SEARCH_INDEX`: (Optional) the name of the vector search index if using Atlas MongoDB, defaults to `vector_index`
- `MONGO_VECTOR_COLLECTION`: Deprecated for MongoDB, please use `ATLAS_SEARCH_INDEX` and `COLLECTION_NAME`
- `AWS_DEFAULT_REGION`: (Optional) defaults to `us-east-1`
- `AWS_ACCESS_KEY_ID`: (Optional) needed for bedrock embeddings
- `AWS_SECRET_ACCESS_KEY`: (Optional) needed for bedrock embeddings
- `GOOGLE_API_KEY`, `GOOGLE_KEY`, `RAG_GOOGLE_API_KEY`: (Optional) Google API key for Google GenAI embeddings. Priority order: RAG_GOOGLE_API_KEY > GOOGLE_KEY > GOOGLE_API_KEY
- `AWS_SESSION_TOKEN`: (Optional) may be needed for bedrock embeddings
- `GOOGLE_APPLICATION_CREDENTIALS`: (Optional) needed for Google VertexAI embeddings. This should be a path to a service account credential file in JSON format.
- `GOOGLE_CLOUD_PROJECT`: (Optional) Google Cloud project ID, needed for VertexAI embeddings.
- `GOOGLE_CLOUD_LOCATION`: (Optional) Google Cloud region for VertexAI embeddings. Defaults to `us-central1`.
- `RAG_CHECK_EMBEDDING_CTX_LENGTH` (Optional) Default is true, disabling this will send raw input to the embedder, use this for custom embedding models.

Make sure to set these environment variables before running the application. You can set them in a `.env` file or as system environment variables.

### Authentication

Requests carry a bearer JWT. Two token generations are recognised.

**Strict tokens** are signed with `RAG_JWT_SECRET` — a key dedicated to this
service — and carry `iss`, `aud`, `sub`, `exp`, a tenant claim, and scopes.
`RAG_JWT_SECRET` must never be the application's `JWT_SECRET`: sharing the key
makes every token minted for rag_api simultaneously a full API session token
for the calling app, and vice versa. Startup refuses to proceed if the two
match.

**Legacy tokens** are the `{"id": userId}` shape older LibreChat releases mint.
They are accepted while `RAG_AUTH_ACCEPT_LEGACY` is true. On the `/v1` service
endpoints the flag relaxes only the *claim shape* — a token signed with the
application `JWT_SECRET` is never accepted there, in any mode.

Legacy tokens predate scopes and entity lists, so they are grandfathered into
both. That grandfather is limited to the `{"id": ...}` shape. A token that states
its own `scopes` or `entities` keeps exactly what it stated even if it fails
strict validation for some other reason — otherwise dropping `exp`, the tenant or
the scopes from a `rag:embed`-only token would hand it every scope and
unrestricted entity access.

- `RAG_JWT_SECRET`: (Optional) dedicated signing secret for this service. At
  least 32 characters for HMAC algorithms. Required when `RAG_SEARCH_API_ENABLED=true`.
  Whenever it is set the service validates it at startup — the key signs tokens
  the middleware honours on `/query` and the upload routes regardless of whether
  the search endpoints are mounted, so a short secret fails startup either way.
- `RAG_JWT_PUBLIC_KEY`: (Optional) verification key when `RAG_JWT_ALGORITHM` is asymmetric.
- `RAG_JWT_ALGORITHM`: (Optional) default `HS256`. `HS*`, `RS*`, `ES*` and `EdDSA` are supported.
- `RAG_JWT_ISSUER`: (Optional) required `iss`, default `librechat`.
- `RAG_JWT_AUDIENCE`: (Optional) required `aud`, default `rag_api`.
- `RAG_JWT_LEEWAY_SECONDS`: (Optional) clock skew allowance, default `0`.
- `RAG_AUTH_ACCEPT_LEGACY`: (Optional) default `true`. Set to `false` once every
  caller mints the full claim set.

Scopes: `rag:embed` grants `POST /v1/embeddings`, `rag:rerank` grants
`POST /v1/rerank`. The tenant claim (`tenant`, or `tenant_id`) is required for
strict tokens; the reserved value `__SYSTEM__` is always refused.

Callers may pass an `entity_id` to `/query` and `/query_multiple` to reach an
agent's knowledge-base files. Strict tokens must list that id in their
`entities` claim. Legacy tokens cannot prove entity access, so the id is taken
at face value — which is the reason to flip `RAG_AUTH_ACCEPT_LEGACY` off once
the migration lands.

### Retrieval scope

Retrieval is scoped by `(tenant, owner/entity, file_id)`, and all three go into
the vector-store predicate before ranking. A chunk outside the caller's scope
matches nothing rather than being filtered out of the result set afterwards.
There is one scope builder (`app/scope.py`); no route writes its own scope
clause.

Chunks record the writing caller's `tenant_id` and `user_id`. Chunks written
before `tenant_id` existed carry no value and normalize to the base tenant
`__BASE__`, so a single-tenant deployment reads them exactly as before while a
named tenant never absorbs untagged content. Documents written before `user_id`
was recorded are no longer visible to any caller; re-embed them if you still
need them.

Writes are scoped too: uploading under an `entity_id` the token does not permit
is refused, so a caller cannot plant content in a knowledge base it cannot read.

The file-addressed routes are scoped by the same builder, because a file id is
caller-supplied and proves nothing about who may read it:

| Route | Scope |
| --- | --- |
| `GET /ids` | lists only the caller's own file ids |
| `GET /documents?ids=` | chunks the caller owns; anything else is `404` |
| `GET /documents/{id}/context` | as above, for one file |
| `DELETE /documents` | deletes only the caller's rows for those ids |

Each accepts an optional `entity_id` query parameter, with the same rule as
`/query`: strict tokens must list the id in their `entities` claim, legacy
tokens are taken at face value. A file outside the scope reads as "not found"
rather than "found but refused", so none of these routes is an existence oracle.

The scope is inside the `DELETE` predicate rather than beside it. Two owners can
hold rows under one file id — the id is chosen by whoever uploads — so a delete
that filtered on the id alone would take both.

**Upgrade note.** Files embedded under an `entity_id` (agent knowledge bases)
are owned by that entity, so deleting or reading them needs `entity_id` on these
routes just as querying them already does. A client that deletes an agent's file
with a plain user token and no `entity_id` now gets `404` and leaves the chunks
in place; pass the entity the file was uploaded under to remove them.

### Authorize before egress

Rerank and embedding send text to an inference provider. That text has left the
trust boundary whether or not the caller ever sees a response, so authorization
happens before the call rather than as a filter on its result:

- `/v1/rerank` probes every candidate id against the store — metadata only, no
  vectors and no document text. An id that exists but resolves to nothing inside
  the caller's scope makes the whole request `403`, and nothing is embedded. Ids
  that match nothing in the store (web-scrape candidates, synthetic ids) are
  unaffected. If the probe cannot run, the request fails closed with `503`
  rather than embedding text it could not check.
- `/v1/embeddings` reads no store at all: it embeds only text the authenticated
  caller supplied in the request body. Scope, quota and limit rejections all
  happen before the backend is called.

### Search service endpoints

Enabled by `RAG_SEARCH_API_ENABLED=true` (default `false`); the service refuses
to start if enabled without a valid signing configuration.

```text
POST /v1/embeddings
{ "space": "chat-v1", "input_type": "query" | "document",
  "inputs": [{ "id": "...", "text": "..." }] }
-> { "space", "model", "dimensions", "normalized",
     "items": [{ "id", "content_hash", "embedding" }], "usage" }

POST /v1/rerank
{ "profile": "fast-v1", "query": "...",
  "candidates": [{ "id", "text", "base_score" }], "top_n": 25 }
-> { "profile", "model", "results": [{ "id", "index", "score" }] }
```

Limits: 64 inputs and 256,000 aggregate characters per embeddings call; 50
candidates, 8,000 query characters and `top_n <= 25` per rerank call. The
character limits are provider limits, so they bind the payload that is actually
sent — after NFKC normalization and including the space's task prefix, which
applies to every input. Over-limit requests are rejected with `422` before any
text reaches the backend. Caller ids are preserved and tie
ordering is deterministic (ties break on the candidate's position in the
request). `content_hash` is the SHA-256 of the NFKC-normalized,
whitespace-collapsed text that was embedded. Vectors leave the service
L2-normalized.

The `chat-v1` space is substitution-locked: if its backend is unavailable, or
returns a different dimensionality, the call fails with 503 rather than falling
back to another model or space.

- `RAG_SEARCH_API_ENABLED`: (Optional) default `false`.
- `RAG_EMBEDDING_SPACE`: (Optional) space name, default `chat-v1`.
- `RAG_CHAT_EMBEDDING_MODEL`: (Optional) default `qwen3-embedding-8b`.
- `RAG_CHAT_EMBEDDING_DIMENSIONS`: (Optional) default `1024`.
- `RAG_CHAT_EMBEDDING_BASEURL` / `RAG_CHAT_EMBEDDING_API_KEY`: (Optional)
  OpenAI-compatible endpoint for the space, defaulting to `RAG_OPENAI_BASEURL`
  and `RAG_OPENAI_API_KEY`.
- `RAG_CHAT_EMBEDDING_QUERY_PREFIX` / `RAG_CHAT_EMBEDDING_DOCUMENT_PREFIX`:
  (Optional) task prefixes applied to `input_type: "query"` and
  `input_type: "document"` respectively, both empty by default.

`input_type` selects how the text is encoded. Asymmetric models — qwen3-embedding
among them — expect a query and a passage to be encoded differently, so the space
applies the matching task prefix and, where the backend exposes a dedicated batch
query encoder, uses it. The prefixes are part of the space's locked definition:
changing one changes every vector the space produces, so stored vectors have to
be rebuilt to match. `content_hash` is always taken over the un-prefixed
normalized text, so it stays a stable cache key for the same content.

`/v1/rerank` implements the `fast-v1` profile as embed-blend: the query is
embedded once through the retrieval cache, candidate vectors are read from the
pgvector store wherever they already exist, and only vectorless candidates are
embedded. The final score is reciprocal-rank fusion of cosine similarity with
the caller's `base_score` — never pure embedding order, which regresses
identifier and exact-match queries. Candidate ids resolve against the stored
row's `uuid` or the chunk `digest` in its metadata, always restricted to owners
the token permits.

- `RAG_RERANK_RRF_K`: (Optional) fusion constant, default `60`.
- `RAG_RERANK_SIMILARITY_WEIGHT` / `RAG_RERANK_BASE_WEIGHT`: (Optional) arm weights, default `1.0`.
- `RAG_QUERY_EMBEDDING_CACHE_SIZE`: (Optional) shared query-vector cache size, default `128`.

Rate limits apply per tenant and per subject, with separate embedding and
rerank budgets. Counters are process-local, so a multi-pod deployment enforces
`limit x pods`.

- `RAG_RATE_LIMIT_ENABLED`: (Optional) default `true`.
- `RAG_RATE_LIMIT_WINDOW_SECONDS`: (Optional) default `60`.
- `RAG_RATE_LIMIT_EMBED_TENANT` / `RAG_RATE_LIMIT_EMBED_SUBJECT`: (Optional) default `600` / `120`.
- `RAG_RATE_LIMIT_RERANK_TENANT` / `RAG_RATE_LIMIT_RERANK_SUBJECT`: (Optional) default `900` / `180`.

Note for `atlas-mongo` deployments: `/query` and `/query_multiple` now put the
owner predicate in the vector-search `filter`, so `user_id` must be declared as
a filter field on the Atlas vector index.

### Embedding Batch Processing

For large files, you can enable batched embedding processing to reduce memory consumption. This is particularly useful in memory-constrained environments like Kubernetes pods with memory limits.

#### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_BATCH_SIZE` | `500` | Number of document chunks to process per batch. `0` disables batching (original behavior). |
| `EMBEDDING_MAX_QUEUE_SIZE` | `3` | Maximum number of batches to buffer in memory during async processing. |
| `PARALLEL_EXECUTION` | `2` | Maximum number of async embedding/database insertion consumers per file when batching is enabled. |

#### Recommended Settings

For `text-embedding-3-small` model:
- `EMBEDDING_BATCH_SIZE=750` - Good balance of throughput and memory

For memory-constrained environments (< 2GB RAM):
- `EMBEDDING_BATCH_SIZE=100-250`

For high-throughput environments:
- `EMBEDDING_BATCH_SIZE=1000-2000`
- `EMBEDDING_MAX_QUEUE_SIZE=5`
- Increase `PARALLEL_EXECUTION` cautiously; it applies per active file upload.

#### Behavior

When `EMBEDDING_BATCH_SIZE > 0`:
- Documents are processed in batches of the specified size
- Up to `PARALLEL_EXECUTION` batches for the same file can be embedded and inserted concurrently
- `PARALLEL_EXECUTION` is per request/file. Total process concurrency can be roughly `active uploads * PARALLEL_EXECUTION`, bounded indirectly by `RAG_THREAD_POOL_SIZE` and downstream provider/database limits
- On failure, remaining batch work is stopped and successfully inserted documents are rolled back
- Memory usage is bounded by queued plus active batches, roughly `EMBEDDING_BATCH_SIZE * (EMBEDDING_MAX_QUEUE_SIZE + PARALLEL_EXECUTION)`
- Ingestion lifecycle logs include route, user, file, chunk count, file size, elapsed time, and selected process memory context. Per-batch queue/insert progress is logged at debug level

When `EMBEDDING_BATCH_SIZE <= 0`:
- All documents are processed at once (original behavior)
- Better for small files or memory-rich environments

### Use Atlas MongoDB as Vector Database

Instead of using the default pgvector, we could use [Atlas MongoDB](https://www.mongodb.com/products/platform/atlas-vector-search) as the vector database. To do so, set the following environment variables

```env
VECTOR_DB_TYPE=atlas-mongo
ATLAS_MONGO_DB_URI=<mongodb+srv://...>
COLLECTION_NAME=<vector collection>
ATLAS_SEARCH_INDEX=<vector search index>
```

The `ATLAS_MONGO_DB_URI` could be the same or different from what is used by LibreChat. Even if it is the same, the `$COLLECTION_NAME` collection needs to be a completely new one, separate from all collections used by LibreChat. In addition,  create a vector search index for collection above (remember to assign `$ATLAS_SEARCH_INDEX`) with the following json:

```json
{
  "fields": [
    {
      "numDimensions": 1536,
      "path": "embedding",
      "similarity": "cosine",
      "type": "vector"
    },
    {
      "path": "file_id",
      "type": "filter"
    },
    {
      "path": "user_id",
      "type": "filter"
    },
    {
      "path": "tenant_id",
      "type": "filter"
    }
  ]
}
```

Follow one of the [four documented methods](https://www.mongodb.com/docs/atlas/atlas-vector-search/create-index/#procedure) to create the vector index.

#### Migrating an existing Atlas vector index

`/query` and `/query_multiple` scope every search by owner and tenant, and those
predicates are Atlas **pre-filters**. Atlas rejects a `$vectorSearch` whose filter
references a path that is not declared as a filter field, so an index created
before this release — one carrying only `file_id` — makes those endpoints fail
rather than return results.

If your index predates this release:

1. In the Atlas UI, open **Atlas Search → your `$ATLAS_SEARCH_INDEX` → Edit Index
   Definition**, or use `mongosh`/the Admin API with the JSON above.
2. Add the `user_id` and `tenant_id` filter entries, keeping `numDimensions` and
   `similarity` at whatever your deployment already uses.
3. Save and wait for the index status to return to **Active**. Atlas rebuilds the
   index in place; no re-embedding is required.
4. Backfill `tenant_id` on chunks embedded before this release. They carry no
   such field, and Atlas vector-search pre-filters match on declared scalar
   values rather than on absent fields, so an untagged chunk stays invisible
   until it is stamped with the base tenant:

   ```javascript
   db.getCollection("<COLLECTION_NAME>").updateMany(
     { tenant_id: { $exists: false } },
     { $set: { tenant_id: "__BASE__" } }
   )
   ```

   `__BASE__` is the tenant every caller without a `tenant` claim resolves to, so
   this restores exactly the visibility those chunks had before. Run it once,
   after the index reaches **Active**.

Queries against `file_id` continue to work throughout — only the owner- and
tenant-scoped paths wait on the rebuild.

#### Create a `file_id` Index (recommended)

We recommend creating a standard MongoDB index on `file_id` to keep lookups fast. After creating the collection, run the following once (via Atlas UI, Compass, or `mongosh`):

```javascript
db.getCollection("<COLLECTION_NAME>").createIndex({ file_id: 1 })
```

Replace `<COLLECTION_NAME>` with the same collection used by the RAG API. This ensures lookups remain fast even as the number of embedded documents grows.


### Proxy Configuration

When using the RAG API with LibreChat and you need to configure proxy settings, you can set the `HTTP_PROXY` and `HTTPS_PROXY` environment variables in the [`docker-compose.override.yml`](https://www.librechat.ai/docs/configuration/docker_override) file (from the LibreChat repository):

```yaml
rag_api:
    environment:
        - HTTP_PROXY=<your-proxy>
        - HTTPS_PROXY=<your-proxy>
```

This configuration will ensure that all HTTP/HTTPS requests from the RAG API container are routed through your specified proxy server.


### Cloud Installation Settings:

#### AWS:
Make sure your RDS Postgres instance adheres to this requirement:

`The pgvector extension version 0.5.0 is available on database instances in Amazon RDS running PostgreSQL 15.4-R2 and higher, 14.9-R2 and higher, 13.12-R2 and higher, and 12.16-R2 and higher in all applicable AWS Regions, including the AWS GovCloud (US) Regions.`

In order to setup RDS Postgres with RAG API, you can follow these steps:

* Create a RDS Instance/Cluster using the provided [AWS Documentation](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/USER_CreateDBInstance.html).
* Login to the RDS Cluster using the Endpoint connection string from the RDS Console or from your IaC Solution output.
* The login is via the *Master User*.
* Create a dedicated database for rag_api:
``` create database rag_api;```.
* Create a dedicated user\role for that database:
``` create role rag;```

* Switch to the database you just created: ```\c rag_api```
* Enable the Vector extension: ```create extension vector;```
* Use the documentation provided above to set up the connection string to the RDS Postgres Instance\Cluster.

Notes:
  * Even though you're logging with a Master user, it doesn't have all the super user privileges, that's why we cannot use the command: ```create role x with superuser;```
  * If you do not enable the extension, rag_api service will throw an error that it cannot create the extension due to the note above.

### Dev notes:

#### Running Tests

##### Prerequisites

Install test dependencies:

```bash
pip install -r test_requirements.txt
```

##### Running All Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage (if pytest-cov is installed)
pytest --cov=app
```

##### Running Specific Test Files

```bash
# Run batch processing unit tests
pytest tests/test_batch_processing.py -v

# Run batch processing integration tests (memory optimization tests)
pytest tests/test_batch_processing_integration.py -v

# Run main API tests
pytest tests/test_main.py -v
```

##### Running Tests by Category

```bash
# Run only integration tests (marked with @pytest.mark.integration)
pytest -m integration -v

# Skip integration tests
pytest -m "not integration" -v

# Run only async tests
pytest -k "async" -v
```

##### Test Categories

| Test File | Description |
|-----------|-------------|
| `test_batch_processing.py` | Unit tests for batch processing functions |
| `test_batch_processing_integration.py` | Memory optimization and integration tests |
| `test_main.py` | API endpoint tests |
| `test_config.py` | Configuration tests |
| `test_middleware.py` | Middleware tests |
| `test_models.py` | Model tests |

##### Memory Optimization Tests

The `test_batch_processing_integration.py` file includes tests that verify the memory optimization behavior:

- **`test_memory_bounded_by_batch_size`**: Verifies that the number of documents in memory at any time is bounded by `EMBEDDING_BATCH_SIZE`
- **`test_memory_tracking_with_tracemalloc`**: Uses Python's `tracemalloc` to monitor memory usage during batch processing
- **`test_sync_memory_bounded_by_batch_size`**: Same verification for the synchronous code path

Run memory tests specifically:

```bash
pytest tests/test_batch_processing_integration.py::TestMemoryOptimization -v
pytest tests/test_batch_processing_integration.py::TestSyncBatchedMemory -v
```

#### Installing pre-commit formatter

Run the following commands to install pre-commit formatter, which uses [black](https://github.com/psf/black) code formatter:

```bash
pip install pre-commit
pre-commit install
```

