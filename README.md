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

### Environment Variables

The following environment variables are required to run the application:

- `RAG_OPENAI_API_KEY`: The API key for OpenAI API Embeddings (if using default settings).
    - Note: `OPENAI_API_KEY` will work but `RAG_OPENAI_API_KEY` will override it in order to not conflict with LibreChat setting.
- `RAG_OPENAI_BASEURL`: (Optional) The base URL for your OpenAI API Embeddings
- `RAG_OPENAI_PROXY`: (Optional) Proxy for OpenAI API Embeddings
    - Note: When using with LibreChat, you can also set `HTTP_PROXY` and `HTTPS_PROXY` environment variables in the `docker-compose.override.yml` file (see [Proxy Configuration](#proxy-configuration) section below)
- `VECTOR_DB_TYPE`: (Optional) select vector database type, default to `pgvector`.
- `POSTGRES_USE_UNIX_SOCKET`: (Optional) Set to "True" when connecting to the PostgreSQL database server with Unix Socket.
- `POSTGRES_DB`: (Optional) The name of the PostgreSQL database, used when `VECTOR_DB_TYPE=pgvector`.
- `POSTGRES_USER`: (Optional) The username for connecting to the PostgreSQL database.
- `POSTGRES_PASSWORD`: (Optional) The password for connecting to the PostgreSQL database.
- `DB_HOST`: (Optional) The hostname or IP address of the PostgreSQL database server.
- `DB_PORT`: (Optional) The port number of the PostgreSQL database server.
- `RAG_HOST`: (Optional) The hostname or IP address where the API server will run. Defaults to "0.0.0.0"
- `RAG_PORT`: (Optional) The port number where the API server will run. Defaults to port 8000.
- `JWT_SECRET`: (Optional) The secret key used for verifying JWT tokens for requests.
  - The secret is only used for verification. This basic approach assumes a signed JWT from elsewhere.
  - Omit to run API without requiring authentication

- `COLLECTION_NAME`: (Optional) The name of the collection in the vector store. Default value is "testcollection".
- `CHUNK_SIZE`: (Optional) The size of the chunks for text processing. Default value is "1500".
- `CHUNK_OVERLAP`: (Optional) The overlap between chunks during text processing. Default value is "100".
- `RAG_UPLOAD_DIR`: (Optional) The directory where uploaded files are stored. Default value is "./uploads/".
- `PDF_EXTRACT_IMAGES`: (Optional) A boolean value indicating whether to extract images from PDF files. Default value is "False".
- `DEBUG_RAG_API`: (Optional) Set to "True" to show more verbose logging output in the server console, and to enable postgresql database routes
- `DEBUG_PGVECTOR_QUERIES`: (Optional) Set to "True" to enable detailed PostgreSQL query logging for pgvector operations. Useful for debugging performance issues with vector database queries.
- `CONSOLE_JSON`: (Optional) Set to "True" to log as json for Cloud Logging aggregations
- `EMBEDDINGS_PROVIDER`: (Optional) either "openai", "bedrock", "azure", "huggingface", "huggingfacetei", "vertexai", "ollama", or "nvidia", where "huggingface" uses sentence_transformers; defaults to "openai"
- `EMBEDDINGS_MODEL`: (Optional) Set a valid embeddings model to use from the configured provider.
    - **Defaults**
    - openai: "text-embedding-3-small"
    - azure: "text-embedding-3-small" (will be used as your Azure Deployment)
    - huggingface: "sentence-transformers/all-MiniLM-L6-v2"
    - huggingfacetei: "http://huggingfacetei:3000". Hugging Face TEI uses model defined on TEI service launch.
    - vertexai: "text-embedding-004"
    - ollama: "nomic-embed-text"
    - bedrock: "amazon.titan-embed-text-v1"
    - nvidia: "nvidia/llama-3.2-nemoretriever-300m-embed-v1"
- `EMBEDDINGS_PROVIDER_BACKUP`: (Optional) Backup provider for automatic failover ("openai", "bedrock", "azure", "huggingface", "huggingfacetei", "vertexai", "ollama", "nvidia")
- `EMBEDDINGS_MODEL_BACKUP`: (Optional) Backup model to use when primary provider fails
- `PRIMARY_FAILOVER_COOLDOWN_MINUTES`: (Optional) Minutes to wait before retrying failed primary provider (default: 1)
- `EMBED_CONCURRENCY_LIMIT`: (Optional) Maximum concurrent embedding requests to prevent overload (default: 3)

#### Backup Embedding Provider Configuration
The RAG API supports intelligent backup embedding providers for high availability:
- **Automatic failover**: When primary provider fails, requests automatically switch to backup
- **Smart cooldown**: Failed primary providers are avoided for configurable time period
- **Transparent operation**: LibreChat receives success responses when backup succeeds
- **Automatic recovery**: Primary provider is retried when cooldown expires

#### NVIDIA Embedding Provider Configuration
- `NVIDIA_BASE_URL`: (Optional) NVIDIA API endpoint URL (default: "http://localhost:8003/v1")
- `NVIDIA_API_KEY`: (Optional) API key for NVIDIA embedding service
- `NVIDIA_MODEL`: (Optional) NVIDIA model to use (default: "nvidia/llama-3.2-nemoretriever-300m-embed-v1")
- `NVIDIA_INPUT_TYPE`: (Optional) Input type for embeddings ("query", "passage", default: "passage")
- `NVIDIA_ENCODING_FORMAT`: (Optional) Encoding format ("float", "base64", default: "float")
- `NVIDIA_TRUNCATE`: (Optional) Truncate input if too long ("NONE", "START", "END", default: "NONE")
- `NVIDIA_MAX_RETRIES`: (Optional) Maximum retry attempts (default: 3)
- `NVIDIA_TIMEOUT`: (Optional) Read timeout in seconds (default: 3, connection timeout: 2s)
- `NVIDIA_MAX_BATCH_SIZE`: (Optional) Maximum texts per batch (default: 32)

#### AWS Bedrock Enhanced Configuration
- `BEDROCK_EMBEDDING_DIMENSIONS`: (Optional) For Titan V2 models - embedding dimensions (256, 512, or 1024, default: 1024)
- `BEDROCK_EMBEDDING_NORMALIZE`: (Optional) For Titan V2 models - normalize embeddings ("true"/"false", default: "true")
- `BEDROCK_MAX_BATCH_SIZE`: (Optional) Maximum texts per Bedrock batch (default: 15)
- `BEDROCK_INITIAL_RETRY_DELAY`: (Optional) Initial retry delay in seconds for rate limiting (default: 1.0)
- `BEDROCK_MAX_RETRY_DELAY`: (Optional) Maximum retry delay in seconds (default: 60.0)
- `BEDROCK_BACKOFF_FACTOR`: (Optional) Exponential backoff multiplier (default: 2.0)

**Bedrock Timeout Configuration (optimized for backup failover):**
- Connection timeout: 5 seconds 
- Read timeout: 30 seconds (reduced from default 60s)

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
- `AWS_SESSION_TOKEN`: (Optional) may be needed for bedrock embeddings
- `GOOGLE_APPLICATION_CREDENTIALS`: (Optional) needed for Google VertexAI embeddings. This should be a path to a service account credential file in JSON format, as accepted by [langchain](https://python.langchain.com/api_reference/google_vertexai/index.html)
- `RAG_CHECK_EMBEDDING_CTX_LENGTH` (Optional) Default is true, disabling this will send raw input to the embedder, use this for custom embedding models.

Make sure to set these environment variables before running the application. You can set them in a `.env` file or as system environment variables.

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
    }
  ]
}
```

Follow one of the [four documented methods](https://www.mongodb.com/docs/atlas/atlas-vector-search/create-index/#procedure) to create the vector index.

### High Availability Configuration Example

For production environments requiring maximum uptime, you can configure redundant embedding providers with automatic failover. This example uses NVIDIA as the primary provider with AWS Bedrock as backup:

```env
# Primary Provider - NVIDIA Embeddings (Local/On-Premises)
EMBEDDINGS_PROVIDER=nvidia
EMBEDDINGS_MODEL=nvidia/llama-3.2-nemoretriever-300m-embed-v1
NVIDIA_BASE_URL=http://your-nvidia-server:8003/v1
NVIDIA_API_KEY=your-nvidia-api-key
NVIDIA_MAX_BATCH_SIZE=32
NVIDIA_TIMEOUT=3

# Backup Provider - AWS Bedrock Titan V2 Embeddings  
EMBEDDINGS_PROVIDER_BACKUP=bedrock
EMBEDDINGS_MODEL_BACKUP=amazon.titan-embed-text-v2:0
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_DEFAULT_REGION=us-west-2
BEDROCK_EMBEDDING_DIMENSIONS=512
BEDROCK_EMBEDDING_NORMALIZE=true

# Failover Configuration
PRIMARY_FAILOVER_COOLDOWN_MINUTES=2
EMBED_CONCURRENCY_LIMIT=3

# Performance Tuning
CHUNK_SIZE=1500
CHUNK_OVERLAP=100
```

**How this works:**
- **Primary**: NVIDIA embeddings serve all requests when available
- **Failover**: If NVIDIA fails, requests automatically switch to Bedrock
- **Cooldown**: After failure, NVIDIA is not retried for 2 minutes (prevents cascading failures)
- **Recovery**: NVIDIA is automatically retried when cooldown expires
- **Transparency**: LibreChat receives successful responses when backup succeeds

This configuration ensures high availability with seamless failover while maintaining optimal performance and cost efficiency.

### Proxy Configuration

When using the RAG API with LibreChat and you need to configure proxy settings, you can set the `HTTP_PROXY` and `HTTPS_PROXY` environment variables in the [`docker-compose.override.yml`](https://www.librechat.ai/docs/configuration/docker_override) file (from the LibreChat repository):

```yaml
rag_api:
    environment:
        - HTTP_PROXY=<your-proxy>
        - HTTPS_PROXY=<your-proxy>
```

This configuration will ensure that all HTTP/HTTPS requests from the RAG API container are routed through your specified proxy server.

### Failover Testing

The RAG API includes comprehensive testing tools to validate backup provider functionality and performance under real-world conditions.

#### Automated Failover Testing

Test the backup embedding system with realistic service interruptions:

```bash
# Run comprehensive failover test with service toggling
sudo -v && python3 test_failover_automation.py
```

**What this test does:**
- Uploads all PDFs from `$RAG_UPLOAD_DIR/pdfs` (or `~/uploads_rag/pdfs`)
- Randomly blocks/unblocks primary provider using iptables
- Measures failover performance and success rates
- Automatically cleans up all uploaded documents
- Generates detailed statistics and timing analysis

**Test Results Example:**
```
📊 TEST RESULTS SUMMARY
📈 UPLOAD STATISTICS:
   Total uploads: 28
   ✅ Successful: 28 (100.0%)
   Average duration: 5.2s
   Fastest upload: 0.8s
🔀 FAILOVER STATISTICS:
   Port blocks: 9
   Port unblocks: 8
```

#### Test Data Setup

Download sample PDFs for testing:

```bash
# Set upload directory (default: ~/uploads_rag)
export RAG_UPLOAD_DIR=${RAG_UPLOAD_DIR:-~/uploads_rag}
mkdir -p $RAG_UPLOAD_DIR/pdfs

# Download 30 sample PDFs from arXiv
for i in $(seq -f "%05g" 1 30); do
    wget "https://arxiv.org/pdf/2301.$i.pdf" -P $RAG_UPLOAD_DIR/pdfs
    sleep 1
done
```

#### Manual Testing

For manual testing with iptables service toggling:

```bash
# Block primary provider (triggers backup failover)
sudo iptables -A OUTPUT -p tcp --dport 8003 -j REJECT

# Unblock primary provider (triggers recovery)  
sudo iptables -D OUTPUT -p tcp --dport 8003 -j REJECT
```

Use these commands while uploading documents through LibreChat to observe failover behavior in the logs.

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

#### Installing pre-commit formatter

Run the following commands to install pre-commit formatter, which uses [black](https://github.com/psf/black) code formatter:

```bash
pip install pre-commit
pre-commit install
```
