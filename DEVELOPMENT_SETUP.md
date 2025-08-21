# RAG API - Development Setup Guide

## Project Overview
This is a RAG (Retrieval-Augmented Generation) API built with FastAPI, LangChain, and PostgreSQL/pgvector. It provides document ingestion, vector storage, and retrieval capabilities for AI applications.

## Setup Status ✅
- ✅ Project cloned from GitHub
- ✅ Python 3.11 virtual environment created
- ✅ All dependencies installed
- ✅ PostgreSQL with pgvector extension set up
- ✅ Environment configuration ready
- ✅ Application imports successfully

## Prerequisites Installed
- Python 3.11 (via Homebrew)
- PostgreSQL 14 (via Homebrew)
- pgvector extension (via Homebrew)

## Project Structure
```
rag_api_project/
├── app/                    # Main application code
├── tests/                  # Test files
├── main.py                # FastAPI application entry point
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (created by setup)
├── setup_env.sh          # Environment setup script
└── venv/                 # Python virtual environment
```

## Quick Start

### 1. Activate Virtual Environment
```bash
source venv/bin/activate
```

### 2. Configure Environment
Edit the `.env` file and add your OpenAI API key:
```bash
# Replace YOUR_OPENAI_API_KEY_HERE with your actual API key
RAG_OPENAI_API_KEY=sk-your-actual-openai-api-key-here
```

### 3. Start the Development Server
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Database Setup
PostgreSQL is already configured with:
- Database: `rag_api`
- User: `andrew` (your system user)
- Extension: `vector` (pgvector)

To manage PostgreSQL:
```bash
# Start PostgreSQL
brew services start postgresql@14

# Stop PostgreSQL
brew services stop postgresql@14

# Connect to database
psql rag_api
```

## Development Commands

### Install Additional Dependencies
```bash
source venv/bin/activate
pip install package_name
pip freeze > requirements.txt  # Update requirements
```

### Run Tests
```bash
source venv/bin/activate
python -m pytest tests/
```

### Code Quality
```bash
# Install development tools
pip install black flake8 mypy

# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .
```

## API Features
Based on the repository, this RAG API supports:
- Document upload and processing (PDF, DOCX, TXT, etc.)
- Text chunking and embedding generation
- Vector storage with PostgreSQL/pgvector
- Semantic search and retrieval
- Multiple embedding providers (OpenAI, HuggingFace, Azure, etc.)
- JWT authentication (optional)
- File management and metadata storage

## Environment Variables
Key configuration options in `.env`:

### Required
- `RAG_OPENAI_API_KEY`: Your OpenAI API key

### Database
- `VECTOR_DB_TYPE`: Vector database type (pgvector/atlas-mongo)
- `POSTGRES_DB`: Database name
- `POSTGRES_USER`: Database user
- `DB_HOST`: Database host
- `DB_PORT`: Database port

### Server
- `RAG_HOST`: Server host (default: 0.0.0.0)
- `RAG_PORT`: Server port (default: 8000)

### RAG Configuration
- `COLLECTION_NAME`: Vector collection name
- `CHUNK_SIZE`: Text chunk size (default: 1500)
- `CHUNK_OVERLAP`: Chunk overlap (default: 100)
- `EMBEDDINGS_PROVIDER`: Embedding provider (openai/huggingface/azure/etc.)

## Troubleshooting

### PostgreSQL Issues
```bash
# Check if PostgreSQL is running
brew services list | grep postgresql

# Restart PostgreSQL
brew services restart postgresql@14

# Check database connection
psql rag_api -c "SELECT version();"
```

### Python Environment Issues
```bash
# Recreate virtual environment
rm -rf venv
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Import Errors
Make sure you're in the project directory and virtual environment is activated:
```bash
cd /Users/andrew/Documents/projects/rag_api_project
source venv/bin/activate
```

## Next Steps for Development
1. Add your OpenAI API key to `.env`
2. Test document upload via the API docs at http://localhost:8000/docs
3. Explore the codebase in the `app/` directory
4. Add custom endpoints or modify existing functionality
5. Write tests for new features

## Useful Resources
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [LangChain Documentation](https://python.langchain.com/)
- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [Original Repository](https://github.com/VaitaR/rag_api)

## Support
If you encounter issues:
1. Check the logs when running the server
2. Verify all environment variables are set correctly
3. Ensure PostgreSQL is running and accessible
4. Check that all dependencies are installed in the virtual environment
