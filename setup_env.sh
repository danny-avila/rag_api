#!/bin/bash

# RAG API Setup Script
echo "Setting up RAG API environment..."

# Create .env file from template
cat > .env << 'EOF'
# RAG API Configuration
# Fill in your actual values

# OpenAI API Configuration
RAG_OPENAI_API_KEY=YOUR_OPENAI_API_KEY_HERE

# Database Configuration (PostgreSQL with pgvector)
VECTOR_DB_TYPE=pgvector
POSTGRES_DB=rag_api
POSTGRES_USER=andrew
POSTGRES_PASSWORD=
DB_HOST=localhost
DB_PORT=5432

# Server Configuration
RAG_HOST=0.0.0.0
RAG_PORT=8000

# RAG Configuration
COLLECTION_NAME=testcollection
CHUNK_SIZE=1500
CHUNK_OVERLAP=100
RAG_UPLOAD_DIR=./uploads/
PDF_EXTRACT_IMAGES=False

# Debug Settings
DEBUG_RAG_API=True
DEBUG_PGVECTOR_QUERIES=False
CONSOLE_JSON=False

# Embeddings Configuration
EMBEDDINGS_PROVIDER=openai
EMBEDDINGS_MODEL=text-embedding-3-small
RAG_CHECK_EMBEDDING_CTX_LENGTH=True
EOF

# Create uploads directory
mkdir -p uploads

echo "Environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file and add your OpenAI API key"
echo "2. Set up PostgreSQL database with pgvector extension"
echo "3. Run: source venv/bin/activate"
echo "4. Run: uvicorn main:app --reload"
echo ""
echo "For development with hot reload:"
echo "uvicorn main:app --reload --host 0.0.0.0 --port 8000"
