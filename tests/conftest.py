import sys
import os

# Add the project root directory to sys.path.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set environment variables for testing.
os.environ["RAG_HOST"] = "127.0.0.1"
os.environ["RAG_PORT"] = "8000"
os.environ["JWT_SECRET"] = "testsecret"
os.environ["EMBEDDINGS_PROVIDER"] = "openai"
os.environ["OPENAI_API_KEY"] = "dummy"