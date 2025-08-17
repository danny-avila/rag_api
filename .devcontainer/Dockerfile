# --------------------------------------------------------
# Development Dockerfile for RAG API
# --------------------------------------------------------
ARG PYTHON_VERSION=3.10
FROM python:${PYTHON_VERSION}-bookworm

# --------------------------------------------------------
# Install system dependencies
# --------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Essential for development
    git \
    sudo \
    # Document processing (for RAG API)
    pandoc \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------
# Create non-root development user
# --------------------------------------------------------
RUN useradd -m -s /bin/bash vscode \
    && usermod -aG sudo vscode \
    && echo "vscode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# --------------------------------------------------------
# Configure Python environment
# --------------------------------------------------------
WORKDIR /workspace
RUN pip install --no-cache-dir --upgrade pip

# --------------------------------------------------------
# Configure environment
# --------------------------------------------------------
ENV PYTHONPATH=/workspace \
    PYTHONUNBUFFERED=1 \
    SCARF_NO_ANALYTICS=true

# Switch to non-root user
USER vscode
