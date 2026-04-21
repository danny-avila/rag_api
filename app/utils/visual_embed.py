"""Optional visual ingest pipeline for Multimodal-RAG.

Sits alongside the existing text pipeline. When VISUAL_EMBED_URL is set
and the uploaded file is a PDF, this module:

  1. Uses PyMuPDF (fitz) to render each page as a PNG at VISUAL_PAGE_DPI.
  2. POSTs each PNG to the CLIP embed service (``VISUAL_EMBED_URL``).
  3. Persists (file_id, page_number, image_path, embedding) into the
     ``visual_chunks`` pgvector table.

Everything is soft-fail by design — a broken sidecar, PyMuPDF raising
on a malformed PDF, or unreachable database must NEVER break the text
ingest path. When disabled (URL empty) the helper is a no-op and
imports are lazy.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import List, Optional

from app.config import (
    VISUAL_EMBED_URL,
    VISUAL_EMBED_TIMEOUT,
    VISUAL_PAGE_DPI,
    VISUAL_SCORE_THRESHOLD,
    VISUAL_STORAGE_ROOT,
    VISUAL_TEXT_EMBED_URL,
    logger,
)


def _visuals_enabled() -> bool:
    return bool(VISUAL_EMBED_URL)


def _page_number_from_path(p: Path) -> Optional[int]:
    """Parse ``prefix-3.png`` or ``prefix-003.png`` → 3. Returns None if unparseable."""
    try:
        return int(p.stem.rsplit("-", 1)[-1])
    except (ValueError, IndexError):
        return None


def _vector_literal(embedding: List[float]) -> str:
    """pgvector's text input format: '[0.1,0.2,...]' (no spaces to keep it compact)."""
    return "[" + ",".join(f"{v:.6f}" for v in embedding) + "]"


def render_pdf_pages(pdf_path: str, out_dir: Path, dpi: int) -> List[Path]:
    """Render each PDF page to ``out_dir/page-N.png`` via PyMuPDF.

    Switched away from `pdftoppm` so the container image doesn't need
    poppler-utils installed — PyMuPDF ships as a self-contained wheel.
    Raises RuntimeError on any failure so the caller can soft-fail.
    Kept as a sync function: called from a ThreadPoolExecutor.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError("PyMuPDF not installed") from exc

    # Zoom factor maps DPI onto the PDF's default 72-dpi coordinate space.
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    page_paths: List[Path] = []
    try:
        with fitz.open(pdf_path) as doc:
            # Zero-pad page numbers to match the pdftoppm naming convention
            # our callers (and tests) expect: page-01.png … page-NN.png
            width = max(2, len(str(doc.page_count)))
            for i in range(doc.page_count):
                page = doc.load_page(i)
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                out = out_dir / f"page-{str(i + 1).zfill(width)}.png"
                pix.save(str(out))
                page_paths.append(out)
    except Exception as exc:
        raise RuntimeError(f"PyMuPDF render failed: {exc}") from exc

    return sorted(page_paths)


def embed_image(image_path: Path) -> List[float]:
    """Synchronous HTTP call to the CLIP sidecar. Raises on error."""
    import requests

    with open(image_path, "rb") as f:
        resp = requests.post(
            VISUAL_EMBED_URL,
            files={"file": (image_path.name, f, "image/png")},
            timeout=VISUAL_EMBED_TIMEOUT,
        )
    resp.raise_for_status()
    payload = resp.json()
    embedding = payload.get("embedding")
    if not isinstance(embedding, list) or not embedding:
        raise RuntimeError(f"clip sidecar returned malformed payload: {payload!r}")
    return embedding


def embed_text_query(query: str) -> List[float]:
    """Call the CLIP text endpoint so the returned vector lives in the
    same space as image embeddings. Used by the /query visual path."""
    import requests

    url = VISUAL_TEXT_EMBED_URL or (
        VISUAL_EMBED_URL.replace("/embed/image", "/embed/text")
        if VISUAL_EMBED_URL and "/embed/image" in VISUAL_EMBED_URL
        else None
    )
    if not url:
        raise RuntimeError("VISUAL_TEXT_EMBED_URL not configured")
    resp = requests.post(url, json={"text": query}, timeout=VISUAL_EMBED_TIMEOUT)
    resp.raise_for_status()
    payload = resp.json()
    embedding = payload.get("embedding")
    if not isinstance(embedding, list) or not embedding:
        raise RuntimeError(f"clip sidecar returned malformed payload: {payload!r}")
    return embedding


async def persist_visual_chunk(
    pool,
    file_id: str,
    page_number: int,
    image_path: str,
    embedding: List[float],
    cmetadata: dict,
) -> None:
    """Upsert one row into ``visual_chunks``. Unique on (file_id, page_number)."""
    vec = _vector_literal(embedding)
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO visual_chunks (file_id, page_number, image_path, embedding, cmetadata)
            VALUES ($1, $2, $3, $4::vector, $5::jsonb)
            ON CONFLICT (file_id, page_number) DO UPDATE
              SET image_path = EXCLUDED.image_path,
                  embedding  = EXCLUDED.embedding,
                  cmetadata  = EXCLUDED.cmetadata
            """,
            file_id,
            page_number,
            image_path,
            vec,
            json.dumps(cmetadata),
        )


async def similarity_search_visual(
    pool,
    query_embedding: List[float],
    file_ids: List[str],
    k: int,
) -> List[dict]:
    """Cosine-similarity search over ``visual_chunks`` filtered by file_ids.

    Returns a list of dicts shaped as the /query response expects.
    Results below VISUAL_SCORE_THRESHOLD are dropped.
    """
    if not file_ids:
        return []
    vec = _vector_literal(query_embedding)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT file_id, page_number, image_path,
                   1 - (embedding <=> $1::vector) AS score
              FROM visual_chunks
             WHERE file_id = ANY($2::text[])
             ORDER BY embedding <=> $1::vector
             LIMIT $3
            """,
            vec,
            file_ids,
            k,
        )
    return [
        {
            "file_id": r["file_id"],
            "page_number": r["page_number"],
            "image_path": r["image_path"],
            "score": float(r["score"]),
        }
        for r in rows
        if float(r["score"]) >= VISUAL_SCORE_THRESHOLD
    ]


async def maybe_embed_visuals(
    file_path: str,
    file_id: str,
    file_ext: str,
    user_id: str,
    executor,
) -> int:
    """Entry point — call after the text pipeline has completed.

    Returns the number of pages successfully embedded (0 if disabled or
    soft-failed). Never raises.
    """
    if not _visuals_enabled():
        return 0
    if file_ext.lower() != "pdf":
        return 0

    out_dir = Path(VISUAL_STORAGE_ROOT) / file_id
    loop = asyncio.get_running_loop()

    # 1. Render pages
    try:
        page_paths = await loop.run_in_executor(
            executor, render_pdf_pages, file_path, out_dir, VISUAL_PAGE_DPI
        )
    except RuntimeError as exc:
        logger.warning("visual ingest: pdftoppm step failed for %s: %s", file_id, exc)
        return 0

    if not page_paths:
        logger.info("visual ingest: 0 pages rendered for %s", file_id)
        return 0

    # 2. Lazily grab a DB pool. If pgvector is unavailable (mongo deploy)
    # we still keep the PNGs — but skip persistence.
    try:
        from app.services.database import PSQLDatabase

        pool = await PSQLDatabase.get_pool()
    except Exception as exc:
        logger.warning("visual ingest: cannot get DB pool, skipping persistence: %s", exc)
        return 0

    # 3. Embed + persist each page
    persisted = 0
    for page_path in page_paths:
        page_num = _page_number_from_path(page_path)
        if page_num is None:
            continue
        try:
            embedding = await loop.run_in_executor(executor, embed_image, page_path)
        except Exception as exc:
            logger.warning(
                "visual ingest: embed failed for %s p%s: %s", file_id, page_num, exc
            )
            continue
        try:
            await persist_visual_chunk(
                pool,
                file_id=file_id,
                page_number=page_num,
                image_path=str(page_path),
                embedding=embedding,
                cmetadata={"user_id": user_id, "source": os.path.basename(file_path)},
            )
        except Exception as exc:
            logger.warning(
                "visual ingest: persist failed for %s p%s: %s", file_id, page_num, exc
            )
            continue
        persisted += 1

    logger.info("visual ingest: %d/%d pages embedded for %s", persisted, len(page_paths), file_id)
    return persisted


def cleanup_visual_storage(file_id: str) -> None:
    """Optional: delete the on-disk page PNGs for a file_id. Used by tests and rollback."""
    target = Path(VISUAL_STORAGE_ROOT) / file_id
    if target.exists() and target.is_dir():
        shutil.rmtree(target, ignore_errors=True)
