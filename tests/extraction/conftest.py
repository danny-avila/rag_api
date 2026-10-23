import asyncio
import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path

import pytest

from app.services.extraction_client import ExtractionClient, ExtractionSettings

CAPABILITIES = {
    "version": 1,
    "operation": "document.extract-text",
    "formats": ["pdf", "docx"],
    "structure": {
        "pdf": "ordered pages, one-based index",
        "docx": "document body, paragraphs and tables",
    },
    "limitations": {"pdf": "Text only; no OCR", "docx": "Body only"},
    "limits": {
        "inputBytes": 10 * 1024 * 1024,
        "textBytes": 1024 * 1024,
        "resultBytes": 3 * 1024 * 1024,
        "deadlineMs": 10000,
        "maxSegments": 128,
        "concurrency": 2,
        "expandedBytes": 32 * 1024 * 1024,
        "archiveEntries": 512,
        "tableColumns": 512,
    },
}


class Processor:
    def __init__(self, socket):
        self.socket = socket
        self.capabilities = copy.deepcopy(CAPABILITIES)
        self.requests = []
        self.mutate = lambda value: value
        self.raw = None
        self.error = None
        self.hold = False
        self.before_response = None
        self.entered = asyncio.Event()
        self.closed = asyncio.Event()
        self.active = set()
        self.texts = ["First\x00 page café", "Second page"]

    def client(self, deadline=10):
        return ExtractionClient(
            ExtractionSettings(socket=self.socket, deadline=deadline)
        )

    async def handle(self, reader, writer):
        task = asyncio.current_task()
        self.active.add(task)
        try:
            headers = (await reader.readuntil(b"\r\n\r\n")).decode("ascii")
            lines = headers.split("\r\n")
            method, path, _ = lines[0].split()
            fields = dict(line.split(": ", 1) for line in lines[1:] if line)
            length = int(fields.get("Content-Length", "0"))
            body = await reader.readexactly(length)
            self.requests.append((method, path, fields, body))
            if path == "/v1/capabilities":
                value = self.capabilities
                status = 200
            else:
                self.entered.set()
                if self.hold:
                    await reader.read()
                    self.closed.set()
                    return
                format = fields["X-Extraction-Format"]
                texts = (
                    self.texts if format == "pdf" else ["Body paragraph\nLeft\tRight"]
                )
                value = self.mutate(
                    {
                        "version": 1,
                        "operation": "document.extract-text",
                        "format": format,
                        "inputSha256": hashlib.sha256(body).hexdigest(),
                        "textBytes": sum(len(text.encode("utf-8")) for text in texts),
                        "segments": [
                            {
                                "kind": "page" if format == "pdf" else "document",
                                "index": i,
                                "text": text,
                            }
                            for i, text in enumerate(texts, 1)
                        ],
                    }
                )
                status = 200
                if self.error:
                    status, code = self.error
                    value = {"error": {"code": code}}
                if self.raw is not None:
                    writer.write(self.raw)
                    await writer.drain()
                    return
                if self.before_response is not None:
                    self.before_response()
            wire = json.dumps(value).encode("utf-8")
            writer.write(
                f"HTTP/1.1 {status} Result\r\nContent-Type: application/json; charset=utf-8\r\n"
                f"Content-Length: {len(wire)}\r\nConnection: close\r\n\r\n".encode(
                    "ascii"
                )
                + wire
            )
            await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionError:
                pass
            self.active.discard(task)


@pytest.fixture
async def processor():
    # Keep AF_UNIX path below 108 bytes even in long worktree/pytest directories.
    with tempfile.TemporaryDirectory(
        prefix="rag-", dir=os.environ.get("TMPDIR")
    ) as directory:
        service = Processor(str(Path(directory, "s")))
        server = await asyncio.start_unix_server(service.handle, path=service.socket)
        async with server:
            yield service
        for task in list(service.active):
            task.cancel()
        await asyncio.gather(*service.active, return_exceptions=True)


@pytest.fixture
def input_file(tmp_path):
    file = tmp_path / "input"
    file.write_bytes(b"caller-owned opaque document bytes")
    return str(file)
