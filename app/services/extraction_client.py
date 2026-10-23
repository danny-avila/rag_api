"""Bounded v1 extraction transport. Only caller-owned bytes cross the socket."""

import asyncio
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass

import aiofiles

INPUT_BYTES = 10 * 1024 * 1024
TEXT_BYTES = 1024 * 1024
RESULT_BYTES = 3 * 1024 * 1024
HEADER_BYTES = 4096
CAPABILITY_BYTES = 8192
CHUNK_BYTES = 64 * 1024
OPERATION = "document.extract-text"
ERROR_STATUS = {
    "INVALID_REQUEST": 400,
    "INPUT_LIMIT": 413,
    "OUTPUT_LIMIT": 413,
    "DECOMPRESSION_LIMIT": 413,
    "STRUCTURE_LIMIT": 413,
    "BUSY": 429,
    "UNAVAILABLE": 503,
    "DEADLINE": 504,
    **dict.fromkeys(
        (
            "CANCELLED",
            "INVALID_DOCUMENT",
            "EMPTY_OUTPUT",
            "ENCRYPTED_DOCUMENT",
            "UNSUPPORTED_ENCODING",
            "UNSUPPORTED_CONTENT",
            "RESOURCE_LIMIT",
        ),
        422,
    ),
}


class ExtractionError(Exception):
    def __init__(self, code: str, status_code: int = None):
        self.code = code
        self.status_code = status_code or ERROR_STATUS.get(code, 502)
        super().__init__(code)


@dataclass(frozen=True)
class ExtractionSettings:
    enabled: bool = False
    socket: str = "/socket/extraction.sock"
    deadline: float = 10.0
    formats: frozenset[str] = frozenset({"pdf", "docx"})

    def __post_init__(self):
        if not self.enabled:
            return
        if not os.path.isabs(self.socket) or "\x00" in self.socket:
            raise ValueError(
                "RAG_EXTRACTION_SOCKET must be an absolute Unix socket path"
            )
        if not math.isfinite(self.deadline) or not 0 < self.deadline <= 10:
            raise ValueError("RAG_EXTRACTION_DEADLINE_SECONDS must be in (0, 10]")
        if not self.formats or not self.formats <= {"pdf", "docx"}:
            raise ValueError("RAG_EXTRACTION_FORMATS must select pdf and/or docx")

    @classmethod
    def from_env(cls):
        enabled = os.getenv("RAG_EXTRACTION_ENABLED", "false").lower() in (
            "true",
            "1",
            "yes",
            "on",
        )
        if not enabled:
            return cls()
        return cls(
            enabled=True,
            socket=os.getenv("RAG_EXTRACTION_SOCKET", "/socket/extraction.sock"),
            deadline=float(os.getenv("RAG_EXTRACTION_DEADLINE_SECONDS", "10")),
            formats=frozenset(
                part.strip().lower()
                for part in os.getenv("RAG_EXTRACTION_FORMATS", "pdf,docx").split(",")
            ),
        )


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def _json(body: bytes):
    try:
        return json.loads(
            body.decode("utf-8", "strict"),
            object_pairs_hook=_object,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("constant")),
        )
    except (UnicodeError, ValueError, RecursionError):
        raise ExtractionError("INVALID_RESPONSE") from None


def _keys(value, expected):
    return type(value) is dict and value.keys() == set(expected)


def _integer(value, minimum, maximum):
    return type(value) is int and minimum <= value <= maximum


def validate_capabilities(value, format: str, size: int):
    if not _keys(
        value, ("version", "operation", "formats", "structure", "limitations", "limits")
    ):
        raise ExtractionError("INCOMPATIBLE_SERVICE")
    if (
        type(value["version"]) is not int
        or value["version"] != 1
        or value["operation"] != OPERATION
        or type(value["formats"]) is not list
        or not value["formats"]
        or any(
            type(item) is not str or item not in {"pdf", "docx"}
            for item in value["formats"]
        )
        or len(set(value["formats"])) != len(value["formats"])
        or format not in value["formats"]
    ):
        raise ExtractionError("INCOMPATIBLE_SERVICE")
    for key in ("structure", "limitations"):
        if not _keys(value[key], value["formats"]):
            raise ExtractionError("INCOMPATIBLE_SERVICE")
        for text in value[key].values():
            try:
                if (
                    type(text) is not str
                    or not text.strip()
                    or len(text.encode("utf-8")) > 2048
                ):
                    raise ValueError()
            except (ValueError, UnicodeError):
                raise ExtractionError("INCOMPATIBLE_SERVICE") from None
    limits = value["limits"]
    maximums = {
        "inputBytes": INPUT_BYTES,
        "textBytes": TEXT_BYTES,
        "resultBytes": RESULT_BYTES,
        "deadlineMs": 10_000,
        "maxSegments": 128,
        "concurrency": 2,
        "expandedBytes": 32 * 1024 * 1024,
        "archiveEntries": 512,
        "tableColumns": 512,
    }
    if not _keys(limits, maximums) or any(
        not _integer(limits[key], 1, maximum) for key, maximum in maximums.items()
    ):
        raise ExtractionError("INCOMPATIBLE_SERVICE")
    if size > limits["inputBytes"]:
        raise ExtractionError("INPUT_LIMIT")
    return limits


def validate_result(value, format: str, digest: str, limits):
    if not _keys(
        value,
        ("version", "operation", "format", "inputSha256", "textBytes", "segments"),
    ):
        raise ExtractionError("INVALID_RESPONSE")
    if (
        type(value["version"]) is not int
        or value["version"] != 1
        or value["operation"] != OPERATION
        or value["format"] != format
        or value["inputSha256"] != digest
        or not _integer(value["textBytes"], 1, limits["textBytes"])
        or type(value["segments"]) is not list
        or not 1
        <= len(value["segments"])
        <= (limits["maxSegments"] if format == "pdf" else 1)
    ):
        raise ExtractionError("INVALID_RESPONSE")
    size = 0
    nonempty = False
    for index, segment in enumerate(value["segments"], 1):
        if (
            not _keys(segment, ("kind", "index", "text"))
            or segment["kind"] != ("page" if format == "pdf" else "document")
            or type(segment["index"]) is not int
            or segment["index"] != index
            or type(segment["text"]) is not str
        ):
            raise ExtractionError("INVALID_RESPONSE")
        try:
            size += len(segment["text"].encode("utf-8", "strict"))
        except UnicodeError:
            raise ExtractionError("INVALID_RESPONSE") from None
        nonempty |= bool(segment["text"].strip())
    if size != value["textBytes"] or size > limits["textBytes"] or not nonempty:
        raise ExtractionError("INVALID_RESPONSE")
    return value["segments"]


class ExtractionClient:
    def __init__(self, settings: ExtractionSettings):
        self.settings = settings
        self._admitted = 0

    async def _request(
        self, method: str, *, file=None, size=0, format=None, maximum=CAPABILITY_BYTES
    ):
        reader, writer = await asyncio.open_unix_connection(
            self.settings.socket, limit=HEADER_BYTES
        )
        # At most one chunk may be queued without drain, including on cancellation.
        writer.transport.set_write_buffer_limits(high=CHUNK_BYTES, low=CHUNK_BYTES // 2)
        digest = hashlib.sha256()
        try:
            path = "/v1/extract-text" if file is not None else "/v1/capabilities"
            headers = (
                f"{method} {path} HTTP/1.1\r\nHost: extraction\r\nConnection: close\r\n"
            )
            if file is not None:
                headers += (
                    f"Content-Length: {size}\r\nContent-Type: application/octet-stream\r\n"
                    f"X-Extraction-Version: 1\r\nX-Extraction-Format: {format}\r\n"
                )
            writer.write((headers + "\r\n").encode("ascii"))
            await writer.drain()
            remaining = size
            while remaining:
                chunk = await file.read(min(CHUNK_BYTES, remaining))
                if not chunk:
                    raise ExtractionError("INVALID_REQUEST")
                digest.update(chunk)
                writer.write(chunk)
                await writer.drain()
                remaining -= len(chunk)
            if file is not None and await file.read(1):
                raise ExtractionError("INPUT_LIMIT")
            try:
                header = await reader.readuntil(b"\r\n\r\n")
            except (asyncio.LimitOverrunError, asyncio.IncompleteReadError):
                raise ExtractionError("INVALID_RESPONSE") from None
            if len(header) > HEADER_BYTES:
                raise ExtractionError("INVALID_RESPONSE")
            lines = header[:-4].split(b"\r\n")
            if not re.fullmatch(rb"HTTP/1\.1 [1-5][0-9]{2} [\x20-\x7e]*", lines[0]):
                raise ExtractionError("INVALID_RESPONSE")
            status = int(lines[0][9:12])
            fields = {}
            for line in lines[1:]:
                key, sep, val = line.partition(b":")
                key = key.lower()
                if not sep or not re.fullmatch(rb"[a-z0-9-]+", key) or key in fields:
                    raise ExtractionError("INVALID_RESPONSE")
                if any(byte < 32 or byte > 126 for byte in val):
                    raise ExtractionError("INVALID_RESPONSE")
                fields[key] = val.strip()
            length = fields.get(b"content-length", b"")
            if (
                not re.fullmatch(rb"[1-9][0-9]{0,7}", length)
                or int(length) > maximum
                or fields.get(b"content-type") != b"application/json; charset=utf-8"
                or fields.get(b"connection", b"").lower() != b"close"
                or b"transfer-encoding" in fields
                or b"content-encoding" in fields
            ):
                raise ExtractionError("INVALID_RESPONSE")
            try:
                body = await reader.readexactly(int(length))
                if await reader.read(1):
                    raise ExtractionError("INVALID_RESPONSE")
            except asyncio.IncompleteReadError:
                raise ExtractionError("INVALID_RESPONSE") from None
            value = _json(body)
            if status != 200:
                if (
                    _keys(value, ("error",))
                    and _keys(value["error"], ("code",))
                    and type(value["error"]["code"]) is str
                    and ERROR_STATUS.get(value["error"]["code"]) == status
                ):
                    raise ExtractionError(value["error"]["code"])
                raise ExtractionError("INVALID_RESPONSE")
            return value, digest.hexdigest()
        finally:
            # Abort instead of flushing pending input on timeout/cancellation.
            writer.transport.abort()

    async def extract(self, file_path: str, format: str):
        if format not in self.settings.formats or format not in {"pdf", "docx"}:
            raise ExtractionError("UNSUPPORTED_CONTENT")
        if self._admitted >= 2:
            raise ExtractionError("BUSY")
        self._admitted += 1
        try:
            async with asyncio.timeout(self.settings.deadline):
                async with aiofiles.open(file_path, "rb") as file:
                    stat = await asyncio.to_thread(os.fstat, file.fileno())
                    if not 0 < stat.st_size <= INPUT_BYTES:
                        raise ExtractionError("INPUT_LIMIT")
                    # No cache: each job checks the current socket's contract before upload.
                    capabilities, _ = await self._request("GET")
                    limits = validate_capabilities(capabilities, format, stat.st_size)
                    result, digest = await self._request(
                        "POST",
                        file=file,
                        size=stat.st_size,
                        format=format,
                        maximum=limits["resultBytes"],
                    )
                    return validate_result(result, format, digest, limits)
        except TimeoutError:
            raise ExtractionError("DEADLINE") from None
        except OSError:
            raise ExtractionError("UNAVAILABLE") from None
        finally:
            self._admitted -= 1
