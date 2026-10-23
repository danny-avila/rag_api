"""Opt-in real client qualification against the shipped hardened processor image."""

import os
import subprocess
import time
import uuid
from pathlib import Path

import aiofiles
import pytest

pytestmark = pytest.mark.integration


def docker(*args, check=True):
    result = subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=40
    )
    if check:
        assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_python_client_to_hardened_processor():
    image = os.getenv("RAG_EXTRACTION_TEST_IMAGE")
    if not image:
        pytest.skip(
            "RAG_EXTRACTION_TEST_IMAGE must name a built Code Interpreter extraction runtime"
        )
    run_id = uuid.uuid4().hex
    name = f"rag-extraction-{run_id}"
    volume = f"rag-extraction-{run_id}"
    caller = f"rag-extraction-caller-{run_id}"
    root = Path(__file__).resolve().parents[2]
    client = root / "app/services/extraction_client.py"
    fixtures = Path(__file__).parent / "fixtures"
    library = Path(aiofiles.__file__).parent
    docker("volume", "create", volume)
    try:
        docker(
            "run",
            "-d",
            "--name",
            name,
            "--init",
            "--network",
            "none",
            "--read-only",
            "--user",
            "10001:10001",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--memory",
            "768m",
            "--cpus",
            "2",
            "--pids-limit",
            "32",
            "--tmpfs",
            "/jobs:rw,noexec,nosuid,nodev,size=96m,nr_inodes=1024,uid=10001,gid=10001,mode=0700",
            "--mount",
            f"type=volume,src={volume},dst=/socket",
            image,
        )
        for _ in range(30):
            status = docker(
                "inspect", "--format", "{{.State.Health.Status}}", name
            ).stdout.strip()
            if status == "healthy":
                break
            if status == "unhealthy":
                pytest.fail(
                    "Hardened extraction processor failed its isolation/health probe"
                )
            time.sleep(1)
        else:
            pytest.fail("Hardened extraction processor did not become healthy")
        script = """
import asyncio
import sys
sys.path.insert(0, "/")
from client import ExtractionClient, ExtractionSettings
async def check():
    client = ExtractionClient(ExtractionSettings(enabled=True))
    pdf = await client.extract("/fixtures/pages.pdf", "pdf")
    assert pdf == [
        {"kind": "page", "index": 1, "text": "First page text"},
        {"kind": "page", "index": 2, "text": "Second page text"},
    ], pdf
    docx = await client.extract("/fixtures/body.docx", "docx")
    assert len(docx) == 1 and docx[0]["kind"] == "document" and docx[0]["index"] == 1
    text = docx[0]["text"]
    assert text.index("First paragraph") < text.index("Left cell") < text.index("Right cell") < text.index("Last paragraph")
asyncio.run(check())
print("Python client to hardened PDF/DOCX processor passed")
"""
        result = docker(
            "run",
            "--rm",
            "--name",
            caller,
            "--init",
            "--network",
            "none",
            "--read-only",
            "--user",
            "10001:10001",
            "--group-add",
            "10001",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--mount",
            f"type=volume,src={volume},dst=/socket,readonly",
            "--mount",
            f"type=bind,src={client},dst=/client.py,readonly",
            "--mount",
            f"type=bind,src={fixtures},dst=/fixtures,readonly",
            "--mount",
            f"type=bind,src={library},dst=/usr/local/lib/python3.13/site-packages/aiofiles,readonly",
            image,
            "python",
            "-B",
            "-c",
            script,
        )
        assert "processor passed" in result.stdout
        assert (
            docker(
                "exec",
                name,
                "node",
                "-e",
                'if(require("node:fs").readdirSync("/jobs").length)process.exit(1)',
            ).returncode
            == 0
        )
    finally:
        docker("rm", "-f", caller, check=False)
        docker("rm", "-f", name, check=False)
        docker("volume", "rm", volume, check=False)
