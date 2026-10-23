"""Current Python client qualification against the default hardened deployment."""

import json
import subprocess
import uuid
from pathlib import Path

import aiofiles
import pytest

from tests.extraction.test_deployment import (
    assert_hardened_processor,
    compose_command,
    render,
    stage_compose_files,
)

pytestmark = pytest.mark.integration


def docker(*args, check=True):
    result = subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=40
    )
    if check:
        assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_python_client_to_default_hardened_processor(tmp_path):
    directory = stage_compose_files(tmp_path)
    assert_hardened_processor(
        render(directory, "docker-compose.yaml")["services"]["extraction"]
    )
    run_id = uuid.uuid4().hex
    project = f"rag-extraction-{run_id}"
    volume = f"{project}_extraction-socket"
    caller = f"rag-extraction-caller-{run_id}"
    compose = compose_command(directory, project, "docker-compose.yaml")
    image = f"{project}-extraction"

    def run_compose(*args, check=True, timeout=40):
        result = subprocess.run(
            [*compose, *args], capture_output=True, text=True, timeout=timeout
        )
        if check:
            assert result.returncode == 0, result.stdout + result.stderr
        return result

    root = Path(__file__).resolve().parents[2]
    client = root / "app/services/extraction_client.py"
    fixtures = Path(__file__).parent / "fixtures"
    library = Path(aiofiles.__file__).parent
    try:
        run_compose("build", "extraction", timeout=240)
        run_compose("up", "-d", "--wait", "--wait-timeout", "30", "extraction")
        name = run_compose("ps", "-q", "extraction").stdout.strip()
        image = docker("inspect", "--format", "{{.Config.Image}}", name).stdout.strip()
        runtime = json.loads(docker("inspect", name).stdout)[0]
        host = runtime["HostConfig"]
        assert (
            host["NetworkMode"] == "none"
            and host["ReadonlyRootfs"]
            and not host["Privileged"]
        )
        assert (
            host["CapDrop"] == ["ALL"]
            and "no-new-privileges:true" in host["SecurityOpt"]
        )
        assert runtime["Config"]["User"] == "10001:10001"
        assert host["Memory"] == 768 * 1024 * 1024 and host["NanoCpus"] == 2 * 10**9
        assert host["PidsLimit"] == 32 and set(host["Tmpfs"]) == {"/jobs"}
        assert (
            len(runtime["Mounts"]) == 1
            and runtime["Mounts"][0]["Destination"] == "/socket"
        )
        script = """
import asyncio
import errno
import os
import stat
import sys
sys.path.insert(0, "/")
from client import ExtractionClient, ExtractionSettings
async def check():
    settings = ExtractionSettings.from_env()
    assert settings.socket == "/socket/extraction.sock"
    assert stat.S_IMODE(os.stat("/socket").st_mode) == 0o770
    assert stat.S_IMODE(os.stat(settings.socket).st_mode) == 0o660
    try:
        open("/socket/not-writable", "wb")
    except OSError as error:
        assert error.errno == errno.EROFS
    else:
        raise AssertionError("caller socket mount is writable")
    client = ExtractionClient(settings)
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
            "65532:65532",
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
        run_compose("down", "-v", "--remove-orphans", check=False)
        if image:
            docker("image", "rm", image, check=False)
