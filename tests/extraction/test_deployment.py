"""Contracts for the processor required by both standard RAG deployments."""

import json
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROCESSOR_REF = "77bcd2ec2e63be523bfab2dfa11159a2ceec0088"
PROCESSOR_CONTEXT = (
    f"https://github.com/LibreChat-AI/code-interpreter.git#{PROCESSOR_REF}"
)


def stage_compose_files(directory):
    for name in ("docker-compose.yaml", "api-compose.yaml", "extraction-compose.yaml"):
        shutil.copyfile(ROOT / name, directory / name)
    # Never read deployment credentials while rendering or qualifying the service.
    (directory / ".env").write_text("")
    return directory


def compose_command(directory, project, *files):
    command = ["docker", "compose", "--env-file", "/dev/null", "-p", project]
    for file in files:
        command.extend(["-f", str(directory / file)])
    return command


def render(directory, *files):
    command = compose_command(directory, f"rag-contract-{uuid.uuid4().hex}", *files)
    result = subprocess.run(
        [*command, "config", "--no-env-resolution", "--format", "json"],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    return json.loads(result.stdout)


def assert_hardened_processor(service):
    assert service["build"]["context"] == PROCESSOR_CONTEXT
    assert service["build"]["dockerfile"] == "packages/extraction/Dockerfile"
    assert service["build"]["target"] == "runtime"
    assert service["init"] is True and service["user"] == "10001:10001"
    assert service["read_only"] is True
    assert service["cap_drop"] == ["ALL"]
    assert service["security_opt"] == ["no-new-privileges:true"]
    assert service["network_mode"] == "none"
    assert int(service["mem_limit"]) == 768 * 1024 * 1024
    assert float(service["cpus"]) == 2 and int(service["pids_limit"]) == 32
    assert service["tmpfs"] == [
        "/jobs:rw,noexec,nosuid,nodev,size=96m,nr_inodes=1024,uid=10001,gid=10001,mode=0700"
    ]
    assert len(service["volumes"]) == 1
    mount = service["volumes"][0]
    assert mount["type"] == "volume" and mount["source"] == "extraction-socket"
    assert mount["target"] == "/socket" and not mount.get("read_only", False)
    for key in (
        "environment",
        "env_file",
        "ports",
        "expose",
        "profiles",
        "privileged",
        "cap_add",
        "command",
        "entrypoint",
    ):
        assert not service.get(key), key


@pytest.mark.parametrize(
    "recipe,caller",
    [("docker-compose.yaml", "fastapi"), ("api-compose.yaml", "rag_api")],
)
def test_standard_deployment_requires_one_hardened_processor(tmp_path, recipe, caller):
    config = render(stage_compose_files(tmp_path), recipe)
    assert_hardened_processor(config["services"]["extraction"])
    app = config["services"][caller]
    assert "10001" in app["group_add"]
    assert app["depends_on"]["extraction"]["condition"] == "service_healthy"
    mount = next(item for item in app["volumes"] if item["target"] == "/socket")
    assert (
        mount["type"] == "volume"
        and mount["source"] == "extraction-socket"
        and mount["read_only"] is True
    )
    assert not app.get("profiles")
    assert (
        not {"RAG_EXTRACTION_ENABLED", "RAG_EXTRACTION_FORMATS"}
        & app.get("environment", {}).keys()
    )


def test_combined_recipes_share_one_processor_and_socket(tmp_path):
    config = render(
        stage_compose_files(tmp_path), "docker-compose.yaml", "api-compose.yaml"
    )
    assert set(config["services"]) == {"db", "fastapi", "rag_api", "extraction"}
    assert_hardened_processor(config["services"]["extraction"])
    for caller in ("fastapi", "rag_api"):
        mount = next(
            item
            for item in config["services"][caller]["volumes"]
            if item["target"] == "/socket"
        )
        assert mount["source"] == "extraction-socket" and mount["read_only"] is True
