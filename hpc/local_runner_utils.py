"""Shared utilities for local Ray/vLLM runners.

This module consolidates common code used by:
- eval/local/run_eval.py
- data/local/run_tracegen.py

It provides managed subprocess handling for Ray clusters and vLLM servers,
datagen config parsing, Docker runtime setup, and Harbor command building.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from hpc.vllm_utils import _build_vllm_cli_args


@dataclass
class ManagedProcess:
    """A subprocess with graceful shutdown support."""

    name: str
    proc: subprocess.Popen
    _log_handle: Optional[object] = field(default=None, repr=False)

    def stop(self, timeout: float = 10.0) -> None:
        """Stop the process gracefully, falling back to kill if needed."""
        if self.proc.poll() is not None:
            return
        try:
            self.proc.terminate()
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        finally:
            if self._log_handle:
                try:
                    self._log_handle.close()
                except Exception:
                    pass


def maybe_int(value: object) -> Optional[int]:
    """Parse a value as int, returning None if not possible."""
    if value in (None, "", "None"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _open_log_file(log_path: Optional[Path]) -> tuple:
    """Open a log file with line buffering for real-time tail access.

    Returns:
        Tuple of (stdout_dest, stderr_dest, log_file_handle)
    """
    if log_path:
        log_file = open(log_path, "w", encoding="utf-8", buffering=1)
        return log_file, log_file, log_file
    return None, None, None


def start_ray(
    host: str,
    ray_port: int,
    num_gpus: int,
    num_cpus: int,
    log_path: Optional[Path] = None,
) -> ManagedProcess:
    """Start a single-node Ray cluster head.

    Args:
        host: IP address to bind to
        ray_port: Port for Ray head node
        num_gpus: Number of GPUs to expose
        num_cpus: Number of CPUs to expose
        log_path: Optional path for Ray logs (line-buffered)

    Returns:
        ManagedProcess wrapping the Ray head process
    """
    cmd = [
        "ray",
        "start",
        "--head",
        f"--node-ip-address={host}",
        f"--port={ray_port}",
        f"--num-gpus={num_gpus}",
        f"--num-cpus={num_cpus}",
        "--dashboard-host=0.0.0.0",
        "--block",
    ]

    env = os.environ.copy()
    stdout, stderr, log_file = _open_log_file(log_path)

    popen = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, env=env)
    process = ManagedProcess(name="ray", proc=popen, _log_handle=log_file)
    return process


def start_vllm_controller(
    model: str,
    host: str,
    ray_port: int,
    api_port: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    data_parallel_size: int,
    endpoint_path: Path,
    controller_script: Path,
    log_path: Optional[Path] = None,
    served_model_name: Optional[str] = None,
    extra_cli_args: Optional[List[str]] = None,
    extra_env_vars: Optional[dict] = None,
) -> ManagedProcess:
    """Start a vLLM controller process.

    Args:
        model: Model path/name for vLLM
        host: IP address to bind to
        ray_port: Ray head port to connect to
        api_port: Port for vLLM OpenAI-compatible API
        tensor_parallel_size: Number of GPUs for tensor parallelism
        pipeline_parallel_size: Number of pipeline stages
        data_parallel_size: Number of data parallel replicas
        endpoint_path: Path to write endpoint JSON
        controller_script: Path to start_vllm_ray_controller.py
        log_path: Optional path for vLLM logs (line-buffered)
        served_model_name: Optional custom model name for the API
        extra_cli_args: Additional CLI args to pass to vLLM
        extra_env_vars: Additional environment variables

    Returns:
        ManagedProcess wrapping the vLLM controller process
    """
    env = os.environ.copy()
    env["VLLM_MODEL_PATH"] = model
    env["PYTHONUNBUFFERED"] = "1"  # Ensure real-time log output

    if extra_env_vars:
        env.update(extra_env_vars)

    cmd = [
        sys.executable,
        str(controller_script),
        "--ray-address",
        f"{host}:{ray_port}",
        "--host",
        host,
        "--port",
        str(api_port),
        "--model",
        model,
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--pipeline-parallel-size",
        str(pipeline_parallel_size),
        "--data-parallel-size",
        str(data_parallel_size),
        "--endpoint-json",
        str(endpoint_path),
    ]

    if served_model_name:
        cmd.extend(["--served-model-name", served_model_name])

    if extra_cli_args:
        cmd.extend(extra_cli_args)

    stdout, stderr, log_file = _open_log_file(log_path)

    popen = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, env=env)
    process = ManagedProcess(name="vllm_controller", proc=popen, _log_handle=log_file)
    return process


def wait_for_endpoint(
    endpoint_path: Path,
    controller: ManagedProcess,
    timeout: int = 300,
) -> None:
    """Wait for the vLLM endpoint JSON file to be created.

    Args:
        endpoint_path: Path to the endpoint JSON file
        controller: The vLLM controller process to monitor
        timeout: Maximum seconds to wait

    Raises:
        RuntimeError: If the controller exits before creating the endpoint
        TimeoutError: If the endpoint is not created within timeout
    """
    start = time.time()
    while time.time() - start < timeout:
        if controller.proc.poll() is not None:
            raise RuntimeError(
                "vLLM controller exited before writing the endpoint JSON. Check logs."
            )
        if endpoint_path.exists():
            return
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for endpoint JSON at {endpoint_path}")


def terminate_processes(processes: List[ManagedProcess]) -> None:
    """Terminate a list of managed processes in order."""
    for proc in processes:
        try:
            proc.stop()
        except Exception:
            pass


def extract_agent_kwargs_from_config(harbor_config: dict, agent_name: str) -> dict:
    """Extract kwargs for the specified agent from harbor config.

    The Harbor YAML is the ground truth for agent configuration. This function
    finds the agent by name and returns a copy of its kwargs dict.

    Args:
        harbor_config: Parsed harbor config dict (from YAML)
        agent_name: Name of the agent to find (e.g., "terminus-2")

    Returns:
        Copy of the agent's kwargs dict, or empty dict if not found
    """
    agents = harbor_config.get("agents", [])
    for agent in agents:
        if agent.get("name") == agent_name:
            return copy.deepcopy(agent.get("kwargs", {}))
    # Fallback: return first agent's kwargs if no match (backwards compat)
    if agents and isinstance(agents[0], dict):
        return copy.deepcopy(agents[0].get("kwargs", {}))
    return {}


# ---------------------------------------------------------------------------
# Harbor command building utilities
# ---------------------------------------------------------------------------


def _timestamp() -> str:
    """Generate a timestamp string for job names."""
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def default_job_name(prefix: str, dataset_label: str, model_label: str) -> str:
    """Generate a default job name.

    Args:
        prefix: Job type prefix (e.g., "eval", "tracegen")
        dataset_label: Dataset/tasks identifier
        model_label: Model identifier

    Returns:
        Formatted job name like "eval-dataset-model-20240101_120000"
    """
    sanitized_dataset = Path(dataset_label).name.replace("/", "-").replace(" ", "_")
    sanitized_model = model_label.replace("/", "-").replace(" ", "_")
    return f"{prefix}-{sanitized_dataset}-{sanitized_model}-{_timestamp()}"


def apply_nested_key(target: dict, dotted_key: str, value: Any) -> None:
    """Apply a value to a nested dict using dotted key notation.

    Args:
        target: Dict to modify in-place
        dotted_key: Key like "model_info.max_tokens" for nested access
        value: Value to set
    """
    parts = dotted_key.split(".")
    cursor = target
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def parse_agent_kwarg_strings(entries: List[str]) -> Tuple[Dict[str, Any], List[str]]:
    """Parse --agent-kwarg CLI entries into overrides and passthrough.

    Args:
        entries: List of "key=value" strings (or passthrough entries without =)

    Returns:
        Tuple of (overrides dict, passthrough list)
    """
    overrides: Dict[str, Any] = {}
    passthrough: List[str] = []
    for entry in entries:
        if "=" not in entry:
            passthrough.append(entry)
            continue
        key, raw_value = entry.split("=", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if not key:
            passthrough.append(entry)
            continue
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        overrides[key] = value
    return overrides, passthrough


def serialize_agent_kwargs(kwargs: dict) -> List[str]:
    """Serialize agent kwargs dict to CLI argument strings.

    Args:
        kwargs: Dict of agent kwargs

    Returns:
        List of "key=value" strings suitable for --agent-kwarg
    """
    serialized: List[str] = []
    for key, value in kwargs.items():
        if isinstance(value, (dict, list)):
            serialized.append(f"{key}={json.dumps(value)}")
        else:
            serialized.append(f"{key}={value}")
    return serialized


def build_harbor_command(
    harbor_binary: str,
    harbor_config_path: str,
    harbor_config_data: dict,
    job_name: str,
    agent_name: str,
    model_name: str,
    env_type: str,
    n_concurrent: int,
    n_attempts: int,
    endpoint_meta: dict,
    agent_kwarg_overrides: List[str],
    harbor_extra_args: List[str],
    dataset_slug: Optional[str] = None,
    dataset_path: Optional[str] = None,
) -> List[str]:
    """Build the harbor jobs start command.

    The Harbor YAML is the ground truth for agent configuration. This function:
    1. Extracts ALL kwargs from the Harbor YAML for the specified agent
    2. Overrides with endpoint-specific values (api_base, metrics_endpoint)
    3. Applies CLI --agent-kwarg overrides with highest precedence

    Args:
        harbor_binary: Path to harbor CLI
        harbor_config_path: Path to harbor config YAML
        harbor_config_data: Parsed harbor config dict
        job_name: Name for this harbor job
        agent_name: Agent to run (e.g., "terminus-2")
        model_name: Model identifier for --model flag
        env_type: Environment type for --env flag (daytona, docker, modal)
        n_concurrent: Number of concurrent trials
        n_attempts: Number of attempts per task
        endpoint_meta: Dict with api_base and metrics_endpoint from vLLM
        agent_kwarg_overrides: Raw --agent-kwarg strings from CLI
        harbor_extra_args: Additional args to pass through to harbor
        dataset_slug: Harbor dataset slug (mutually exclusive with dataset_path)
        dataset_path: Path to tasks directory (mutually exclusive with dataset_slug)

    Returns:
        Complete harbor command as list of strings
    """
    # Build agent kwargs - start with ALL kwargs from Harbor YAML as ground truth
    agent_kwargs = extract_agent_kwargs_from_config(harbor_config_data, agent_name)

    # Override with endpoint-specific values
    if endpoint_meta.get("metrics_endpoint"):
        agent_kwargs["metrics_endpoint"] = endpoint_meta["metrics_endpoint"]
    if endpoint_meta.get("api_base"):
        agent_kwargs["api_base"] = endpoint_meta["api_base"]

    # CLI --agent-kwarg flags take highest precedence (supports dotted keys)
    override_kwargs, passthrough = parse_agent_kwarg_strings(agent_kwarg_overrides)
    for dotted_key, override_value in override_kwargs.items():
        apply_nested_key(agent_kwargs, dotted_key, override_value)

    # Build base command
    cmd = [
        harbor_binary,
        "jobs",
        "start",
        "--config",
        harbor_config_path,
        "--job-name",
        job_name,
        "--agent",
        agent_name,
        "--model",
        model_name,
        "--env",
        env_type,
        "--n-concurrent",
        str(n_concurrent),
        "--n-attempts",
        str(n_attempts),
    ]

    # Add dataset (slug or path)
    if dataset_slug:
        cmd.extend(["--dataset", dataset_slug])
    elif dataset_path:
        cmd.extend(["-p", dataset_path])

    # Add serialized agent kwargs
    for kw in serialize_agent_kwargs(agent_kwargs):
        cmd.extend(["--agent-kwarg", kw])
    for passthrough_kw in passthrough:
        cmd.extend(["--agent-kwarg", passthrough_kw])

    # Process extra args with sensible defaults
    extra_args = list(harbor_extra_args or [])

    def _flag_present(flag: str) -> bool:
        return any(arg == flag or arg.startswith(f"{flag}=") for arg in extra_args)

    if not (_flag_present("--export-traces") or _flag_present("--no-export-traces")):
        extra_args.append("--export-traces")
    if not (_flag_present("--export-verifier-metadata") or _flag_present("--no-export-verifier-metadata")):
        extra_args.append("--export-verifier-metadata")
    if not _flag_present("--export-episodes"):
        extra_args.extend(["--export-episodes", "last"])

    for extra in extra_args:
        cmd.append(extra)

    return cmd


# ---------------------------------------------------------------------------
# Shared utilities for local runners (eval + tracegen)
# ---------------------------------------------------------------------------

# REPO_ROOT is needed for health check script path - computed lazily
_REPO_ROOT: Optional[Path] = None


def _get_repo_root() -> Path:
    """Get the repository root directory."""
    global _REPO_ROOT
    if _REPO_ROOT is None:
        # This file is at hpc/local_runner_utils.py, so repo root is parent
        _REPO_ROOT = Path(__file__).resolve().parent.parent
    return _REPO_ROOT


def resolve_jobs_dir_path(jobs_dir_value: Optional[str], repo_root: Optional[Path] = None) -> Path:
    """Resolve jobs_dir from harbor config to an absolute path.

    Args:
        jobs_dir_value: The jobs_dir value from harbor config (or None)
        repo_root: Repository root path (defaults to auto-detected)

    Returns:
        Absolute path to jobs directory
    """
    if repo_root is None:
        repo_root = _get_repo_root()
    raw_value = jobs_dir_value or "jobs"
    path = Path(raw_value)
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    return path


def run_endpoint_health_check(
    endpoint_json: Path,
    max_attempts: int,
    retry_delay: int,
    repo_root: Optional[Path] = None,
) -> None:
    """Run the vLLM endpoint health check script.

    Args:
        endpoint_json: Path to the endpoint JSON file
        max_attempts: Maximum number of health check attempts
        retry_delay: Delay in seconds between attempts
        repo_root: Repository root path (defaults to auto-detected)

    Raises:
        subprocess.CalledProcessError: If health check fails
    """
    if repo_root is None:
        repo_root = _get_repo_root()

    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "vllm" / "wait_for_endpoint.py"),
        "--endpoint-json",
        str(endpoint_json),
        "--max-attempts",
        str(max_attempts),
        "--retry-delay",
        str(retry_delay),
        "--health-path",
        "v1/models",
    ]
    subprocess.run(cmd, check=True)


def load_endpoint_metadata(endpoint_json: Path) -> Dict[str, Any]:
    """Load and parse vLLM endpoint metadata from JSON file.

    Computes api_base and metrics_endpoint URLs from the endpoint_url.

    Args:
        endpoint_json: Path to the endpoint JSON file

    Returns:
        Dict with endpoint data plus computed api_base and metrics_endpoint
    """
    data = json.loads(endpoint_json.read_text())
    base_url = (data.get("endpoint_url") or "").rstrip("/")
    api_base = f"{base_url}/v1" if base_url else ""
    metrics = base_url.rstrip("/")
    if metrics.endswith("/v1"):
        metrics = metrics[:-3].rstrip("/")
    metrics = f"{metrics}/metrics" if metrics else ""
    data["api_base"] = api_base
    data["metrics_endpoint"] = metrics
    return data


def apply_datagen_defaults(args: argparse.Namespace) -> None:
    """Load datagen config and apply defaults to args.

    Extracts vLLM settings from datagen config YAML and sets:
    - args.model (if not set)
    - args.tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    - args.ray_port, api_port
    - args._vllm_cli_args, args._vllm_env_vars

    Args:
        args: Parsed argparse namespace with datagen_config attribute
    """
    args._vllm_cli_args: List[str] = []
    args._vllm_env_vars: Dict[str, str] = {}

    datagen_config = getattr(args, "datagen_config", None)
    if not datagen_config:
        return

    cfg_path = Path(datagen_config).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Datagen config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as handle:
        datagen_cfg = yaml.safe_load(handle) or {}
    args.datagen_config = str(cfg_path)

    engine_cfg = datagen_cfg.get("engine") or {}
    backend_cfg = datagen_cfg.get("backend") or {}
    vllm_cfg = datagen_cfg.get("vllm_server") or {}

    # Model path
    if getattr(args, "model", None) is None:
        args.model = vllm_cfg.get("model_path") or engine_cfg.get("model")

    # Parallelism settings
    tp_default = maybe_int(vllm_cfg.get("tensor_parallel_size")) or maybe_int(
        backend_cfg.get("tensor_parallel_size")
    )
    pp_default = maybe_int(vllm_cfg.get("pipeline_parallel_size")) or maybe_int(
        backend_cfg.get("pipeline_parallel_size")
    )
    dp_default = maybe_int(vllm_cfg.get("data_parallel_size")) or maybe_int(
        backend_cfg.get("data_parallel_size")
    )

    if getattr(args, "tensor_parallel_size", None) is None and tp_default:
        args.tensor_parallel_size = tp_default
    if getattr(args, "pipeline_parallel_size", None) is None and pp_default:
        args.pipeline_parallel_size = pp_default
    if getattr(args, "data_parallel_size", None) is None and dp_default:
        args.data_parallel_size = dp_default

    # Port settings
    if getattr(args, "ray_port", None) is None:
        args.ray_port = maybe_int(backend_cfg.get("ray_port")) or 6379
    if getattr(args, "api_port", None) is None:
        args.api_port = maybe_int(backend_cfg.get("api_port")) or 8000

    # Build CLI args and env vars from vllm_server config
    merged_cfg = {**engine_cfg, **vllm_cfg}
    cli_args, env_vars = _build_vllm_cli_args(merged_cfg)
    args._vllm_cli_args = cli_args
    args._vllm_env_vars = env_vars


def setup_docker_runtime_if_needed(env_type: str) -> None:
    """Configure Docker/Podman runtime if using docker backend.

    Detects available Docker/Podman runtime, sets DOCKER_HOST environment
    variable, and verifies connectivity.

    Args:
        env_type: Harbor environment type (daytona, docker, modal)

    Raises:
        SystemExit: If docker backend requested but no runtime found
    """
    if env_type.lower() != "docker":
        return

    # Import docker runtime utilities (lazy import to avoid circular deps)
    from hpc.docker_runtime import (
        detect_docker_runtime,
        setup_docker_environment,
        DockerRuntimeType,
        check_docker_connectivity,
    )

    print("[docker] Detecting Docker/Podman runtime...")
    runtime = detect_docker_runtime()

    if runtime.runtime_type == DockerRuntimeType.UNAVAILABLE:
        print("[docker] ERROR: Docker backend requested but no Docker/Podman runtime found.")
        print("[docker] Please ensure Docker or Podman is installed and running,")
        print("[docker] or set DOCKER_HOST to point to a remote Docker daemon.")
        sys.exit(1)

    # Set up environment variables
    env = setup_docker_environment(runtime)
    os.environ.update(env)

    print(f"[docker] Runtime type: {runtime.runtime_type.value}")
    print(f"[docker] DOCKER_HOST: {runtime.docker_host}")

    # Verify connectivity
    if not check_docker_connectivity(timeout=10):
        print("[docker] WARNING: Docker daemon not responding. Continuing anyway...")
    else:
        print("[docker] Docker daemon is accessible.")


def load_harbor_config(harbor_config_path: str) -> Dict[str, Any]:
    """Load and parse Harbor config YAML.

    Args:
        harbor_config_path: Path to harbor config file

    Returns:
        Parsed harbor config dict (empty dict if file not found)
    """
    try:
        with open(harbor_config_path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except FileNotFoundError:
        return {}


__all__ = [
    # Process management
    "ManagedProcess",
    "start_ray",
    "start_vllm_controller",
    "wait_for_endpoint",
    "terminate_processes",
    # Config and setup utilities
    "maybe_int",
    "apply_datagen_defaults",
    "setup_docker_runtime_if_needed",
    "load_harbor_config",
    "resolve_jobs_dir_path",
    # Endpoint utilities
    "run_endpoint_health_check",
    "load_endpoint_metadata",
    # Harbor command building
    "extract_agent_kwargs_from_config",
    "default_job_name",
    "apply_nested_key",
    "parse_agent_kwarg_strings",
    "serialize_agent_kwargs",
    "build_harbor_command",
    # Re-exports
    "_build_vllm_cli_args",
]
