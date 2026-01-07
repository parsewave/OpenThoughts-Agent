#!/usr/bin/env python3
"""
Local eval runner.

Starts a single-node Ray cluster + vLLM controller and then launches a Harbor eval
job that targets the freshly booted endpoint. Designed for non-SLURM Linux hosts
where we have exclusive access to the box.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENTS_DIR = REPO_ROOT / "eval_runs"
DEFAULT_ENDPOINT = "vllm_endpoint.json"

# Import shared utilities for local runners
from hpc.local_runner_utils import (
    ManagedProcess,
    start_ray,
    start_vllm_controller,
    wait_for_endpoint,
    terminate_processes,
    default_job_name,
    build_harbor_command,
    apply_datagen_defaults,
    setup_docker_runtime_if_needed,
    load_harbor_config,
    resolve_jobs_dir_path,
    run_endpoint_health_check,
    load_endpoint_metadata,
)
from hpc.launch_utils import generate_served_model_id, hosted_vllm_alias


def _ensure_database_module_path() -> None:
    db_path = REPO_ROOT / "database"
    if db_path.exists():
        db_path_str = str(db_path)
        if db_path_str not in sys.path:
            sys.path.insert(0, db_path_str)


def _sanitize_hf_repo_id(repo_id: str, max_length: int = 96) -> str:
    """
    Sanitize a Hugging Face repo_id to comply with naming rules.
    Based on the sbatch uploader helper used in HPC eval jobs.
    """

    def collapse(value: str) -> str:
        prev = None
        while value != prev:
            prev = value
            value = value.replace("--", "-").replace("..", ".")
        return value

    org, name = repo_id.split("/", 1) if "/" in repo_id else (None, repo_id)
    name = re.sub(r"[^A-Za-z0-9._-]", "-", name)
    name = collapse(name).strip("-.")
    if not name:
        name = "repo"
    limit = max_length - (len(org) + 1 if org else 0)
    if len(name) > limit > 8:
        digest = hashlib.sha1(name.encode()).hexdigest()[:8]
        keep = max(1, limit - len(digest))
        base = name[:keep].rstrip("-.") or "r"
        name = collapse(f"{base}{digest}").strip("-.")
    if name[0] in "-.":
        name = f"r{name[1:]}"
    if name[-1] in "-.":
        name = f"{name[:-1]}0"
    return f"{org}/{name}" if org else name


def _derive_default_hf_repo_id(args: argparse.Namespace, job_name: str) -> str:
    # Use shared utility for benchmark derivation
    from hpc.launch_utils import derive_benchmark_repo
    benchmark_repo = derive_benchmark_repo(
        harbor_dataset=args.dataset,
        dataset_path=args.dataset_path,
        explicit_repo=getattr(args, "eval_benchmark_repo", None),
    )
    if "/" in benchmark_repo:
        org = benchmark_repo.split("/", 1)[0]
    else:
        org = benchmark_repo or "openthoughts-agent"
    return f"{org}/{job_name}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Harbor evals against a local Ray/vLLM server.")
    parser.add_argument("--harbor-config", required=True, help="Path to Harbor eval YAML.")
    parser.add_argument(
        "--dataset",
        help="Harbor dataset slug (e.g., terminal-bench@2.0). Mutually exclusive with --dataset-path.",
    )
    parser.add_argument(
        "--dataset-path",
        help="Path to a Harbor task directory. Mutually exclusive with --dataset.",
    )
    parser.add_argument("--model", help="Trace model identifier (used for Harbor + vLLM).")
    parser.add_argument("--agent", help="Harbor agent name to run (default terminus-2).")
    parser.add_argument("--eval-env", default="daytona", choices=["daytona", "docker", "modal"],
                        help="Harbor environment backend: daytona (cloud), docker (local/podman), modal. (default: daytona)")
    parser.add_argument("--n-concurrent", type=int, default=16, help="Concurrent eval trials.")
    parser.add_argument("--n-attempts", type=int, default=3, help="Retries (Harbor --n-attempts flag).")
    parser.add_argument(
        "--experiments-dir",
        default=str(DEFAULT_EXPERIMENTS_DIR),
        help="Directory for logs + endpoint JSON.",
    )
    parser.add_argument("--job-name", help="Optional Harbor job name override.")
    parser.add_argument("--host", default="127.0.0.1", help="Host/IP for Ray + vLLM.")
    parser.add_argument("--ray-port", type=int, default=6379, help="Ray head port.")
    parser.add_argument("--api-port", type=int, default=8000, help="vLLM OpenAI server port.")
    parser.add_argument("--gpus", type=int, help="GPUs to expose to Ray.")
    parser.add_argument("--cpus", type=int, help="CPUs to expose to Ray.")
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--pipeline-parallel-size", type=int)
    parser.add_argument("--data-parallel-size", type=int)
    parser.add_argument("--health-max-attempts", type=int, default=100)
    parser.add_argument("--health-retry-delay", type=int, default=30)
    parser.add_argument("--harbor-binary", default="harbor", help="Harbor CLI executable.")
    parser.add_argument(
        "--agent-kwarg",
        action="append",
        default=[],
        help="Additional --agent-kwarg entries (key=value).",
    )
    parser.add_argument(
        "--controller-log",
        help="Optional path for vLLM controller stdout/stderr.",
    )
    parser.add_argument(
        "--ray-log",
        help="Optional path for Ray stdout/stderr.",
    )
    parser.add_argument(
        "--endpoint-json",
        help="Optional endpoint JSON path. Defaults to <experiments_dir>/vllm_endpoint.json",
    )
    parser.add_argument(
        "--harbor-extra-arg",
        action="append",
        default=[],
        help="Additional passthrough args for `harbor jobs start`.",
    )
    parser.add_argument(
        "--harbor-log",
        help="Optional path for Harbor CLI stdout/stderr.",
    )
    parser.add_argument(
        "--datagen-config",
        help="Optional datagen YAML whose vLLM settings will seed defaults for this script.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing Harbor.",
    )
    # Upload options:
    # - Traces (full rollout data) -> HuggingFace
    # - Result abstracts (job/trial metadata, metrics) -> Supabase
    parser.add_argument(
        "--upload-to-database",
        action="store_true",
        help="After Harbor finishes, upload result abstracts to Supabase and traces to HuggingFace.",
    )
    parser.add_argument(
        "--upload-username",
        help="Username for Supabase result attribution (defaults to $UPLOAD_USERNAME or current user).",
    )
    parser.add_argument(
        "--upload-error-mode",
        choices=["skip_on_error", "rollback_on_error"],
        default="skip_on_error",
        help="Supabase upload error handling (default: skip_on_error).",
    )
    parser.add_argument(
        "--upload-hf-repo",
        help="HuggingFace repo for traces upload (defaults to <org>/<job_name>).",
    )
    parser.add_argument(
        "--upload-hf-token",
        help="HuggingFace token for traces upload (defaults to $HF_TOKEN).",
    )
    parser.add_argument(
        "--upload-hf-private",
        action="store_true",
        help="Create the HuggingFace traces repo as private.",
    )
    parser.add_argument(
        "--upload-hf-episodes",
        choices=["last", "all"],
        default="last",
        help="Which episodes to include in HuggingFace traces upload.",
    )
    parser.add_argument(
        "--upload-forced-update",
        action="store_true",
        help="Allow overwriting existing Supabase result records for the same job.",
    )
    return parser.parse_args()


def _ensure_mutually_exclusive(dataset: Optional[str], dataset_path: Optional[str]) -> None:
    if dataset and dataset_path:
        raise ValueError("Specify either --dataset or --dataset-path (not both).")
    if not dataset and not dataset_path:
        raise ValueError("Must provide --dataset or --dataset-path.")


def _maybe_upload_results(args: argparse.Namespace) -> None:
    """Upload eval results using the shared upload functions from hpc.launch_utils."""
    if not getattr(args, "upload_to_database", False):
        return
    if args.dry_run:
        print("[upload] Skipping Supabase upload because --dry-run was set.")
        return
    job_name = getattr(args, "_harbor_job_name", None)
    jobs_dir_path = getattr(args, "_jobs_dir_path", None)
    if not job_name or jobs_dir_path is None:
        print("[upload] Unable to determine job directory; upload skipped.")
        return
    run_dir = Path(jobs_dir_path) / job_name
    if not run_dir.exists():
        print(f"[upload] Expected Harbor job directory {run_dir} does not exist; upload skipped.")
        return

    # Use shared utilities from hpc.launch_utils
    from hpc.launch_utils import sync_eval_to_database, derive_benchmark_repo

    benchmark_name = derive_benchmark_repo(
        harbor_dataset=args.dataset,
        dataset_path=args.dataset_path,
    )
    hf_repo_id = args.upload_hf_repo or _derive_default_hf_repo_id(args, job_name)
    if hf_repo_id:
        hf_repo_id = _sanitize_hf_repo_id(hf_repo_id)

    result = sync_eval_to_database(
        job_dir=run_dir,
        username=args.upload_username,
        error_mode=args.upload_error_mode,
        agent_name=args.agent,
        model_name=args.model,
        benchmark_name=benchmark_name,
        register_benchmark=True,
        hf_repo_id=hf_repo_id,
        hf_private=args.upload_hf_private,
        hf_token=args.upload_hf_token,
        hf_episodes=args.upload_hf_episodes,
        forced_update=args.upload_forced_update,
        dry_run=args.dry_run,
    )

    if not result.get("success"):
        print(f"[upload] Upload failed: {result.get('error', 'unknown error')}")


def _run_harbor_cli(cmd: List[str], log_path: Path | None) -> None:
    # Use shared PTY-based runner
    from hpc.cli_utils import run_harbor_cli
    run_harbor_cli(cmd, log_path)


def main() -> None:
    args = _parse_args()
    apply_datagen_defaults(args)

    # Set up Docker runtime if using docker backend
    setup_docker_runtime_if_needed(args.eval_env)

    if args.agent is None:
        args.agent = "terminus-2"
    if args.tensor_parallel_size is None:
        args.tensor_parallel_size = 1
    if args.pipeline_parallel_size is None:
        args.pipeline_parallel_size = 1
    if args.data_parallel_size is None:
        args.data_parallel_size = 1
    if args.model is None:
        raise ValueError("Provide --model or supply a datagen config with vllm_server.model_path.")
    served_model_id = generate_served_model_id()
    args._served_model_id = served_model_id
    args._harbor_model_name = hosted_vllm_alias(served_model_id)
    if args.ray_port is None:
        args.ray_port = 6379
    if args.api_port is None:
        args.api_port = 8000
    if args.gpus is None:
        args.gpus = max(
            1,
            args.tensor_parallel_size * args.pipeline_parallel_size * args.data_parallel_size,
        )
    if args.cpus is None:
        args.cpus = os.cpu_count() or 16
    args.harbor_config = str(Path(args.harbor_config).expanduser().resolve())
    harbor_config_data = load_harbor_config(args.harbor_config)
    jobs_dir_value = harbor_config_data.get("jobs_dir") if isinstance(harbor_config_data, dict) else None
    args._jobs_dir_path = resolve_jobs_dir_path(jobs_dir_value, REPO_ROOT)
    args._harbor_config_data = harbor_config_data
    if args.dataset_path:
        args.dataset_path = str(Path(args.dataset_path).expanduser().resolve())

    _ensure_mutually_exclusive(args.dataset, args.dataset_path)

    experiments_dir = Path(args.experiments_dir).expanduser().resolve()
    experiments_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = experiments_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    endpoint_json = Path(args.endpoint_json or (experiments_dir / DEFAULT_ENDPOINT))
    if endpoint_json.exists():
        endpoint_json.unlink()

    os.chdir(REPO_ROOT)

    dataset_label = args.dataset or args.dataset_path or "dataset"
    ray_log = Path(args.ray_log) if args.ray_log else logs_dir / "ray.log"
    controller_log = Path(args.controller_log) if args.controller_log else logs_dir / "vllm_controller.log"
    harbor_log = Path(args.harbor_log).expanduser().resolve() if args.harbor_log else None

    processes: List[ManagedProcess] = []

    def _handle_signal(signum, _frame):
        print(f"\nSignal {signum} received; shutting down...", file=sys.stderr)
        terminate_processes(processes)
        subprocess.run(["ray", "stop", "--force"], check=False)
        sys.exit(1)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    controller_script = REPO_ROOT / "scripts" / "vllm" / "start_vllm_ray_controller.py"

    ray_proc = start_ray(
        host=args.host,
        ray_port=args.ray_port,
        num_gpus=args.gpus,
        num_cpus=args.cpus,
        log_path=ray_log,
    )
    processes.append(ray_proc)

    vllm_proc = start_vllm_controller(
        model=args.model,
        host=args.host,
        ray_port=args.ray_port,
        api_port=args.api_port,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        data_parallel_size=args.data_parallel_size,
        endpoint_path=endpoint_json,
        controller_script=controller_script,
        log_path=controller_log,
        served_model_name=getattr(args, "_served_model_id", None),
        extra_cli_args=getattr(args, "_vllm_cli_args", []),
        extra_env_vars=getattr(args, "_vllm_env_vars", {}),
    )
    processes.append(vllm_proc)

    try:
        wait_for_endpoint(endpoint_json, vllm_proc)
        run_endpoint_health_check(endpoint_json, args.health_max_attempts, args.health_retry_delay, REPO_ROOT)
        endpoint_meta = load_endpoint_metadata(endpoint_json)

        # Compute job name and store for later use (e.g., uploads)
        harbor_model = getattr(args, "_harbor_model_name", args.model)
        job_model_label = args.model or harbor_model or "model"
        job_name = args.job_name or default_job_name("eval", dataset_label, job_model_label)
        args._harbor_job_name = job_name

        harbor_cmd = build_harbor_command(
            harbor_binary=args.harbor_binary,
            harbor_config_path=args.harbor_config,
            harbor_config_data=getattr(args, "_harbor_config_data", {}),
            job_name=job_name,
            agent_name=args.agent,
            model_name=harbor_model,
            env_type=args.eval_env,
            n_concurrent=args.n_concurrent,
            n_attempts=args.n_attempts,
            endpoint_meta=endpoint_meta,
            agent_kwarg_overrides=list(args.agent_kwarg or []),
            harbor_extra_args=list(args.harbor_extra_arg or []),
            dataset_slug=args.dataset,
            dataset_path=args.dataset_path,
        )
        print("Harbor command:", " ".join(harbor_cmd))
        if not args.dry_run:
            _run_harbor_cli(harbor_cmd, harbor_log)
            _maybe_upload_results(args)
        elif args.upload_to_database:
            print("[upload] --upload-to-database ignored because --dry-run was requested.")
    finally:
        terminate_processes(processes[::-1])
        subprocess.run(["ray", "stop", "--force"], check=False)


if __name__ == "__main__":
    main()
