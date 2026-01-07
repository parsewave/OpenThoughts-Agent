#!/usr/bin/env python3
"""
Launch OpenThoughts trace generation on a cloud VM via SkyPilot.

This wrapper mirrors the key arguments from data/local/run_tracegen.py, then wraps the
whole invocation inside a SkyPilot Task so we can bring up short-lived GPU nodes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# Handle --list-providers before importing anything heavy
if "--list-providers" in sys.argv:
    from hpc.cloud_providers import list_providers
    print(list_providers(verbose=True))
    sys.exit(0)

from hpc.cloud_launch_utils import CloudLauncher, repo_relative, parse_gpu_count
from hpc.cloud_sync_utils import sync_outputs


class TracegenCloudLauncher(CloudLauncher):
    """Cloud launcher for data/local/run_tracegen.py."""

    task_name = "ot-tracegen-cloud"
    default_output_subdir = "cloud_runs"
    default_n_concurrent = 64

    def add_task_specific_args(self, parser) -> None:
        """Add tracegen-specific arguments."""
        # Core tracegen arguments
        parser.add_argument("--harbor-config", required=True, help="Path (within repo) to Harbor YAML.")
        parser.add_argument("--datagen-config", required=True, help="Datagen config with vLLM settings (required).")
        parser.add_argument("--tasks-input-path", required=True, help="Path to tasks directory OR HuggingFace repo (org/repo-name).")
        parser.add_argument("--model", help="Model identifier (overrides datagen config).")
        parser.add_argument("--agent", default="terminus-2", help="Harbor agent to run.")
        parser.add_argument("--trace-env", default="daytona", choices=["daytona", "docker", "modal"],
                            help="Harbor environment backend: daytona (cloud), docker (local/podman), modal. (default: daytona)")
        parser.add_argument("--harbor-extra-arg", action="append", default=[], help="Extra --harbor jobs start args.")
        parser.add_argument("--agent-kwarg", action="append", default=[], help="Additional --agent-kwarg entries.")
        parser.add_argument("--n-concurrent", type=int, default=self.default_n_concurrent, help="Concurrent trace trials (default: 64).")
        parser.add_argument("--n-attempts", type=int, default=1, help="Times to run each task for repeated trials (default: 1).")
        parser.add_argument("--gpus", type=int, default=None, help="GPUs for run_tracegen (default: inferred from --accelerator).")
        parser.add_argument("--dry-run", action="store_true", help="Pass --dry-run to run_tracegen.")
        parser.add_argument("--job-name", help="Optional override for Harbor job name.")

        # Upload options (HuggingFace only - simpler than eval)
        parser.add_argument(
            "--upload-hf-repo",
            help="HuggingFace repo for traces upload (e.g., my-org/my-traces).",
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

    def get_dataset_arg_name(self) -> Optional[str]:
        """Return the dataset argument name for HF handling."""
        return "tasks_input_path"

    def normalize_paths(self, args) -> None:
        """Normalize repo-relative paths and infer defaults."""
        # Infer --gpus from --accelerator if not explicitly provided
        if args.gpus is None:
            args.gpus = parse_gpu_count(args.accelerator)

        # Normalize paths
        args.harbor_config = repo_relative(args.harbor_config, self.repo_root)
        args.datagen_config = repo_relative(args.datagen_config, self.repo_root)
        if not args.tasks_input_path.startswith("/"):
            args.tasks_input_path = repo_relative(args.tasks_input_path, self.repo_root)

    def build_task_command(self, args, remote_output_dir: str) -> List[str]:
        """Build the run_tracegen.py command."""
        cmd: List[str] = [
            "python", "data/local/run_tracegen.py",
            "--harbor-config", args.harbor_config,
            "--datagen-config", args.datagen_config,
            "--tasks-input-path", args.tasks_input_path,
        ]

        if args.model:
            cmd.extend(["--model", args.model])

        cmd.extend([
            "--agent", args.agent,
            "--trace-env", args.trace_env,
            "--n-concurrent", str(args.n_concurrent),
            "--n-attempts", str(args.n_attempts),
            "--gpus", str(args.gpus),
            "--experiments-dir", remote_output_dir,
        ])

        if args.job_name:
            cmd.extend(["--job-name", args.job_name])
        if args.dry_run:
            cmd.append("--dry-run")

        for kwarg in args.agent_kwarg:
            cmd.extend(["--agent-kwarg", kwarg])
        for extra in args.harbor_extra_arg:
            cmd.extend(["--harbor-extra-arg", extra])

        # Upload options
        if args.upload_hf_repo:
            cmd.extend(["--upload-hf-repo", args.upload_hf_repo])
        if args.upload_hf_token:
            cmd.extend(["--upload-hf-token", args.upload_hf_token])
        if args.upload_hf_private:
            cmd.append("--upload-hf-private")

        return cmd

    def sync_additional_outputs(self, cluster_name: str, args, remote_workdir: str) -> None:
        """Sync Harbor jobs directory."""
        jobs_remote = f"{remote_workdir}/jobs"
        jobs_local = str(Path(args.local_sync_dir) / "jobs")
        print(f"[cloud-sync] Also syncing Harbor jobs from {jobs_remote}...")
        sync_outputs(
            cluster_name=cluster_name,
            remote_path=jobs_remote,
            local_dir=jobs_local,
        )


def main() -> None:
    launcher = TracegenCloudLauncher(REPO_ROOT)
    parser = launcher.create_argument_parser(
        description="Launch data/local/run_tracegen.py on a cloud GPU node via SkyPilot."
    )
    args = parser.parse_args()
    launcher.run(args)


if __name__ == "__main__":
    main()
