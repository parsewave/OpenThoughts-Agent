#!/usr/bin/env python3
"""
Launch OpenThoughts evals on a cloud VM via SkyPilot.

This wrapper mirrors the key arguments from eval/local/run_eval.py, then wraps the
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


class EvalCloudLauncher(CloudLauncher):
    """Cloud launcher for eval/local/run_eval.py."""

    task_name = "ot-eval-cloud"
    default_output_subdir = "cloud_runs"
    default_n_concurrent = 16

    def add_task_specific_args(self, parser) -> None:
        """Add eval-specific arguments."""
        # Core eval arguments
        parser.add_argument("--harbor-config", required=True, help="Path (within repo) to Harbor YAML.")
        parser.add_argument("--datagen-config", help="Optional datagen config to seed defaults.")
        parser.add_argument("--dataset", help="Harbor dataset slug (exclusive with --dataset-path).")
        parser.add_argument("--dataset-path", help="Path to tasks directory (exclusive with --dataset).")
        parser.add_argument("--model", required=True, help="Model identifier used by run_eval.")
        parser.add_argument("--agent", default="terminus-2", help="Harbor agent to run.")
        parser.add_argument("--eval-env", default="daytona", choices=["daytona", "docker", "modal"],
                            help="Harbor environment backend: daytona (cloud), docker (local/podman), modal. (default: daytona)")
        parser.add_argument("--harbor-extra-arg", action="append", default=[], help="Extra --harbor jobs start args.")
        parser.add_argument("--agent-kwarg", action="append", default=[], help="Additional --agent-kwarg entries.")
        parser.add_argument("--n-concurrent", type=int, default=self.default_n_concurrent)
        parser.add_argument("--n-attempts", type=int, default=3, help="Times to run each task for standard error calculation (default: 3).")
        parser.add_argument("--gpus", type=int, default=None, help="GPUs for run_eval (default: inferred from --accelerator).")
        parser.add_argument("--dry-run", action="store_true", help="Pass --dry-run to run_eval.")
        parser.add_argument("--job-name", help="Optional override for Harbor job name.")

        # Upload options (Supabase + HuggingFace)
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

    def get_dataset_arg_name(self) -> Optional[str]:
        """Return the dataset argument name for HF handling."""
        return "dataset_path"

    def normalize_paths(self, args) -> None:
        """Normalize repo-relative paths and infer defaults."""
        # Validate mutually exclusive dataset options
        if args.dataset and args.dataset_path:
            raise ValueError("Specify either --dataset or --dataset-path (not both).")
        if not args.dataset and not args.dataset_path:
            raise ValueError("Must provide --dataset or --dataset-path for eval workloads.")

        # Infer --gpus from --accelerator if not explicitly provided
        if args.gpus is None:
            args.gpus = parse_gpu_count(args.accelerator)

        # Normalize paths
        args.harbor_config = repo_relative(args.harbor_config, self.repo_root)
        if args.datagen_config:
            args.datagen_config = repo_relative(args.datagen_config, self.repo_root)
        if args.dataset_path and not args.dataset_path.startswith("/"):
            args.dataset_path = repo_relative(args.dataset_path, self.repo_root)

    def build_task_command(self, args, remote_output_dir: str) -> List[str]:
        """Build the run_eval.py command."""
        cmd: List[str] = [
            "python", "eval/local/run_eval.py",
            "--harbor-config", args.harbor_config,
            "--model", args.model,
        ]

        if args.datagen_config:
            cmd.extend(["--datagen-config", args.datagen_config])
        if args.dataset:
            cmd.extend(["--dataset", args.dataset])
        elif args.dataset_path:
            cmd.extend(["--dataset-path", args.dataset_path])

        cmd.extend([
            "--agent", args.agent,
            "--eval-env", args.eval_env,
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
        if args.upload_to_database:
            cmd.append("--upload-to-database")
        if args.upload_username:
            cmd.extend(["--upload-username", args.upload_username])
        if args.upload_error_mode:
            cmd.extend(["--upload-error-mode", args.upload_error_mode])
        if args.upload_hf_repo:
            cmd.extend(["--upload-hf-repo", args.upload_hf_repo])
        if args.upload_hf_token:
            cmd.extend(["--upload-hf-token", args.upload_hf_token])
        if args.upload_hf_private:
            cmd.append("--upload-hf-private")
        if args.upload_hf_episodes:
            cmd.extend(["--upload-hf-episodes", args.upload_hf_episodes])
        if args.upload_forced_update:
            cmd.append("--upload-forced-update")

        return cmd

    def sync_additional_outputs(self, cluster_name: str, args, remote_workdir: str) -> None:
        """Sync Harbor trace_jobs directory."""
        trace_jobs_remote = f"{remote_workdir}/trace_jobs"
        trace_jobs_local = str(Path(args.local_sync_dir) / "trace_jobs")
        print(f"[cloud-sync] Also syncing Harbor trace_jobs from {trace_jobs_remote}...")
        sync_outputs(
            cluster_name=cluster_name,
            remote_path=trace_jobs_remote,
            local_dir=trace_jobs_local,
        )


def main() -> None:
    launcher = EvalCloudLauncher(REPO_ROOT)
    parser = launcher.create_argument_parser(
        description="Launch eval/local/run_eval.py on a cloud GPU node via SkyPilot."
    )
    args = parser.parse_args()
    launcher.run(args)


if __name__ == "__main__":
    main()
