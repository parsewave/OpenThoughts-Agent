#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, snapshot_download


def _reset_dir(path: Path) -> None:
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.mkdir(parents=True, exist_ok=True)


def _copy_repo_files(
    src: Path,
    dest: Path,
    *,
    allowed_suffixes: Optional[set[str]] = None,
    skip_existing: bool = False,
) -> None:
    if not src.exists():
        return

    skip_suffixes = {".bin", ".pt", ".ckpt"}
    skip_prefixes = ("pytorch_model", "global_step", "checkpoint", "mp_rank", "zero_pp_rank")
    skip_dirs = {"logs", "wandb"}

    for item in src.rglob("*"):
        if item.is_dir():
            relative = item.relative_to(src)
            if relative.name in skip_dirs or relative.name.startswith(skip_prefixes):
                continue
            continue

        name = item.name
        if any(name.startswith(prefix) for prefix in skip_prefixes):
            continue
        if name.endswith(".safetensors"):
            continue
        suffix = item.suffix.lower()
        if suffix in skip_suffixes:
            continue
        if allowed_suffixes is not None and suffix not in allowed_suffixes:
            continue

        relative = item.relative_to(src)
        target = dest / relative
        if skip_existing and target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target, follow_symlinks=False)


def _infer_model_type(candidates: Iterable[Path]) -> Optional[str]:
    try:
        from transformers import AutoConfig  # type: ignore
    except Exception:
        return None

    for root in candidates:
        try:
            cfg = AutoConfig.from_pretrained(
                str(root),
                trust_remote_code=True,
                local_files_only=True,
            )
        except Exception as exc:
            print(f"Warning: failed to load config from {root}: {exc}", file=sys.stderr)
            continue

        model_type = getattr(cfg, "model_type", None)
        if isinstance(model_type, str) and model_type:
            return model_type.lower()

        architectures = getattr(cfg, "architectures", None)
        if architectures:
            for arch in architectures:
                if isinstance(arch, str) and arch:
                    return arch.lower()
    return None


def _zero_to_fp32_script(project_root: Path) -> Path:
    script_path = project_root / "scripts" / "consolidate" / "zero_to_fp32.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Bundled zero_to_fp32.py not found at {script_path}")
    return script_path


def consolidate(
    input_spec: str,
    input_kind: str,
    base_repo: Optional[str],
    output_repo: str,
    workdir: Path,
    commit_message: str,
    project_root: Path,
) -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN environment variable is required for consolidation.")

    if not os.environ.get("DCFT"):
        os.environ["DCFT"] = str(project_root)

    staged_input_dir = workdir / "input_stage"
    base_dir = workdir / "base_repo"
    merged_dir = workdir / "merged_weights"
    final_dir = workdir / "final_repo"

    print(f"Working directory: {workdir}")
    _reset_dir(workdir)
    _reset_dir(staged_input_dir)
    _reset_dir(base_dir)
    _reset_dir(merged_dir)
    _reset_dir(final_dir)

    normalized_kind = (input_kind or "").strip().lower()
    if normalized_kind not in {"repo", "local"}:
        guessed_path = Path(input_spec).expanduser()
        normalized_kind = "local" if guessed_path.exists() else "repo"

    if normalized_kind == "repo":
        print(f"Downloading input repo: {input_spec}")
        snapshot_download(
            repo_id=input_spec,
            repo_type="model",
            local_dir=staged_input_dir,
            local_dir_use_symlinks=False,
            token=token,
        )
    else:
        source_path = Path(input_spec).expanduser()
        if not source_path.exists():
            raise FileNotFoundError(
                f"Local consolidation input path not found: {source_path}"
            )
        print(f"Staging local input from {source_path}")
        shutil.copytree(source_path, staged_input_dir, dirs_exist_ok=True)

    if base_repo:
        print(f"Downloading base repo: {base_repo}")
        snapshot_download(
            repo_id=base_repo,
            repo_type="model",
            local_dir=base_dir,
            local_dir_use_symlinks=False,
            token=token,
        )

    zero_to_fp32 = _zero_to_fp32_script(project_root)
    print("Running zero_to_fp32 consolidation...")
    subprocess.check_call(
        [
            sys.executable,
            str(zero_to_fp32),
            str(staged_input_dir),
            str(merged_dir),
            "--safe_serialization",
        ]
    )

    print("Collecting ancillary files...")
    _copy_repo_files(staged_input_dir, final_dir)
    if base_repo:
        _copy_repo_files(
            base_dir,
            final_dir,
            allowed_suffixes={".json", ".jsonl", ".jinja"},
            skip_existing=True,
        )

    for item in list(final_dir.rglob("*")):
        if item.is_dir() and item.name in {"logs", "wandb"}:
            shutil.rmtree(item, ignore_errors=True)

    print("Copying merged weights...")
    for item in merged_dir.rglob("*"):
        if item.is_dir():
            continue
        relative = item.relative_to(merged_dir)
        destination = final_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, destination, follow_symlinks=False)

    detected_model_type = _infer_model_type([final_dir, base_dir, staged_input_dir])
    if detected_model_type:
        print(f"Detected model type: {detected_model_type}")
    else:
        print("Warning: unable to infer model type from configuration.", file=sys.stderr)

    print(f"Uploading consolidated checkpoint to Hugging Face repo {output_repo}...")
    api = HfApi()
    existing_paths = api.list_repo_files(repo_id=output_repo, repo_type="model", token=token)
    operations = [CommitOperationDelete(path_in_repo=path) for path in existing_paths]

    for item in final_dir.rglob("*"):
        if item.is_file():
            relative = item.relative_to(final_dir).as_posix()
            operations.append(CommitOperationAdd(path_in_repo=relative, path_or_fileobj=str(item)))

    api.create_commit(
        repo_id=output_repo,
        repo_type="model",
        operations=operations,
        commit_message=commit_message,
        token=token,
    )
    print(f"✓ Uploaded consolidated checkpoint to {output_repo}")

    print("Cleaning up temporary directories...")
    shutil.rmtree(staged_input_dir, ignore_errors=True)
    shutil.rmtree(base_dir, ignore_errors=True)
    shutil.rmtree(merged_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Consolidate ZeRO sharded checkpoints into safetensors and upload.")
    parser.add_argument("--input", required=True, help="Input source: either a local directory or a Hugging Face repo ID.")
    parser.add_argument("--input-kind", choices=["local", "repo"], default="repo", help="Explicit input type to avoid re-detection confusion.")
    parser.add_argument("--output-repo", required=True, help="Destination Hugging Face repo to upload the consolidated checkpoint.")
    parser.add_argument("--base-repo", default="", help="Base repo to copy ancillary files from.")
    parser.add_argument("--workdir", required=True, help="Working directory for consolidation artifacts.")
    parser.add_argument("--commit-message", required=True, help="Commit message for the upload.")
    parser.add_argument("--project-root", required=True, help="Path to OpenThoughts-Agent project root.")
    args = parser.parse_args()

    consolidate(
        input_spec=args.input,
        input_kind=args.input_kind,
        base_repo=args.base_repo or None,
        output_repo=args.output_repo,
        workdir=Path(args.workdir).expanduser().resolve(),
        commit_message=args.commit_message,
        project_root=Path(args.project_root).expanduser().resolve(),
    )


if __name__ == "__main__":
    main()
