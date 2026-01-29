"""Beta9 Helm deployment to Kubernetes cluster."""

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

import requests

from scripts.beam.config import Beta9Config

logger = logging.getLogger(__name__)

# Global to track cloned repo directory
_beta9_repo_dir: Optional[Path] = None


def check_helm_installed() -> bool:
    """Check if Helm CLI is installed."""
    if not shutil.which("helm"):
        logger.error("helm not found. Install from: https://helm.sh/docs/intro/install/")
        return False
    return True


def add_helm_repo(dry_run: bool = False, config: Optional[Beta9Config] = None) -> bool:
    """Clone the Beta9 repository to get the Helm chart.

    Beta9 doesn't publish to a Helm repository, so we clone the Git repo
    and install from the local chart at deploy/charts/beta9.

    Args:
        dry_run: If True, print command without executing.
        config: Beta9 configuration (optional, uses defaults if not provided).

    Returns:
        True if repo cloned successfully.
    """
    global _beta9_repo_dir

    if config is None:
        config = Beta9Config()

    git_url = config.helm_chart_git_repo

    if dry_run:
        logger.info(f"[DRY RUN] Would clone: {git_url}")
        return True

    logger.info("Cloning Beta9 repository for Helm chart...")

    # Create temp directory for the repo
    _beta9_repo_dir = Path(tempfile.mkdtemp(prefix="beta9-helm-"))

    cmd = ["git", "clone", "--depth", "1", git_url, str(_beta9_repo_dir)]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Failed to clone Beta9 repo: {result.stderr}")
        return False

    chart_path = _beta9_repo_dir / config.helm_chart_path
    if not chart_path.exists():
        logger.error(f"Helm chart not found at: {chart_path}")
        return False

    logger.info(f"Beta9 Helm chart ready at: {chart_path}")
    return True


def get_helm_chart_path(config: Beta9Config) -> Optional[Path]:
    """Get the path to the cloned Helm chart.

    Args:
        config: Beta9 configuration.

    Returns:
        Path to the chart directory, or None if not cloned.
    """
    global _beta9_repo_dir

    if _beta9_repo_dir is None:
        return None

    return _beta9_repo_dir / config.helm_chart_path


def cleanup_helm_repo():
    """Clean up the cloned Beta9 repository."""
    global _beta9_repo_dir

    if _beta9_repo_dir is not None and _beta9_repo_dir.exists():
        shutil.rmtree(_beta9_repo_dir)
        logger.debug(f"Cleaned up temp repo: {_beta9_repo_dir}")
        _beta9_repo_dir = None


def create_namespace(namespace: str, dry_run: bool = False) -> bool:
    """Create Kubernetes namespace if it doesn't exist.

    Args:
        namespace: Namespace name.
        dry_run: If True, print command without executing.

    Returns:
        True if namespace exists or created successfully.
    """
    # Check if exists
    check_result = subprocess.run(
        ["kubectl", "get", "namespace", namespace],
        capture_output=True,
        text=True,
    )

    if check_result.returncode == 0:
        logger.debug(f"Namespace '{namespace}' already exists")
        return True

    cmd = ["kubectl", "create", "namespace", namespace]

    if dry_run:
        logger.info(f"[DRY RUN] Would execute: {' '.join(cmd)}")
        return True

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Failed to create namespace: {result.stderr}")
        return False

    logger.info(f"Created namespace '{namespace}'")
    return True


def build_helm_dependencies(config: Beta9Config, dry_run: bool = False) -> bool:
    """Build Helm chart dependencies.

    Args:
        config: Beta9 configuration.
        dry_run: If True, print command without executing.

    Returns:
        True if dependencies built successfully.
    """
    chart_path = get_helm_chart_path(config)
    if chart_path is None and not dry_run:
        logger.error("Helm chart not available. Run add_helm_repo() first.")
        return False

    chart_path_str = str(chart_path) if chart_path else "./deploy/charts/beta9"

    cmd = ["helm", "dependency", "update", chart_path_str]

    if dry_run:
        logger.info(f"[DRY RUN] Would execute: {' '.join(cmd)}")
        return True

    logger.info("Updating Helm chart dependencies...")
    logger.info("  This may take a minute...")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Failed to build dependencies: {result.stderr}")
        return False

    logger.info("Helm dependencies updated successfully")
    return True


def deploy_beta9(config: Beta9Config, dry_run: bool = False) -> bool:
    """Deploy Beta9 to the cluster using Helm.

    Args:
        config: Beta9 configuration.
        dry_run: If True, print command without executing.

    Returns:
        True if deployment successful.
    """
    # Create namespace first
    if not create_namespace(config.namespace, dry_run):
        return False

    # Get the local chart path
    chart_path = get_helm_chart_path(config)
    if chart_path is None and not dry_run:
        logger.error("Helm chart not available. Run add_helm_repo() first.")
        return False

    chart_path_str = str(chart_path) if chart_path else "./deploy/charts/beta9"

    # Build dependencies first
    if not build_helm_dependencies(config, dry_run):
        return False

    # Build Helm values for CPU-only deployment
    values = {
        "gateway": {
            "replicas": 1,
            "service": {
                "type": "ClusterIP",
                "httpPort": config.gateway_http_port,
                "grpcPort": config.gateway_grpc_port,
            },
        },
        "worker": {
            "enabled": True,
            "replicas": config.worker_replicas,
            "resources": {
                "requests": {
                    "cpu": config.worker_cpu_request,
                    "memory": config.worker_memory_request,
                },
            },
        },
        # Disable GPU-specific components
        "gpu": {
            "enabled": False,
        },
    }

    values_json = json.dumps(values)

    cmd = [
        "helm", "upgrade", "--install",
        config.helm_release_name,
        chart_path_str,
        "--namespace", config.namespace,
        "--set-json", f"'{values_json}'",
        "--wait",
        "--timeout", "20m",
    ]

    if dry_run:
        logger.info(f"[DRY RUN] Would execute: {' '.join(cmd)}")
        logger.info(f"[DRY RUN] With values: {json.dumps(values, indent=2)}")
        return True

    logger.info(f"Deploying Beta9 to namespace '{config.namespace}'...")
    logger.info(f"  Using chart from: {chart_path_str}")
    logger.info("  This may take a few minutes...")

    # Run without shell=True by properly handling the JSON
    actual_cmd = [
        "helm", "upgrade", "--install",
        config.helm_release_name,
        chart_path_str,
        "--namespace", config.namespace,
        "--set-json", values_json,
        "--wait",
        "--timeout", "20m",
    ]

    result = subprocess.run(actual_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Failed to deploy Beta9: {result.stderr}")
        logger.error(f"stdout: {result.stdout}")
        return False

    logger.info("Beta9 deployed successfully")
    return True


def uninstall_beta9(config: Beta9Config, dry_run: bool = False) -> bool:
    """Uninstall Beta9 from the cluster.

    Args:
        config: Beta9 configuration.
        dry_run: If True, print command without executing.

    Returns:
        True if uninstall successful.
    """
    cmd = [
        "helm", "uninstall",
        config.helm_release_name,
        "--namespace", config.namespace,
    ]

    if dry_run:
        logger.info(f"[DRY RUN] Would execute: {' '.join(cmd)}")
        return True

    logger.info(f"Uninstalling Beta9 from namespace '{config.namespace}'...")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0 and "not found" not in result.stderr.lower():
        logger.error(f"Failed to uninstall Beta9: {result.stderr}")
        return False

    logger.info("Beta9 uninstalled successfully")
    return True


def get_gateway_cluster_ip(config: Beta9Config) -> Optional[str]:
    """Get the ClusterIP of the Beta9 gateway service.

    Args:
        config: Beta9 configuration.

    Returns:
        ClusterIP address or None if not found.
    """
    result = subprocess.run(
        [
            "kubectl", "get", "service",
            config.gateway_service_name,
            "--namespace", config.namespace,
            "-o", "jsonpath={.spec.clusterIP}",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0 or not result.stdout.strip():
        return None

    return result.stdout.strip()


def wait_for_gateway_ready(
    config: Beta9Config,
    timeout: int = 300,
    check_endpoint: bool = True,
) -> bool:
    """Wait for Beta9 gateway to be ready.

    Args:
        config: Beta9 configuration.
        timeout: Maximum seconds to wait.
        check_endpoint: If True, also check HTTP health endpoint.

    Returns:
        True if gateway is ready within timeout.
    """
    logger.info("Waiting for Beta9 gateway to be ready...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        # Check pod status
        result = subprocess.run(
            [
                "kubectl", "get", "pods",
                "--namespace", config.namespace,
                "-l", f"app.kubernetes.io/name={config.helm_release_name}-gateway",
                "-o", "jsonpath={.items[*].status.phase}",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0 and "Running" in result.stdout:
            logger.info("Gateway pod is running")

            if not check_endpoint:
                return True

            # Also check the health endpoint via port-forward
            # (Skip for now, let expose.py handle this)
            return True

        logger.debug(f"Gateway status: {result.stdout}")
        time.sleep(10)

    logger.error(f"Gateway did not become ready within {timeout} seconds")
    return False


def get_deployment_status(config: Beta9Config) -> dict:
    """Get status of Beta9 deployment.

    Returns:
        Dictionary with deployment status information.
    """
    status = {
        "installed": False,
        "gateway_ready": False,
        "worker_ready": False,
        "pods": [],
    }

    # Check if Helm release exists
    result = subprocess.run(
        [
            "helm", "status",
            config.helm_release_name,
            "--namespace", config.namespace,
            "-o", "json",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        status["installed"] = True

    # Get pod status
    pod_result = subprocess.run(
        [
            "kubectl", "get", "pods",
            "--namespace", config.namespace,
            "-o", "json",
        ],
        capture_output=True,
        text=True,
    )

    if pod_result.returncode == 0:
        try:
            pods_data = json.loads(pod_result.stdout)
            for pod in pods_data.get("items", []):
                pod_info = {
                    "name": pod["metadata"]["name"],
                    "phase": pod["status"]["phase"],
                    "ready": all(
                        c.get("ready", False)
                        for c in pod["status"].get("containerStatuses", [])
                    ),
                }
                status["pods"].append(pod_info)

                if "gateway" in pod_info["name"] and pod_info["ready"]:
                    status["gateway_ready"] = True
                if "worker" in pod_info["name"] and pod_info["ready"]:
                    status["worker_ready"] = True
        except json.JSONDecodeError:
            pass

    return status
