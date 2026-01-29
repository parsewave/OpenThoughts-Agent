"""Beta9 Helm deployment to Kubernetes cluster."""

import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

from scripts.beam.config import Beta9Config

logger = logging.getLogger(__name__)

# Global to track cloned repo directory
_beta9_repo_dir: Optional[Path] = None

# Global to track log collector
_log_collector: Optional["LogCollector"] = None


class LogCollector:
    """Background log collector for Beta9 deployment debugging."""

    def __init__(self, namespace: str, output_dir: Path):
        self.namespace = namespace
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._log_file = self.output_dir / f"beta9_deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    def start(self):
        """Start collecting logs in background threads."""
        logger.info(f"Starting log collection -> {self._log_file}")

        # Thread to collect pod status
        status_thread = threading.Thread(target=self._collect_pod_status, daemon=True)
        status_thread.start()
        self._threads.append(status_thread)

        # Thread to collect events
        events_thread = threading.Thread(target=self._collect_events, daemon=True)
        events_thread.start()
        self._threads.append(events_thread)

        # Thread to collect pod logs
        logs_thread = threading.Thread(target=self._collect_pod_logs, daemon=True)
        logs_thread.start()
        self._threads.append(logs_thread)

    def stop(self):
        """Stop all collection threads."""
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=5)
        logger.info(f"Log collection stopped. Logs saved to: {self._log_file}")

    def _write_log(self, section: str, content: str):
        """Write content to log file with timestamp."""
        with open(self._log_file, "a") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"[{datetime.now().isoformat()}] {section}\n")
            f.write(f"{'='*60}\n")
            f.write(content)
            f.write("\n")

    def _collect_pod_status(self):
        """Periodically collect pod status."""
        while not self._stop_event.is_set():
            try:
                result = subprocess.run(
                    ["kubectl", "get", "pods", "-n", self.namespace, "-o", "wide"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    self._write_log("POD STATUS", result.stdout)
            except Exception as e:
                self._write_log("POD STATUS ERROR", str(e))

            self._stop_event.wait(30)  # Check every 30 seconds

    def _collect_events(self):
        """Periodically collect Kubernetes events."""
        while not self._stop_event.is_set():
            try:
                result = subprocess.run(
                    ["kubectl", "get", "events", "-n", self.namespace, "--sort-by=.lastTimestamp"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    self._write_log("EVENTS", result.stdout)
            except Exception as e:
                self._write_log("EVENTS ERROR", str(e))

            self._stop_event.wait(30)

    def _collect_pod_logs(self):
        """Collect logs from pods that exist."""
        seen_pods: set[str] = set()

        while not self._stop_event.is_set():
            try:
                # Get list of pods
                result = subprocess.run(
                    ["kubectl", "get", "pods", "-n", self.namespace, "-o", "jsonpath={.items[*].metadata.name}"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                if result.returncode == 0:
                    pods = result.stdout.split()
                    for pod in pods:
                        if pod and pod not in seen_pods:
                            seen_pods.add(pod)
                            # Start a thread to tail this pod's logs
                            t = threading.Thread(
                                target=self._tail_pod_log,
                                args=(pod,),
                                daemon=True,
                            )
                            t.start()
                            self._threads.append(t)
            except Exception:
                pass

            self._stop_event.wait(10)

    def _tail_pod_log(self, pod_name: str):
        """Tail logs from a specific pod."""
        try:
            # Get recent logs (last 100 lines) and describe
            result = subprocess.run(
                ["kubectl", "logs", "-n", self.namespace, pod_name, "--tail=100"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.stdout:
                self._write_log(f"LOGS: {pod_name}", result.stdout)
            if result.stderr:
                self._write_log(f"LOGS STDERR: {pod_name}", result.stderr)

            # Also get pod describe for debugging
            describe_result = subprocess.run(
                ["kubectl", "describe", "pod", "-n", self.namespace, pod_name],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if describe_result.stdout:
                self._write_log(f"DESCRIBE: {pod_name}", describe_result.stdout)

        except Exception as e:
            self._write_log(f"POD LOG ERROR: {pod_name}", str(e))


def start_log_collection(namespace: str, output_dir: Optional[Path] = None) -> LogCollector:
    """Start background log collection.

    Args:
        namespace: Kubernetes namespace to monitor.
        output_dir: Directory to save logs. Defaults to ./beta9_logs/

    Returns:
        LogCollector instance.
    """
    global _log_collector

    if output_dir is None:
        output_dir = Path.cwd() / "beta9_logs"

    _log_collector = LogCollector(namespace, output_dir)
    _log_collector.start()
    return _log_collector


def stop_log_collection():
    """Stop background log collection."""
    global _log_collector
    if _log_collector:
        _log_collector.stop()
        _log_collector = None


def check_helm_installed() -> bool:
    """Check if Helm CLI is installed."""
    if not shutil.which("helm"):
        logger.error("helm not found. Install from: https://helm.sh/docs/intro/install/")
        return False
    return True


def add_helm_repo(dry_run: bool = False, config: Optional[Beta9Config] = None) -> bool:
    """Prepare the Helm chart for deployment.

    Uses local chart if config.use_local_chart is True, otherwise clones from Git.

    Args:
        dry_run: If True, print command without executing.
        config: Beta9 configuration (optional, uses defaults if not provided).

    Returns:
        True if chart is ready.
    """
    global _beta9_repo_dir

    if config is None:
        config = Beta9Config()

    # Use local chart if configured
    if config.use_local_chart:
        # Find the project root (where scripts/beam/helm-chart is)
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent  # scripts/beam -> scripts -> project root
        local_chart_path = project_root / config.helm_chart_local_path

        if dry_run:
            logger.info(f"[DRY RUN] Would use local chart: {local_chart_path}")
            return True

        if not local_chart_path.exists():
            logger.error(f"Local Helm chart not found at: {local_chart_path}")
            logger.error("Run: git clone the chart or set use_local_chart=False")
            return False

        # Set the repo dir to None to indicate we're using local chart
        _beta9_repo_dir = None
        logger.info(f"Using local Helm chart: {local_chart_path}")
        return True

    # Otherwise clone from Git
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

    chart_path = _beta9_repo_dir / config.helm_chart_git_path
    if not chart_path.exists():
        logger.error(f"Helm chart not found at: {chart_path}")
        return False

    logger.info(f"Beta9 Helm chart ready at: {chart_path}")
    return True


def get_helm_chart_path(config: Beta9Config) -> Optional[Path]:
    """Get the path to the Helm chart (local or cloned).

    Args:
        config: Beta9 configuration.

    Returns:
        Path to the chart directory, or None if not available.
    """
    global _beta9_repo_dir

    # If using local chart
    if config.use_local_chart:
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
        local_chart_path = project_root / config.helm_chart_local_path
        if local_chart_path.exists():
            return local_chart_path
        return None

    # If using cloned repo
    if _beta9_repo_dir is None:
        return None

    return _beta9_repo_dir / config.helm_chart_git_path


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

    # Start background log collection for debugging
    log_collector = start_log_collection(config.namespace)

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

    try:
        result = subprocess.run(actual_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Failed to deploy Beta9: {result.stderr}")
            logger.error(f"stdout: {result.stdout}")
            return False

        logger.info("Beta9 deployed successfully")
        return True
    finally:
        # Always stop log collection
        stop_log_collection()


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
