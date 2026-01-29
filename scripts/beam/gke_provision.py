"""GKE cluster provisioning using gcloud CLI."""

import logging
import shutil
import subprocess
import time
from typing import Optional

from scripts.beam.config import GKEConfig

logger = logging.getLogger(__name__)


def check_gcloud_installed() -> bool:
    """Check if gcloud CLI is installed and authenticated."""
    if not shutil.which("gcloud"):
        logger.error("gcloud CLI not found. Install from: https://cloud.google.com/sdk/docs/install")
        return False

    # Check authentication
    result = subprocess.run(
        ["gcloud", "auth", "list", "--filter=status:ACTIVE", "--format=value(account)"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        logger.error("gcloud not authenticated. Run: gcloud auth login")
        return False

    return True


def check_kubectl_installed() -> bool:
    """Check if kubectl is installed."""
    if not shutil.which("kubectl"):
        logger.error("kubectl not found. Install from: https://kubernetes.io/docs/tasks/tools/")
        return False
    return True


def check_git_installed() -> bool:
    """Check if git is installed."""
    if not shutil.which("git"):
        logger.error("git not found. Install from: https://git-scm.com/downloads")
        return False
    return True


def cluster_exists(config: GKEConfig) -> bool:
    """Check if the GKE cluster already exists."""
    result = subprocess.run(
        [
            "gcloud", "container", "clusters", "describe",
            config.cluster_name,
            "--project", config.project_id,
            "--region", config.region,
            "--format=value(name)",
        ],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and config.cluster_name in result.stdout


def create_cluster(config: GKEConfig, dry_run: bool = False) -> bool:
    """Create a GKE cluster with CPU-only nodes.

    Args:
        config: GKE configuration.
        dry_run: If True, print command without executing.

    Returns:
        True if cluster created successfully (or dry_run).
    """
    cmd = [
        "gcloud", "container", "clusters", "create", config.cluster_name,
        "--project", config.project_id,
        "--region", config.region,
        "--num-nodes", str(config.num_nodes),
        "--machine-type", config.machine_type,
        "--disk-size", f"{config.disk_size_gb}GB",
        "--network", config.network,
        "--no-enable-autoupgrade",
        "--no-enable-autorepair",
    ]

    if config.enable_ip_alias:
        cmd.append("--enable-ip-alias")

    if dry_run:
        logger.info(f"[DRY RUN] Would execute: {' '.join(cmd)}")
        return True

    logger.info(f"Creating GKE cluster '{config.cluster_name}' with {config.num_nodes} nodes...")
    logger.info(f"  Machine type: {config.machine_type}")
    logger.info(f"  Region: {config.region}")
    logger.info("  This may take 5-10 minutes...")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Failed to create cluster: {result.stderr}")
        return False

    logger.info(f"Cluster '{config.cluster_name}' created successfully")
    return True


def delete_cluster(config: GKEConfig, dry_run: bool = False) -> bool:
    """Delete the GKE cluster.

    Args:
        config: GKE configuration.
        dry_run: If True, print command without executing.

    Returns:
        True if cluster deleted successfully (or dry_run).
    """
    cmd = [
        "gcloud", "container", "clusters", "delete", config.cluster_name,
        "--project", config.project_id,
        "--region", config.region,
        "--quiet",  # Skip confirmation prompt
    ]

    if dry_run:
        logger.info(f"[DRY RUN] Would execute: {' '.join(cmd)}")
        return True

    logger.info(f"Deleting GKE cluster '{config.cluster_name}'...")
    logger.info("  This may take a few minutes...")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Failed to delete cluster: {result.stderr}")
        return False

    logger.info(f"Cluster '{config.cluster_name}' deleted successfully")
    return True


def get_credentials(config: GKEConfig, dry_run: bool = False) -> bool:
    """Get kubectl credentials for the GKE cluster.

    Args:
        config: GKE configuration.
        dry_run: If True, print command without executing.

    Returns:
        True if credentials obtained successfully (or dry_run).
    """
    cmd = [
        "gcloud", "container", "clusters", "get-credentials", config.cluster_name,
        "--project", config.project_id,
        "--region", config.region,
    ]

    if dry_run:
        logger.info(f"[DRY RUN] Would execute: {' '.join(cmd)}")
        return True

    logger.info(f"Getting kubectl credentials for '{config.cluster_name}'...")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Failed to get credentials: {result.stderr}")
        return False

    logger.info("kubectl credentials configured successfully")
    return True


def grant_cluster_admin(dry_run: bool = False) -> bool:
    """Grant cluster-admin role to the current gcloud user.

    This is required for deploying Helm charts that create ClusterRoles
    and ClusterRoleBindings (like Beta9).

    Args:
        dry_run: If True, print command without executing.

    Returns:
        True if successful or binding already exists.
    """
    # Get current gcloud user
    user_result = subprocess.run(
        ["gcloud", "config", "get-value", "account"],
        capture_output=True,
        text=True,
    )

    if user_result.returncode != 0 or not user_result.stdout.strip():
        logger.error("Could not determine current gcloud user")
        return False

    user_email = user_result.stdout.strip()

    cmd = [
        "kubectl", "create", "clusterrolebinding", "cluster-admin-binding",
        "--clusterrole=cluster-admin",
        f"--user={user_email}",
    ]

    if dry_run:
        logger.info(f"[DRY RUN] Would execute: {' '.join(cmd)}")
        return True

    logger.info(f"Granting cluster-admin to {user_email}...")

    result = subprocess.run(cmd, capture_output=True, text=True)

    # "already exists" is fine
    if result.returncode != 0:
        if "already exists" in result.stderr:
            logger.info("cluster-admin binding already exists")
            return True
        logger.error(f"Failed to grant cluster-admin: {result.stderr}")
        return False

    logger.info("cluster-admin granted successfully")
    return True


def get_cluster_status(config: GKEConfig) -> Optional[str]:
    """Get the current status of the cluster.

    Returns:
        Cluster status string (e.g., "RUNNING", "PROVISIONING") or None if not found.
    """
    result = subprocess.run(
        [
            "gcloud", "container", "clusters", "describe",
            config.cluster_name,
            "--project", config.project_id,
            "--region", config.region,
            "--format=value(status)",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        return None

    return result.stdout.strip()


def wait_for_cluster_ready(config: GKEConfig, timeout: int = 600) -> bool:
    """Wait for cluster to reach RUNNING status.

    Args:
        config: GKE configuration.
        timeout: Maximum seconds to wait.

    Returns:
        True if cluster is running within timeout.
    """
    logger.info("Waiting for cluster to be ready...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        status = get_cluster_status(config)
        if status == "RUNNING":
            logger.info("Cluster is ready")
            return True

        logger.debug(f"Cluster status: {status}")
        time.sleep(10)

    logger.error(f"Cluster did not become ready within {timeout} seconds")
    return False


def get_node_count(config: GKEConfig) -> int:
    """Get current number of nodes in the cluster."""
    result = subprocess.run(
        ["kubectl", "get", "nodes", "-o", "name"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        return 0

    return len(result.stdout.strip().split("\n"))
