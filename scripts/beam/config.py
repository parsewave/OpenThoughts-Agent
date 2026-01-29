"""Configuration dataclasses for Beam cluster setup."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GKEConfig:
    """Configuration for GKE cluster provisioning."""

    project_id: str
    cluster_name: str = "beam-test-cluster"
    region: str = "us-central1"
    zone: str = "us-central1-a"
    num_nodes: int = 2  # Nodes per zone (3 zones = 6 total nodes)
    machine_type: str = "e2-standard-4"  # 4 vCPU, 16GB RAM, ~$0.13/hr
    disk_size_gb: int = 100
    network: str = "default"
    enable_ip_alias: bool = True

    def get_full_cluster_name(self) -> str:
        """Return fully qualified cluster name for gcloud commands."""
        return f"projects/{self.project_id}/locations/{self.region}/clusters/{self.cluster_name}"


@dataclass
class FilestoreConfig:
    """Configuration for GKE Filestore (NFS) for shared storage."""

    instance_name: str = "beta9-filestore"
    tier: str = "BASIC_HDD"  # BASIC_HDD (~$0.20/GB/mo) or BASIC_SSD (~$0.30/GB/mo)
    capacity_gb: int = 1024  # Minimum 1TB for Basic tier
    file_share_name: str = "beta9share"
    network: str = "default"

    # These are set after creation
    ip_address: Optional[str] = field(default=None, repr=False)


@dataclass
class Beta9Config:
    """Configuration for Beta9 Helm deployment."""

    namespace: str = "beta9"
    helm_release_name: str = "beta9"
    # Use local helm chart (copied from beta9 repo and customized for GKE)
    helm_chart_local_path: str = "scripts/beam/helm-chart"
    # Original repo (kept for reference)
    helm_chart_git_repo: str = "https://github.com/beam-cloud/beta9.git"
    helm_chart_git_path: str = "deploy/charts/beta9"  # Path within the repo
    use_local_chart: bool = True  # Use local chart by default
    gateway_http_port: int = 1994
    gateway_grpc_port: int = 1993
    gateway_service_name: str = "beta9-gateway"

    # Resource configuration for CPU-only deployment
    worker_cpu_request: str = "1"
    worker_memory_request: str = "1Gi"
    worker_replicas: int = 4


@dataclass
class PinggyConfig:
    """Configuration for Pinggy tunnel exposure."""

    persistent_url: str
    token: str
    local_port: int = 1994
    local_host: str = "localhost"
    pinggy_host: str = "pro.pinggy.io"
    health_check_timeout: int = 60

    def get_ssh_command(self) -> str:
        """Build SSH command for Pinggy tunnel with auto-reconnect."""
        return (
            f'while true; do '
            f'ssh -p 443 '
            f'-R0:{self.local_host}:{self.local_port} '
            f'-o StrictHostKeyChecking=no '
            f'-o ServerAliveInterval=30 '
            f'-o ExitOnForwardFailure=yes '
            f'{self.token}@{self.pinggy_host}; '
            f'sleep 10; '
            f'done'
        )

    def get_public_url(self) -> str:
        """Return the public HTTPS URL for the tunnel."""
        return f"https://{self.persistent_url}"


@dataclass
class LoadBalancerConfig:
    """Configuration for GKE LoadBalancer exposure."""

    namespace: str = "beta9"
    service_name: str = "beta9-gateway"
    port: int = 1994
    wait_timeout: int = 300  # 5 minutes for IP assignment

    external_ip: Optional[str] = field(default=None, repr=False)

    def get_public_url(self) -> str:
        """Return the public HTTP URL for the LoadBalancer."""
        if not self.external_ip:
            raise ValueError("External IP not yet assigned")
        return f"http://{self.external_ip}:{self.port}"


@dataclass
class ClusterSetupConfig:
    """Combined configuration for full cluster setup."""

    gke: GKEConfig
    beta9: Beta9Config
    expose_method: str = "pinggy"  # "pinggy" or "loadbalancer"
    pinggy: Optional[PinggyConfig] = None
    loadbalancer: Optional[LoadBalancerConfig] = None
    filestore: Optional[FilestoreConfig] = None  # Shared NFS storage for ReadWriteMany

    # Workflow flags
    skip_gke: bool = False
    skip_beta9: bool = False
    skip_expose: bool = False
    skip_validation: bool = False
    dry_run: bool = False
    verbose: bool = False

    def __post_init__(self):
        if self.expose_method == "pinggy" and self.pinggy is None:
            raise ValueError("PinggyConfig required when expose_method='pinggy'")
        if self.expose_method == "loadbalancer" and self.loadbalancer is None:
            self.loadbalancer = LoadBalancerConfig(
                namespace=self.beta9.namespace,
                service_name=self.beta9.gateway_service_name,
                port=self.beta9.gateway_http_port,
            )
        # Default Filestore config if not provided
        if self.filestore is None:
            self.filestore = FilestoreConfig()
