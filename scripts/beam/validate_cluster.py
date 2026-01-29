"""Sandbox health check validation for Beta9 cluster."""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a single validation test."""

    test_name: str
    passed: bool
    duration_sec: float
    error: Optional[str] = None
    stdout: Optional[str] = None


@dataclass
class ValidationReport:
    """Full validation report."""

    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    results: list[ValidationResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total_tests == 0:
            return 0.0
        return self.passed / self.total_tests

    def add_result(self, result: ValidationResult):
        self.results.append(result)
        self.total_tests += 1
        if result.passed:
            self.passed += 1
        else:
            self.failed += 1

    def summary(self) -> str:
        lines = [
            f"Validation Report: {self.passed}/{self.total_tests} passed ({self.success_rate:.0%})",
            "-" * 50,
        ]
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            lines.append(f"  [{status}] {r.test_name} ({r.duration_sec:.2f}s)")
            if r.error:
                lines.append(f"         Error: {r.error}")
        return "\n".join(lines)


def configure_beta9_endpoint(gateway_url: str):
    """Configure beta9 SDK to use custom gateway endpoint.

    Args:
        gateway_url: Full URL to Beta9 gateway (e.g., https://mybeam.a.pinggy.link).
    """
    # Beta9 SDK uses environment variables for configuration
    os.environ["BETA9_GATEWAY_HOST"] = gateway_url.replace("https://", "").replace("http://", "").rstrip("/")

    # Also set for potential HTTP API usage
    os.environ["BETA9_API_URL"] = gateway_url.rstrip("/")

    logger.info(f"Configured beta9 endpoint: {gateway_url}")


def validate_sandbox_lifecycle(gateway_url: str, test_id: str = None) -> ValidationResult:
    """Test sandbox create -> exec -> terminate lifecycle.

    Args:
        gateway_url: Beta9 gateway URL.
        test_id: Optional test identifier.

    Returns:
        ValidationResult with test outcome.
    """
    test_name = f"sandbox_lifecycle_{test_id or uuid4().hex[:6]}"
    start_time = time.time()

    try:
        # Import beta9 SDK
        from beta9 import Image, PythonVersion, Sandbox

        # Configure endpoint
        configure_beta9_endpoint(gateway_url)

        # Create sandbox
        logger.info(f"[{test_name}] Creating sandbox...")
        sandbox = Sandbox(
            name=f"health-check-{uuid4().hex[:8]}",
            image=Image(python_version=PythonVersion.Python311),
            cpu=1,
            memory=512,
            env={"TEST_VAR": "beam-health-check"},
        )

        instance = sandbox.create()
        logger.info(f"[{test_name}] Sandbox created")

        # Execute test command
        logger.info(f"[{test_name}] Executing test command...")
        test_string = f"hello-beam-{uuid4().hex[:6]}"

        # Use async execution
        async def run_command():
            process = await instance.aio.process.exec("echo", test_string)
            exit_code = await process.wait()
            stdout = process._sync.stdout.read()
            return exit_code, stdout

        exit_code, stdout = asyncio.run(run_command())

        # Verify output
        if exit_code != 0:
            raise RuntimeError(f"Command exited with code {exit_code}")

        if test_string not in stdout:
            raise RuntimeError(f"Expected '{test_string}' in output, got: {stdout}")

        logger.info(f"[{test_name}] Command executed successfully")

        # Terminate sandbox
        logger.info(f"[{test_name}] Terminating sandbox...")
        instance.terminate()
        logger.info(f"[{test_name}] Sandbox terminated")

        duration = time.time() - start_time
        return ValidationResult(
            test_name=test_name,
            passed=True,
            duration_sec=duration,
            stdout=stdout,
        )

    except ImportError as e:
        duration = time.time() - start_time
        return ValidationResult(
            test_name=test_name,
            passed=False,
            duration_sec=duration,
            error=f"beta9 SDK not installed: {e}",
        )
    except Exception as e:
        duration = time.time() - start_time
        return ValidationResult(
            test_name=test_name,
            passed=False,
            duration_sec=duration,
            error=str(e),
        )


def validate_sandbox_isolation(gateway_url: str) -> ValidationResult:
    """Test that sandboxes are properly isolated.

    Creates two sandboxes and verifies they cannot see each other's files.

    Args:
        gateway_url: Beta9 gateway URL.

    Returns:
        ValidationResult with test outcome.
    """
    test_name = "sandbox_isolation"
    start_time = time.time()

    try:
        from beta9 import Image, PythonVersion, Sandbox

        configure_beta9_endpoint(gateway_url)

        # Create first sandbox and write a file
        sandbox1 = Sandbox(
            name=f"isolation-test-1-{uuid4().hex[:6]}",
            image=Image(python_version=PythonVersion.Python311),
            cpu=1,
            memory=512,
        )
        instance1 = sandbox1.create()

        secret_value = f"secret-{uuid4().hex}"

        async def write_file():
            process = await instance1.aio.process.exec(
                "bash", "-c", f"echo '{secret_value}' > /tmp/secret.txt"
            )
            await process.wait()

        asyncio.run(write_file())

        # Create second sandbox and try to read the file
        sandbox2 = Sandbox(
            name=f"isolation-test-2-{uuid4().hex[:6]}",
            image=Image(python_version=PythonVersion.Python311),
            cpu=1,
            memory=512,
        )
        instance2 = sandbox2.create()

        async def read_file():
            process = await instance2.aio.process.exec(
                "bash", "-c", "cat /tmp/secret.txt 2>/dev/null || echo 'FILE_NOT_FOUND'"
            )
            exit_code = await process.wait()
            stdout = process._sync.stdout.read()
            return stdout

        stdout = asyncio.run(read_file())

        # Cleanup
        instance1.terminate()
        instance2.terminate()

        # Verify isolation
        if secret_value in stdout:
            raise RuntimeError("Sandbox isolation failed: secret visible across sandboxes")

        duration = time.time() - start_time
        return ValidationResult(
            test_name=test_name,
            passed=True,
            duration_sec=duration,
        )

    except ImportError as e:
        duration = time.time() - start_time
        return ValidationResult(
            test_name=test_name,
            passed=False,
            duration_sec=duration,
            error=f"beta9 SDK not installed: {e}",
        )
    except Exception as e:
        duration = time.time() - start_time
        return ValidationResult(
            test_name=test_name,
            passed=False,
            duration_sec=duration,
            error=str(e),
        )


def run_validation_suite(
    gateway_url: str,
    num_lifecycle_tests: int = 3,
    include_isolation_test: bool = True,
) -> ValidationReport:
    """Run full validation suite.

    Args:
        gateway_url: Beta9 gateway URL.
        num_lifecycle_tests: Number of sandbox lifecycle tests to run.
        include_isolation_test: Whether to include isolation test.

    Returns:
        ValidationReport with all test results.
    """
    report = ValidationReport()

    logger.info(f"Running validation suite against: {gateway_url}")
    logger.info(f"  Lifecycle tests: {num_lifecycle_tests}")
    logger.info(f"  Isolation test: {include_isolation_test}")

    # Run lifecycle tests
    for i in range(num_lifecycle_tests):
        result = validate_sandbox_lifecycle(gateway_url, test_id=str(i + 1))
        report.add_result(result)

        # Small delay between tests
        if i < num_lifecycle_tests - 1:
            time.sleep(2)

    # Run isolation test
    if include_isolation_test:
        result = validate_sandbox_isolation(gateway_url)
        report.add_result(result)

    logger.info(f"\n{report.summary()}")
    return report


def quick_health_check(gateway_url: str) -> bool:
    """Run a single quick health check.

    Args:
        gateway_url: Beta9 gateway URL.

    Returns:
        True if health check passes.
    """
    result = validate_sandbox_lifecycle(gateway_url, test_id="quick")
    return result.passed
