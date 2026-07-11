from __future__ import annotations

import importlib.metadata as metadata
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def run_command(command: list[str]) -> str:
    try:
        return subprocess.check_output(
            command,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        return f"Unavailable ({error})"


def cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "Unknown"


def total_memory_gb() -> str:
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemTotal:"):
                memory_kb = int(line.split()[1])
                return f"{memory_kb / 1024**2:.2f} GB"
    return "Unknown"


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


print("=== Repository ===")
print("Git commit:", run_command(["git", "rev-parse", "--short", "HEAD"]))
print("Git branch:", run_command(["git", "branch", "--show-current"]))

print("\n=== Operating system ===")
print("Hostname:", platform.node())
print("OS:", platform.platform())
print("Python:", sys.version.replace("\n", " "))

print("\n=== CPU and memory ===")
print("CPU model:", cpu_model())
print("Logical CPU cores:", os.cpu_count())
print("System memory:", total_memory_gb())

disk = shutil.disk_usage(Path.cwd())
print(f"Free disk space in current filesystem: {disk.free / 1024**3:.2f} GB")

print("\n=== GPU ===")
print(
    run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )
)

print("\n=== Relevant Python packages ===")
for package in ["numpy", "pandas", "matplotlib", "gymnasium", "gym", "tqdm"]:
    print(f"{package}: {package_version(package)}")