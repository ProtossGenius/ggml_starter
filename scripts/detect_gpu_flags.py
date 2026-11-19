#!/usr/bin/env python3
"""Detect host platform and GPU vendor to emit ggml CMake backend flags."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys


SUPPORTED_OS = {"linux", "darwin"}

BACKEND_ALIASES = {
    "auto": "auto",
    "nvidia": "nvidia",
    "cuda": "nvidia",
    "amd": "amd",
    "hip": "amd",
    "rocm": "amd",
    "mac": "mac",
    "metal": "mac",
    "apple": "mac",
    "cpu": "cpu",
    "none": "cpu",
    "off": "cpu",
}

_DISPLAY_LINE_RE = re.compile(
    r"(?:vga compatible controller|3d controller|display controller|\[03(?:00|01|02|80)\])",
    re.IGNORECASE,
)


def normalize_os(system_name: str) -> str:
    """Normalize OS name to 'linux' or 'darwin', rejecting other platforms."""
    normalized = system_name.strip().lower()
    if normalized not in SUPPORTED_OS:
        raise ValueError(
            f"Unsupported OS '{system_name}'. Only Linux and macOS are supported."
        )
    return normalized


def normalize_backend(raw: str) -> str:
    """Normalize user-supplied backend name or alias."""
    key = raw.strip().lower()
    if key not in BACKEND_ALIASES:
        supported = "auto, nvidia (cuda), amd (hip/rocm), mac (metal), cpu (none/off)"
        raise ValueError(
            f"Unsupported GPU backend '{raw}'. Supported values: {supported}."
        )
    return BACKEND_ALIASES[key]


def validate_backend_for_os(system_name: str, backend: str) -> None:
    """Ensure the selected backend is compatible with the target OS."""
    if backend == "mac" and system_name != "darwin":
        raise ValueError("Metal ('mac') backend is only supported on macOS.")
    if backend in ("nvidia", "amd") and system_name == "darwin":
        raise ValueError(
            f"Backend '{backend}' is not supported on macOS; use 'mac' (Metal) or 'cpu'."
        )


def _run_cmd(cmd: list[str], timeout: float = 3.0) -> tuple[int, str, str]:
    """Run a probe command safely, returning (returncode, stdout, stderr)."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except (FileNotFoundError, PermissionError, subprocess.TimeoutExpired, OSError):
        return -1, "", ""


def _is_executable_file(path: Path) -> bool:
    """Return True if path exists as a regular executable file."""
    return path.is_file() and os.access(path, os.X_OK)


def _find_rocm_tool(name: str) -> str | None:
    """Locate a ROCm CLI executable in PATH, $ROCM_PATH/bin, or /opt/rocm/bin."""
    in_path = shutil.which(name)
    if in_path:
        return in_path

    search_roots: list[Path] = []
    rocm_env = os.environ.get("ROCM_PATH", "").strip()
    if rocm_env:
        search_roots.append(Path(rocm_env))
    search_roots.append(Path("/opt/rocm"))

    for root in search_roots:
        candidate = root / "bin" / name
        if _is_executable_file(candidate):
            return str(candidate)
    return None


def _find_rocm_clangxx() -> str | None:
    """Find ROCm's LLVM clang++ for HIP compilation (never returning hipcc)."""
    candidates: list[Path] = []

    hipconfig = _find_rocm_tool("hipconfig")
    if hipconfig:
        rc, stdout, _ = _run_cmd([hipconfig, "--hipclangpath"])
        clang_dir = stdout.strip()
        if rc == 0 and clang_dir:
            candidates.append(Path(clang_dir) / "clang++")

    search_roots: list[Path] = []
    rocm_env = os.environ.get("ROCM_PATH", "").strip()
    if rocm_env:
        search_roots.append(Path(rocm_env))
    search_roots.append(Path("/opt/rocm"))

    for root in search_roots:
        candidates.append(root / "lib" / "llvm" / "bin" / "clang++")
        candidates.append(root / "llvm" / "bin" / "clang++")

    for candidate in candidates:
        candidate_str = str(candidate)
        if any(ch.isspace() for ch in candidate_str):
            continue
        if _is_executable_file(candidate):
            return candidate_str
    return None


def has_nvidia_runtime() -> bool:
    """Return True if an active NVIDIA GPU driver/runtime is detected."""
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        rc, stdout, _ = _run_cmd([nvidia_smi, "-L"])
        return rc == 0 and "GPU " in stdout
    return Path("/dev/nvidia0").exists()


def detect_pci_gpu_vendors() -> set[str]:
    """Detect NVIDIA/AMD display controller vendors from sysfs or lspci."""
    vendors: set[str] = set()
    pci_dir = Path("/sys/bus/pci/devices")
    if pci_dir.is_dir():
        try:
            devices = sorted(pci_dir.iterdir())
        except OSError:
            devices = []

        for dev in devices:
            try:
                cls = (dev / "class").read_text(encoding="utf-8", errors="ignore").strip().lower()
                vendor = (dev / "vendor").read_text(encoding="utf-8", errors="ignore").strip().lower()
            except OSError:
                continue

            if not cls.startswith("0x03"):
                continue
            if vendor == "0x10de":
                vendors.add("nvidia")
            elif vendor == "0x1002":
                vendors.add("amd")

    if not vendors:
        lspci = shutil.which("lspci")
        if lspci:
            rc, stdout, _ = _run_cmd([lspci, "-nn"])
            if rc == 0:
                for line in stdout.splitlines():
                    if not _DISPLAY_LINE_RE.search(line):
                        continue
                    lower = line.lower()
                    if "[10de:" in lower or "nvidia" in lower:
                        vendors.add("nvidia")
                    if (
                        "[1002:" in lower
                        or "amd" in lower
                        or "ati" in lower
                        or "advanced micro devices" in lower
                    ):
                        vendors.add("amd")

    return vendors


def has_amd_rocm_runtime(pci_vendors: set[str]) -> bool:
    """Return True if an AMD GPU is available through ROCm/KFD."""
    rocminfo = _find_rocm_tool("rocminfo")
    if rocminfo:
        rc, stdout, _ = _run_cmd([rocminfo])
        if rc == 0 and re.search(r"Device Type:\s+GPU", stdout):
            return True

    rocm_smi = _find_rocm_tool("rocm-smi")
    if rocm_smi:
        rc, stdout, _ = _run_cmd([rocm_smi, "--showid"])
        if rc == 0 and re.search(r"GPU\[", stdout):
            return True

    # Guard against AMD Ryzen iGPU + NVIDIA dGPU hosts where /dev/kfd exists for the APU.
    if Path("/dev/kfd").exists() and "amd" in pci_vendors and "nvidia" not in pci_vendors:
        return True
    return False


def detect_gpu(system_name: str) -> str:
    """Auto-detect the GPU backend ('nvidia', 'amd', 'mac', or 'cpu')."""
    if system_name == "darwin":
        return "mac"
    if system_name != "linux":
        raise ValueError(f"Unsupported OS '{system_name}'.")

    if has_nvidia_runtime():
        return "nvidia"

    pci_vendors = detect_pci_gpu_vendors()
    if has_amd_rocm_runtime(pci_vendors):
        return "amd"
    if "nvidia" in pci_vendors:
        return "nvidia"
    if "amd" in pci_vendors:
        return "amd"
    return "cpu"


def build_cmake_flags(system_name: str, backend: str) -> tuple[list[str], list[str]]:
    """Return (cmake_flags, warnings) for the resolved OS and backend."""
    validate_backend_for_os(system_name, backend)

    flags: list[str] = []
    warnings: list[str] = []

    if backend == "nvidia":
        flags.append("-DGGML_CUDA=ON")
        if "CUDACXX" not in os.environ and not shutil.which("nvcc"):
            default_nvcc = Path("/usr/local/cuda/bin/nvcc")
            if _is_executable_file(default_nvcc):
                flags.append(f"-DCMAKE_CUDA_COMPILER={default_nvcc}")
            else:
                warnings.append(
                    "NVIDIA GPU selected, but 'nvcc' was not found. Make sure CUDA Toolkit is installed."
                )
    elif backend == "amd":
        flags.append("-DGGML_HIP=ON")
        if "HIPCXX" not in os.environ:
            clangxx = _find_rocm_clangxx()
            if clangxx:
                flags.append(f"-DCMAKE_HIP_COMPILER={clangxx}")
            else:
                rocm_root = Path(os.environ.get("ROCM_PATH", "").strip() or "/opt/rocm")
                if not (_find_rocm_tool("hipconfig") or rocm_root.exists()):
                    warnings.append(
                        "AMD GPU selected, but ROCm/HIP toolchain was not found. Make sure ROCm (>= 6.1) is installed."
                    )
    elif backend == "mac":
        flags.append("-DGGML_METAL=ON")
    elif backend == "cpu":
        if system_name == "darwin":
            # ggml enables Metal by default on Apple unless explicitly disabled.
            flags.append("-DGGML_METAL=OFF")
    else:
        raise ValueError(f"Unexpected backend '{backend}'.")

    return flags, warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Detect OS and GPU vendor to emit ggml CMake backend flags."
    )
    parser.add_argument(
        "-b",
        "--backend",
        default="auto",
        help="GPU backend override: auto, nvidia (cuda), amd (hip/rocm), mac (metal), cpu (none/off).",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress informational detection log on stderr.",
    )
    args = parser.parse_args(argv)

    try:
        system_name = normalize_os(platform.system())
        requested = normalize_backend(args.backend)
        if requested == "auto":
            backend = detect_gpu(system_name)
        else:
            validate_backend_for_os(system_name, requested)
            backend = requested
        flags, warnings = build_cmake_flags(system_name, backend)
    except ValueError as exc:
        print(f"[detect_gpu_flags] Error: {exc}", file=sys.stderr)
        return 1

    flags_str = " ".join(flags)
    if not args.quiet:
        display_flags = flags_str if flags_str else "<none>"
        print(
            f"[detect_gpu_flags] os={system_name} backend={backend} flags={display_flags}",
            file=sys.stderr,
        )
    for warning in warnings:
        print(f"[detect_gpu_flags] Warning: {warning}", file=sys.stderr)

    print(flags_str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
