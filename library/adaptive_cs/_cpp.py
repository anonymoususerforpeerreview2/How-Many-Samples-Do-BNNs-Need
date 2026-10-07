"""Private bridge between Python and the C ABI exposed by the C++ engine.

This module deliberately contains no confidence-sequence logic.  It only
finds/builds the C++ shared library, declares its ABI, and turns C++ arrays
into the small interface used by the Python trackers.
"""

from __future__ import annotations

from ctypes import POINTER, Structure, c_double, c_int, c_size_t, c_ubyte, c_void_p
import ctypes
from enum import IntEnum
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np


class _CppSettings(Structure):
    """ctypes representation of the public ``acs_settings`` C struct."""

    _fields_ = [
        ("alpha", c_double),
        ("c", c_double),
        ("theta", c_double),
        ("x_tolerance", c_double),
        ("y_tolerance", c_double),
        ("max_steps", c_int),
        ("padding", c_double),
        ("endpoint_root_mode", c_int),
        ("root_finder", c_int),
        ("simplex_intersection", c_int),
        ("validate_probabilities", c_int),
    ]


class _FutilityCriterion(IntEnum):
    """Integer values used by the C ABI's futility check."""

    CLASS = 0
    DISTRIBUTION = 1
    THRESHOLD = 2


_ROOT = Path(__file__).resolve().parent
_BUILD_DIR = _ROOT / "build"
_DLL_DIRECTORY_HANDLES: list[Any] = []


def _library_names() -> tuple[str, ...]:
    if os.name == "nt":
        return ("adaptive_cs_native.dll", "libadaptive_cs_native.dll")
    if sys.platform == "darwin":
        return ("libadaptive_cs_native.dylib", "adaptive_cs_native.dylib")
    return ("libadaptive_cs_native.so", "adaptive_cs_native.so")


def _find_library() -> Path | None:
    """Find a packaged library or a library built from this source tree."""

    # There is deliberately one supported build directory.  Do not search
    # IDE-specific or legacy build trees: stale artifacts can be built with a
    # different compiler or with different optional dependencies.
    for directory in (_ROOT, _BUILD_DIR, _BUILD_DIR / "Release"):
        for name in _library_names():
            path = directory / name
            if path.exists():
                return path
    return None


def _source_is_newer_than(library: Path) -> bool:
    """Return whether a native source/build input changed after the DLL."""

    if library.parent == _ROOT:
        return False

    inputs = (
        _ROOT / "native" / "adaptive_cs.cpp",
        _ROOT / "native" / "adaptive_cs.hpp",
        _ROOT / "CMakeLists.txt",
    )
    library_mtime = library.stat().st_mtime
    return any(path.exists() and path.stat().st_mtime > library_mtime for path in inputs)


def _windows_dependency_directories(library: Path) -> list[Path]:
    """Find directories containing DLLs needed by a locally built library."""

    directories = [library.parent]
    configured_dirs = os.environ.get("ADAPTIVE_CS_DLL_DIRS", "")
    directories.extend(
        Path(item).expanduser().resolve()
        for item in configured_dirs.split(os.pathsep)
        if item
    )

    # A CMake build may use MinGW or another compiler whose runtime is not on
    # Python's DLL search path.  Recover the compiler directory from the
    # cache next to the selected artifact.  The same cache also tells us the
    # vcpkg installation holding dynamically linked dependencies such as GSL.
    for cache in (library.parent / "CMakeCache.txt", _BUILD_DIR / "CMakeCache.txt"):
        if not cache.is_file():
            continue
        for line in cache.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("CMAKE_CXX_COMPILER:") or line.startswith("GSL_LIBRARY:"):
                value = line.split("=", 1)[-1].strip()
                path = Path(value)
                if not path.is_absolute():
                    continue
                if path.name.lower() == "gsl.lib":
                    directories.append(path.parent.parent / "bin")
                elif path.suffix.lower() in {".exe", ".bat"}:
                    directories.append(path.parent)

    # Keep the normal process PATH available as a fallback for installed
    # compiler runtimes, while add_dll_directory makes the search explicit.
    directories.extend(
        Path(item).resolve()
        for item in os.environ.get("PATH", "").split(os.pathsep)
        if item
    )
    return list(dict.fromkeys(directory for directory in directories if directory.is_dir()))


def _build_cpp_library() -> Path:
    """Build the canonical Release shared library with CMake."""

    if not (_ROOT / "CMakeLists.txt").is_file():
        library = _find_library()
        if library is not None and library.parent == _ROOT:
            return library
        raise RuntimeError(
            "The installed adaptive_cs wheel is missing its library; reinstall it."
        )

    cmake = os.environ.get("CMAKE_COMMAND") or shutil.which("cmake")
    if not cmake:
        raise RuntimeError(
            "CMake is required to build adaptive_cs; install it and add it "
            "to PATH or set CMAKE_COMMAND."
        )

    configure_command = [
        cmake,
        "-S",
        str(_ROOT),
        "-B",
        str(_BUILD_DIR),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DADAPTIVE_CS_ENABLE_OPENMP=ON",
    ]
    toolchain = os.environ.get("CMAKE_TOOLCHAIN_FILE")
    if toolchain:
        configure_command.append(f"-DCMAKE_TOOLCHAIN_FILE={toolchain}")
    for stage, command in (
        ("configuration", configure_command),
        (
            "build",
            [cmake, "--build", str(_BUILD_DIR), "--config", "Release", "--parallel"],
        ),
    ):
        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"CMake {stage} failed:\n{exc.stdout}") from exc

    library = _find_library()
    if library is None:
        raise RuntimeError(
            "CMake completed but the canonical shared library was not found in "
            f"{_BUILD_DIR}."
        )
    return library


def build_cpp() -> Path:
    """Build and return the path to the C++ confidence-sequence library."""

    return _build_cpp_library()


def _load_cpp_library() -> Any:
    library = _find_library()
    if library is None or _source_is_newer_than(library):
        library = _build_cpp_library()
    if os.name == "nt":
        # Python 3.8+ does not reliably search PATH for DLL dependencies.
        # Keep handles alive for the lifetime of the process, as required by
        # add_dll_directory on Windows.
        for directory in _windows_dependency_directories(library):
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(directory)))
    cpp_library = (ctypes.WinDLL if os.name == "nt" else ctypes.CDLL)(str(library))

    cpp_library.acs_batch_create.argtypes = [_CppSettings, c_size_t]
    cpp_library.acs_batch_create.restype = c_void_p
    cpp_library.acs_batch_destroy.argtypes = [c_void_p]
    cpp_library.acs_batch_destroy.restype = None
    cpp_library.acs_batch_update_active.argtypes = [
        c_void_p,
        POINTER(c_double),
        c_size_t,
        c_size_t,
        c_size_t,
        POINTER(c_size_t),
        c_size_t,
        c_int,
        POINTER(c_double),
        POINTER(c_double),
    ]
    cpp_library.acs_batch_update_active.restype = c_int
    cpp_library.acs_batch_futile.argtypes = [
        c_void_p,
        POINTER(c_size_t),
        c_size_t,
        c_int,
        c_double,
        c_double,
        c_size_t,
        POINTER(c_ubyte),
    ]
    cpp_library.acs_batch_futile.restype = c_int
    cpp_library.acs_trace_hedged_root.argtypes = [
        POINTER(c_double),
        POINTER(c_double),
        c_size_t,
        c_int,
        c_double,
        c_double,
        c_double,
        c_double,
        c_double,
        c_double,
        c_int,
        POINTER(c_double),
        POINTER(c_double),
        POINTER(c_size_t),
        c_size_t,
        POINTER(c_size_t),
        POINTER(c_size_t),
        POINTER(c_double),
        POINTER(c_double),
        POINTER(c_int),
    ]
    cpp_library.acs_trace_hedged_root.restype = c_int
    cpp_library.acs_last_error.argtypes = []
    cpp_library.acs_last_error.restype = ctypes.c_char_p
    return cpp_library


def _cpp_settings_struct(settings: Any, *, validate: bool) -> _CppSettings:
    return _CppSettings(
        settings.alpha,
        settings.c,
        settings.theta,
        settings.x_tolerance,
        settings.y_tolerance,
        settings.max_steps,
        settings.padding,
        int(settings.endpoint_root_mode == "componentwise"),
        {"brent": 0, "bisection": 1, "boost": 2}[settings.root_finder],
        int(settings.simplex_intersection),
        int(validate),
    )


class CppBatch:
    """
    Small Python wrapper around one C++ batch of trackers.

    CppBatch is the Python wrapper around the C++ batch tracker. It handles the Python/C++ boundary:
    - loads the shared library
    - creates a C++ BatchedTracker
    - converts NumPy arrays to contiguous C-compatible memory
    - calls functions such as acs_batch_update_active
    - retrieves C++ errors
    - destroys the C++ object when finished
    """

    def __init__(self, batch_size: int, settings: Any, *, validate: bool):
        # `validate` whether to validate probabilities
        self.cpp_library = _load_cpp_library()
        self.handle = self.cpp_library.acs_batch_create(
            _cpp_settings_struct(settings, validate=validate), batch_size
        )
        if not self.handle:
            raise RuntimeError(self._error())
        self.batch_size = batch_size

    def _error(self) -> str:
        value = self.cpp_library.acs_last_error()
        return value.decode("utf-8") if value else "unknown C++ error"

    def update(
        self,
        values: np.ndarray,
        active: np.ndarray,
        *,
        compute_bounds: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        # values: [num_draws, num_active_inputs, num_classes]
        # active: [num_active_inputs]
        # outputs: [batch_size, num_classes], [batch_size, num_classes]

        # values becomes compatible with double*, while active becomes compatible with integer indices.
        values = np.ascontiguousarray(values, dtype=np.float64)
        active = np.ascontiguousarray(active, dtype=np.int64)

        lower = np.empty((self.batch_size, values.shape[-1]), dtype=np.float64) # [batch_size, num_classes]
        upper = np.empty_like(lower)

        # Call the C++ function through the ctypes ABI.
        code = self.cpp_library.acs_batch_update_active(
            self.handle, # C++ handle. it contains configured settings (from `settings` and `validate`)
            values.ctypes.data_as(POINTER(c_double)), # pointer to the raw data in `values`

            values.shape[0], # number of draws
            self.batch_size, # batch size
            values.shape[2], # number of classes

            active.ctypes.data_as(POINTER(c_size_t)), # Pointer to the raw `active`` indices.
            active.size, # number of active inputs
            int(compute_bounds),
            lower.ctypes.data_as(POINTER(c_double)), # pointer to the output `lower`
            upper.ctypes.data_as(POINTER(c_double)), # pointer to the output `upper`
        )

        # handle c++ error (0 means success; any non-zero value is an error)
        if code:
            raise ValueError(self._error())

        return lower, upper

    def futile(
        self,
        active: np.ndarray,
        criterion: int,
        tolerance: float,
        criterion_level: float,
        max_samples: int,
    ) -> np.ndarray:
        active = np.ascontiguousarray(active, dtype=np.int64)
        output = np.empty(active.size, dtype=np.uint8)
        code = self.cpp_library.acs_batch_futile(
            self.handle,
            active.ctypes.data_as(POINTER(c_size_t)),
            active.size,
            criterion,
            tolerance,
            criterion_level,
            max_samples,
            output.ctypes.data_as(POINTER(c_ubyte)),
        )
        if code:
            raise ValueError(self._error())
        return output.astype(bool)

    def __del__(self):
        handle = getattr(self, "handle", None)
        cpp_library = getattr(self, "cpp_library", None)
        if handle and cpp_library is not None:
            cpp_library.acs_batch_destroy(handle)
            self.handle = None


class CppRootTracer:
    """Thin wrapper for the native, instrumented TOMS 748 diagnostic."""

    CALL_TYPES = {"crossing": 0, "lower": 1, "upper": 2}

    def __init__(self) -> None:
        self.cpp_library = _load_cpp_library()

    def trace(
        self,
        observations: np.ndarray,
        bets: np.ndarray,
        call_type: str,
        *,
        level: float = 1.0,
        center: float = 0.5,
        c: float = 0.75,
        theta: float = 0.5,
        x_tolerance: float = 1e-4,
        y_tolerance: float = 1e-6,
        max_steps: int = 80,
    ) -> dict[str, Any]:
        observations = np.ascontiguousarray(observations, dtype=np.float64)
        bets = np.ascontiguousarray(bets, dtype=np.float64)
        if observations.ndim != 1 or bets.shape != observations.shape:
            raise ValueError("observations and bets must be matching one-dimensional arrays")
        if call_type not in self.CALL_TYPES:
            raise ValueError(f"unknown root-call type: {call_type}")

        capacity = max_steps + 3
        lower = np.empty(capacity, dtype=np.float64)
        upper = np.empty(capacity, dtype=np.float64)
        evaluations = np.empty(capacity, dtype=np.uintp)
        point_count = c_size_t()
        total_evaluations = c_size_t()
        root_lower = c_double()
        root_upper = c_double()
        status = c_int()
        code = self.cpp_library.acs_trace_hedged_root(
            observations.ctypes.data_as(POINTER(c_double)),
            bets.ctypes.data_as(POINTER(c_double)),
            observations.size,
            self.CALL_TYPES[call_type],
            level,
            center,
            c,
            theta,
            x_tolerance,
            y_tolerance,
            max_steps,
            lower.ctypes.data_as(POINTER(c_double)),
            upper.ctypes.data_as(POINTER(c_double)),
            evaluations.ctypes.data_as(POINTER(c_size_t)),
            capacity,
            ctypes.byref(point_count),
            ctypes.byref(total_evaluations),
            ctypes.byref(root_lower),
            ctypes.byref(root_upper),
            ctypes.byref(status),
        )
        if code:
            value = self.cpp_library.acs_last_error()
            message = value.decode("utf-8") if value else "unknown C++ error"
            raise RuntimeError(message)
        count = point_count.value
        return {
            "lower": lower[:count].copy(),
            "upper": upper[:count].copy(),
            "function_evaluations": evaluations[:count].astype(np.int64),
            "total_function_evaluations": int(total_evaluations.value),
            "root_lower": float(root_lower.value),
            "root_upper": float(root_upper.value),
            "status": int(status.value),
        }
