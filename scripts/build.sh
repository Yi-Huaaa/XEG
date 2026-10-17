#!/usr/bin/env bash
# Configure and build XEG (CP-SAT backend).
#
# Usage:
#   scripts/build.sh [target ...]      # default target: simple
#
# Environment (all optional):
#   XEG_BUILD_DIR   build directory (default: <repo>/build)
#   XEG_CUDA_ARCH   CUDA architecture, e.g. 86 or 120
#                   (default: detected from the local GPU by CMake)
#   ORTOOLS_HOME    OR-Tools C++ install directory
#                   (default: see g_ext/CMakeLists.txt)
#
# nvcc and a host C/C++ compiler supported by it must be on PATH
# (or set CUDACXX / CC / CXX).
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${XEG_BUILD_DIR:-${repo_root}/build}"
targets=("$@")
if [ "${#targets[@]}" -eq 0 ]; then
  targets=(simple)
fi

cmake_args=(-S "${repo_root}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release)
if [ -n "${XEG_CUDA_ARCH:-}" ]; then
  cmake_args+=("-DCMAKE_CUDA_ARCHITECTURES=${XEG_CUDA_ARCH}")
fi

cmake "${cmake_args[@]}"
cmake --build "${build_dir}" --target "${targets[@]}" -j "$(nproc)"
