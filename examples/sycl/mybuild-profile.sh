#!/usr/bin/env bash
#  MIT license
#  Copyright (C) 2024 Intel Corporation
#  SPDX-License-Identifier: MIT

# Profiling build: same options as mybuild.sh plus device debug info, into build-profile/
# so the measured Release build in build/ is never disturbed. Device code stays JIT, the
# same path build/ uses, so this differs from the measured build in debug info alone.

# No -u: oneAPI's vars.sh reads OCL_ICD_FILENAMES and friends while unset, and nounset
# makes sourcing it fail with "unbound variable".
set -eo pipefail

# The cmake source dir is the repo root, two levels up from examples/sycl.
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${SRC_DIR}/build-profile"

source /opt/intel/oneapi/setvars.sh

# Device debug info needs -Xarch_device -g. Verified on oneAPI 2026.1 by object size:
# no flags 37120 B, -Xarch_device -gline-tables-only 37120 B (accepted but emits NOTHING),
# -Xarch_device -g 79400 B. Plain -gline-tables-only is host-only for SYCL and icpx warns
# "not currently supported for target spir64-unknown-unknown". These are icpx driver flags:
# do NOT put them in GGML_SYCL_DEVICE_ARCH, which is spliced into ocloc's -device argument
# and makes ocloc fail with "Invalid option (arg 9): -gline-tables-only".
#
# -Xarch_device -g is full device debug info, not line tables, so it can change device
# register allocation. Treat this build as attribution-only and keep taking throughput
# numbers from build/.
DEVICE_DEBUG="-Xarch_device -g"

# host side: line tables are enough to resolve host frames
PROFILE_FLAGS="-march=native -O3 -gline-tables-only -fdebug-info-for-profiling ${DEVICE_DEBUG}"

mkdir -p "${BUILD_DIR}"
cmake -S "${SRC_DIR}" -B "${BUILD_DIR}" \
 -DCMAKE_BUILD_TYPE=Release \
 -DCMAKE_C_FLAGS="${PROFILE_FLAGS}" \
 -DCMAKE_CXX_FLAGS="${PROFILE_FLAGS}" \
 -DCMAKE_EXE_LINKER_FLAGS="-fsycl -lur_loader" \
 -DGGML_SYCL_DEVICE_ARCH="" \
 -DGGML_VULKAN=ON -DGGML_SYCL=ON -DGGML_SYCL_DNN=ON \
 -DGGML_NATIVE=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_SYCL_F16=ON -DLLAMA_OPENSSL=ON

cmake --build "${BUILD_DIR}" --config Release -j16

echo "profiling build in ${BUILD_DIR}/bin (build/ untouched)"
