#!/usr/bin/env bash
#  MIT license
#  Copyright (C) 2024 Intel Corporation
#  SPDX-License-Identifier: MIT

mkdir -p build
cd build
#source /opt/intel/oneapi/setvars.sh
source /opt/intel/oneapi/setvars.sh

#DGGML_SYCL_DEVICE_ARCH="bmg-g31 -options -ze-opt-large-register-file"

# -DGGML_SYCL_TARGET=INTEL # -DGGML_SYCL_DEVICE_ARCH="bmg-g31" 
#-DGGML_SYCL_DEVICE_ARCH="bmg-g21 -gline-tables-only -fdebug-info-for-profiling"
#for FP16
cmake ..  -DCMAKE_C_FLAGS="-march=native -O3"  \
 -DCMAKE_CXX_FLAGS="-march=native -O3"  \
 -DCMAKE_EXE_LINKER_FLAGS="-fsycl -lur_loader"  \
 -DGGML_VULKAN=ON  -DGGML_SYCL=ON  -DGGML_SYCL_DNN=ON  \
 -DGGML_NATIVE=ON  -DCMAKE_C_COMPILER=icx  -DCMAKE_CXX_COMPILER=icpx  -DGGML_SYCL_F16=ON  -DLLAMA_OPENSSL=ON
# faster for long-prompt inference

#for FP32
#cmake .. -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DLLAMA_OPENSSL=ON

#build example/main
#cmake --build . --config Release --target main

#build example/llama-bench
#cmake --build . --config Release --target llama-bench

#build all binary
cmake --build . --config Release -j16 -v
