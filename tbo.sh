#!/bin/bash
# usage: tbo.sh <bindir> <tag>
set +u; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
B=$1; T=$2; O=/root/perf_scratch/awc/tbo
mkdir -p $O; md5sum $B/libggml-sycl.so.0.25.3 $B/libggml-base.so.0.25.3 > $O/$T.md5
timeout 2400 $B/test-backend-ops test -b SYCL0 -o FLASH_ATTN_EXT,MUL_MAT,MUL_MAT_ID,QSA_SPARSE_FA,QSA_MASK_ADD > $O/$T.main.out 2>&1; echo "main rc=$?"
GGML_SYCL_FA_MAX_MEM_MIB=1 timeout 2400 $B/test-backend-ops test -b SYCL0 -o FLASH_ATTN_EXT,QSA_SPARSE_FA > $O/$T.cap1.out 2>&1; echo "cap1 rc=$?"
