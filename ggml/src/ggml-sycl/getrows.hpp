//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef GGML_SYCL_GETROWS_HPP
#define GGML_SYCL_GETROWS_HPP

#include "common.hpp"

void ggml_sycl_op_get_rows(ggml_backend_sycl_context & ctx, ggml_tensor *dst);

// CONT(PERMUTE(score)) -> GET_ROWS -> PERMUTE -> CONT at node_idx: the QSA indexer gather.
// can_fuse() is structural only, so ggml-alloc and the compute loop always agree on it.
bool ggml_sycl_can_fuse_qsa_gather(const ggml_cgraph * cgraph, int node_idx);
int  ggml_sycl_fuse_qsa_gather(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int node_idx);

#endif // GGML_SYCL_GETROWS_HPP
