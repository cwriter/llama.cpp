# SYCL: graph-boundary dtypes and fusion opportunities for `qwen4exp` (IQ3_XXS)

Tree: HEAD `5757288f0` (`topk_radix_select` lineage, base `49c0dc82b`). Analysis date 2026-09-10.
Model: `unsloth/Qwen3.8-Flash-Next-GGUF`, snapshot `38bb39ee9`, directory **`UD-Q3_K_XL`** —
this is the flavour whose *experts* are IQ3_XXS. There is no `UD-IQ3_XXS` directory in the
HF cache; see §2.

Companion docs: [IQ_TYPE_FUSION.md](IQ_TYPE_FUSION.md) (IQ types in the fused MoE GEMV
dispatch — a prerequisite for §6.2 below), [TOP_K_ANALYSIS.md](TOP_K_ANALYSIS.md).

---

## 0. Method

The graph was dumped **without touching a GPU**, so it can be reproduced at any time while the
cards are busy:

```bash
source /opt/intel/oneapi/setvars.sh
M=~/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/\
38bb39ee97821de2c9009abb7e93950eec396e66/UD-Q3_K_XL/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf

GGML_SCHED_DEBUG=2 llama-completion -m $M --device none -ngl 0 \
    -c 256 -n 1 -p "hello" --no-warmup -t 16 -v > sched_cpu.log 2>&1
```

`GGML_SCHED_DEBUG=2` makes `ggml_backend_sched_print_assignments()`
(`ggml/src/ggml-backend.cpp:997`) emit one line per node: index, op, name, byte size, backend,
and every src. The log contains 16 graph dumps; the **last** one is the bs=1 decode graph
(`model.input_embed` = 10 KiB = one token). View ops are skipped by the printer, so the counts
below are *executed* nodes.

The tensor census in §2 comes from a standalone GGUF header parser (pure Python, no numpy on
this host) reading splits 2 and 3 — split 1 is metadata only (0 tensors, 10.9 MB).

CPU-only dumping is legitimate here because graph *construction* is backend-agnostic:
`llama_model_qwen4exp::graph` emits the same nodes regardless of where they are placed. Only
backend *assignment* and split counts differ, and those are not what this document is about.

---

## 1. The claim under test

> llama.cpp builds compute graphs where the boundaries are always converted to f32 first.
> Fusions help in some parts by removing this translation boundary.

**Partly true, but the mechanism is not what the phrasing suggests.** Three separate things are
being conflated. Taking them apart is what makes the fusion list in §6 fall out.

### 1a. "Graph boundaries are f32" — TRUE, and structurally so

Every activation edge in a ggml graph is a materialised F32 tensor in the compute buffer. The
matmul constructors hard-code it, independent of how `src0` is quantised:

```c
// ggml/src/ggml.c:3348   ggml_mul_mat
const int64_t ne[4] = { a->ne[1], b->ne[1], b->ne[2], b->ne[3] };
struct ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

// ggml/src/ggml.c:3407   ggml_mul_mat_id
const int64_t ne[4] = { as->ne[1], ids->ne[0], b->ne[2], 1 };
struct ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);
```

So every node is *read f32 from VRAM → compute → write f32 to VRAM*. But this is **not a
conversion**: f32 is the native activation format throughout. Nothing is being translated at
the boundary, and calling it a "translation boundary" points the optimisation effort at the
wrong thing.

### 1b. There ARE two real conversion boundaries on SYCL — fusion removes only one

**Decode (`src1->ne[1] <= MMVQ_MAX_BATCH_SIZE`, = 8):** the activation is converted f32 → q8_1
by a **separate kernel launch** before each mat-vec — `ggml-sycl.cpp:3920`, `:4017`, and inside
the fused MoE path at `:5533-5543`:

```c
// ggml/src/ggml-sycl/ggml-sycl.cpp:5533   ggml_sycl_mul_mat_id_mmvq_fused
ggml_sycl_pool_alloc<char> src1_q8_alloc(ctx.pool(), ...);
quantize_row_q8_1_sycl<quantize_q8_1>((const float *) src1->data, src1_ddq, ...);
```

`ffn_moe_gate` and `ffn_moe_up` receive the **identical** `src1` (both are `build_moe_ffn`'s
`cur`), and each one runs its own `quantize_row_q8_1_sycl` over it. **This** is a genuine
redundant conversion, and it is exactly what a gate+up fusion removes. It is the only place
the original claim is literally correct.

**Prefill (batch > 8):** far worse, and fusion cannot touch it.

```c
// ggml/src/ggml-sycl/ggml-sycl.cpp:4485
inline bool ggml_sycl_supports_mmq(enum ggml_type type) {
    // TODO: accuracy issues in MMQ
    GGML_UNUSED(type);
    return false;
}
```

`use_mul_mat_q` is therefore **always false** — there is no quantised-domain GEMM on the SYCL
backend at all. Every large-batch matmul falls through `ggml_sycl_mul_mat()` to
`ggml_sycl_op_mul_mat_sycl()`, which dequantises the whole weight tile into a pool buffer
before calling oneDNN:

```c
// ggml/src/ggml-sycl/ggml-sycl.cpp:3437
const to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src0->type, dst);
size_t ne = row_diff*ne00;
src0_as_f16.alloc(ne);
to_fp16_sycl(src0_dd_i, src0_as_f16.get(), ne, stream);     // <- the real conversion boundary
```

IQ3_XXS has a converter (`convert.cpp:722`), so this path is taken, not refused. For one
expert matrix `[2560 x 640]`:

| | bytes |
|---|---|
| IQ3_XXS source (3.0625 bpw) | 0.63 MB |
| f16 scratch written | 3.28 MB |
| f16 scratch read back by the GEMM | 3.28 MB |
| **moved, to multiply 0.63 MB of weights** | **7.19 MB — 11.4x** |

MoE makes it worse: for `ne12 != 1`, `ggml_sycl_mul_mat_id()` (`ggml-sycl.cpp:5626`) does a
host `memcpy` of the ids, a **full `stream->wait()` pipeline drain**, a CPU counting sort, and
then one `ggml_sycl_mul_mat()` per expert — each of which dequantises that expert in full.

**No fusion removes this.** It needs either a real MMQ (quantised-domain tiled GEMM, i.e.
fixing the `return false` above) or a dequantise-in-prologue GEMM that never round-trips the
f16 copy through VRAM.

### 1c. "Fusions remove the translation boundary" — mostly FALSE as stated

What the SYCL fusions actually remove is the **kernel launch plus the VRAM round-trip of the
f32 intermediate**. Look at what `ggml_sycl_op_add_add_fused` or `ggml_sycl_op_rms_norm_fused`
do: same dtypes in, same dtype out, one launch instead of two, intermediate never written.
The mul_mat+mul_mat+GLU family is the single exception that *also* removes a conversion (the
duplicate q8_1 quantise from §1b).

This distinction matters for prioritisation: if fusion removed conversions, the payoff would
scale with bytes converted. It removes *dispatches*, so the payoff scales with **node count**,
which is why §5 counts nodes and §3 establishes that this model is dispatch-bound.

---

## 2. What is actually on disk

There is no `IQ3_XXS` directory in `~/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF`.
Two flavours are present, `UD-IQ4_XS` and `UD-Q3_K_XL`; the one whose experts are IQ3_XXS is
**`UD-Q3_K_XL`**. (`llama-b60-perf-findings.md` refers to it as "UD-IQ3_XXS" — same file, the
label is off.) Tensor census read from the GGUF headers:

| tensor | type(s) | shape |
|---|---|---|
| `blk.N.ffn_gate_exps` / `ffn_up_exps` | **IQ3_XXS x94**, IQ4_XS x2 | `[2560, 640, 512]` |
| `blk.N.ffn_down_exps` | IQ4_NL x43, Q8_0 x5 | `[640, 2560, 512]` |
| `blk.N.ffn_{gate,up,down}_shexp` | Q8_0 x48 each | `[2560, 640]` / `[640, 2560]` |
| `blk.N.ffn_gate_inp` (router) | **F32** x48 | `[2560, 512]` |
| `blk.N.attn_{qkv,gate}` (GDN layers) | Q8_0 x36 each | `[2560, 10240]` / `[2560, 6144]` |
| `blk.N.attn_{q,k,v,output}` (attn layers) | Q8_0 x12 each | |
| `blk.N.hc_{attn,ffn}_{down,up}` | Q8_0 x48 each | `[10240, 320]` / `[320, 10240]` |
| `blk.N.indexer.{q,k}_proj` | BF16 x12 each | |
| `blk.N.ssm_out` | Q8_0 x36 | `[6144, 2560]` |
| `output.weight` | Q6_K | `[2560, 248320]` |
| `per_layer_token_embd.weight` | IQ4_NL | `[160, 320001536]` = 26.82 GiB |

Totals: **83.81 GiB** on disk (splits 2+3 = 89.99 GB), of which the PLE table is 26.82 GiB
(host-resident via `-ot` + lazy read), leaving **56.99 GiB of LM weights** — 3.4 GiB less than
`UD-IQ4_XS` (60.42 GiB), i.e. more headroom on the 3-card `-ncmoe 0` config.

Architecture (from the KV block): 48 layers, `full_attention_interval = 4` → 12 full-attention
layers (idx 3,7,...,47) + 36 gated-delta-net layers; 512 experts, 10 routed + 1 shared,
`expert_feed_forward_length = 640`; hyper-connections `hc = 4`, low rank 320;
sparse-attention indexer with `top_k = 2048`.

---

## 3. The decode graph is dispatch-bound, not bandwidth-bound

This is the load-bearing measurement: it is what makes node-count reduction the right lever.

**Graph size (measured, bs=1):** `graph nodes = 8022`, of which **5054 execute** (the rest are
views/reshapes). **4799 of the 5054 (95%) write <= 64 KiB.** Total activation bytes written per
token: ~299 MiB.

Op histogram of the bs=1 graph:

```
ADD 1016   MUL_MAT 797   MUL 564   SCALE 471   SIGMOID 326   RMS_NORM 256
CONT 251   GET_ROWS 223  MUL_MAT_ID 144  CPY 144  SILU 134   REPEAT 98
SWIGLU 96  SUM_ROWS 49   CLAMP 49   SOFT_MAX 48  ARGSORT 48  DIV 48
ROPE 48    CONCAT 37     SSM_CONV 36  SOFTPLUS 36  GATED_DELTA_NET 36
SET_ROWS 36  FILL 24     RELU 12    TOP_K 12   FLASH_ATTN_EXT 12
SGN 1      ABS 1         SQRT 1
```

**Weight traffic per decoded token** (computed from the tensor census, 10/512 experts active,
PLE excluded as a host-side gather):

| tensor group | MB/token |
|---|---|
| `attn_qkv` (36 GDN layers) | 1002.7 |
| `attn_gate` | 601.6 |
| `ssm_out` | 601.6 |
| `output.weight` | 521.5 |
| `ffn_down_exps` (IQ4_NL, 10/512) | 483.3 |
| `attn_q` (12 attn layers) | 401.1 |
| `ffn_gate_exps` + `ffn_up_exps` (**IQ3_XXS**, 10/512) | 607.0 |
| `ffn_gate_inp` (router, **F32**) | 251.7 |
| `attn_output` | 200.5 |
| `hc_{attn,ffn}_{down,up}` | 668.4 |
| everything else | ~440 |
| **total** | **5.77 GB** |

At 456 GB/s — and `--split-mode layer` serialises the cards, so one card's bandwidth is the
budget — that is **12.6 ms/token, a 79 t/s roofline**.

Measured (3 cards, `-ncmoe 0`, `llama-b60-perf-findings.md`): **13.62 t/s = 73.4 ms/token**.

> **Weight streaming is 17% of the token. The other ~61 ms is dispatch.**
> 61 ms spread over the ~4290 dispatches that survive today's fusions (§4) is **~14 us of
> non-bandwidth time per node**, ~17 us per node all-in. That is the right order for a Level
> Zero kernel launch on an in-order queue with `GGML_SYCL_ENABLE_GRAPH=0` (forced off by
> multi-GPU, and `MUL_MAT_ID` disables SYCL graphs on its own — see `check_graph_compatibility`
> at `ggml-sycl.cpp:6553`).

Note also that each quantised-weight matmul costs a **second** launch for its q8_1 quantise
(§1b), which the node count above does not include — roughly 700-900 more dispatches per token.

This also explains the shape of the already-measured `-ncmoe` sweep: moving 8 whole layers of
experts from host to VRAM bought only 8.17 -> 9.56 t/s. Expert residency is a bandwidth lever
against a cost that is 83% not bandwidth.

---

## 4. What the SYCL backend fuses today

Dispatch order in `ggml_backend_sycl_graph_compute` (`ggml-sycl.cpp:6476-6526`):

| rule | site | hits in this graph | nodes saved |
|---|---|---|---|
| `GATED_DELTA_NET` + `CPY` (state scatter) | `:6479` | 36 | 36 |
| `{RMS_NORM, MUL, ADD}` | `:6487` | 0 | 0 |
| `{RMS_NORM, MUL}` | `:6493` | 184 | 184 |
| `{ADD, ADD}` | `:6499` | 314 pairs | 314 |
| `{UNARY, MUL}` (silu/sigmoid/softplus only) | `:6505` | 230 | 230 |
| `L2_NORM` sibling batching | `:6515` | 0 (no L2_NORM in qwen4exp) | 0 |
| `{MUL_MAT, MUL_MAT, GLU}` | `:6522` | **0** | 0 |
| | | **total** | **764** |

**5054 -> 4290 dispatched nodes.**

The two GLU fusions both miss this model, for different reasons:

- `{MUL_MAT, MUL_MAT, GLU}` is gated to Q4_K only —
  `ggml/src/ggml-sycl/fusion.cpp:34`: `if (wu->type != GGML_TYPE_Q4_K || wu->ne[0] % QK_K != 0) return false;`
  The shared expert here is **Q8_0**. The gate exists because the fused kernel is the
  *reorder/SoA* GEMV, which so far only Q4_K has.
- The MoE experts go through `MUL_MAT_ID`, which that rule does not match at all.

CUDA, for comparison, handles both: the fusion loop at `ggml-cuda.cu:3683` runs
`for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID })`, and its type gate
(`ggml_cuda_should_fuse_mul_mat_vec_q`, `:1786`) asks only `ggml_is_quantized(src0->type)` —
IQ3_XXS included.

---

## 5. Ranked fusion opportunities

Counted by replaying the measured bs=1 node list through a greedy left-to-right matcher in the
same order the backend applies its rules.

| # | pattern | hits | nodes saved | risk |
|---|---|---|---|---|
| 1 | **multi-`ADD` up to 8** (generalise `add_add_fused` 2 -> N) | 158 runs | **253** | low |
| 2 | **`{MUL_MAT_ID, MUL_MAT_ID, SWIGLU}`** (MoE gate+up) | 48 | **96** | medium |
| 3 | **`{SCALE, UNARY, SCALE}`** | 96 | **192** | low |
| 4 | **`{MUL, ADD}`** | 147 | **147** | low |
| 5 | **`{MUL_MAT, MUL_MAT, SWIGLU}` for non-Q4_K** (shared expert) | 48 | **96** | medium |
| 6 | **`{SCALE, UNARY}`** | 97 | **97** | low |
| | **combined 1-6** | | **4290 -> 3409, -20.5%** | |
| 7 | hc-stream mean: `CONT,ADD,ADD,ADD,SCALE` -> one strided-mean kernel | 97 | ~291 (subsumes part of 1) | medium |

If per-dispatch cost is the dominant term, -20.5% of dispatches maps to roughly **+15-20%
decode**, i.e. ~13.6 -> ~16 t/s at `-ncmoe 0`. That is an estimate, not a measurement; §7 says
how to calibrate it before committing to any of this.

**Honest caveat:** the matcher checks the op sequence only, not `ggml_can_fuse`'s single-use /
same-shape / not-a-view / not-an-output predicates. The counts are therefore upper bounds. Items
1, 2 and 5 were additionally verified by hand against the dump and against `qwen4exp.cpp`, and
those three are structurally sound.

---

## 6. The opportunities in detail

### 6.1 Multi-`ADD` (253 nodes) — highest count, lowest risk

`ggml_sycl_op_add_add_fused` folds exactly **two** adds. CUDA folds up to **eight**
(`ggml-cuda.cu:3591-3610`):

```c
if (node->op == GGML_OP_ADD || node->op == GGML_OP_MUL) {
    int n_fuse = 0; ggml_op ops[8]; std::fill(ops, ops + 8, node->op);
    for (; n_fuse <= 6; ++n_fuse) {
        if (!ggml_can_fuse(cgraph, i + n_fuse, ops + n_fuse, 2)) break;
        if (cgraph->nodes[i + n_fuse] != cgraph->nodes[i + n_fuse + 1]->src[0]) break;
        if (!ggml_are_same_layout(cgraph->nodes[i + n_fuse]->src[1],
                                  cgraph->nodes[i + n_fuse + 1]->src[1])) break;
    }
    n_fuse++;
```

ADD-run length histogram in this graph: `{1: 243, 2: 1, 3: 97, 4: 12, 9: 48}`.

The **48 runs of 9** are the MoE expert reduction. `build_moe_ffn` sums the 10 weighted expert
outputs with a 9-deep chain of `[2560]` f32 adds; from the dump (layer 1, nodes 375-383) every
one has `src[0]` = the running sum and `src[1]` = a stride-`nb1` view of `ffn_moe_weighted`:

```
node #375 (ADD): node_375 (10K) <- ffn_moe_weighted-1 (10K), ffn_moe_weighted-1 (10K)
node #376 (ADD): node_376 (10K) <- node_375 (10K),          ffn_moe_weighted-1 (10K)
...
node #383 (ADD): ffn_moe_out-1  (10K) <- node_382 (10K),    ffn_moe_weighted-1 (10K)
```

All `src[1]` share a layout, so CUDA's predicate accepts all eight. Per layer:

| | launches | traffic |
|---|---|---|
| unfused | 9 | 270 KB |
| SYCL today (`ADD,ADD`) | 5 | 190 KB |
| CUDA-style (<=8) | 2 | 120 KB |
| ideal (single reduction) | 1 | 110 KB |

The **97 runs of 3** are the hyper-connection stream mean (§6.4).

#### Do the pairs stack? Yes — sequentially, and they always fire. But not as a tree.

Three questions worth settling before touching this, because they change how much is left to win.

**They do stack.** The dispatch loop is a plain `for (int i = 0; i < cgraph->n_nodes; i++)`
(`ggml-sycl.cpp:6454`), and the `{ADD, ADD}` arm ends in `i++; continue;` (`:6499-6503`), so a
fused pair advances the cursor by 2 and the *next* pair is tested immediately. A run of 9
therefore consumes as `(375,376) (377,378) (379,380) (381,382) (383)` = **5 dispatches**, not
"one fusion, then 7 unfused adds". Repeated application already happens.

**They always fire, at least here.** Counting the whole graph from the run-length histogram,
`0*243 + 1*1 + 1*97 + 2*12 + 4*48 = 314` — which is *exactly* the 314 recorded in §4. Every
eligible pair in this model fuses; none are declined by `ggml_can_fuse`'s single-use /
same-shape / not-a-view / not-an-output predicates, by SYCL's extra `add1->src[0] != add0`
check (`fusion.cpp:158`), or by `ggml_sycl_add_kernel_supports`. So there is no latent win from
*fixing* the existing rule — the only win is from *widening* it.

**It cannot become a tree, and that does not matter.** `ggml_can_fuse_ext` requires each node to
be a src of the next (`ggml-impl.h:698`), so a fusable run is by construction a **linear
dependency chain**: `a1 = e0+e1`, `a2 = a1+e2`, ... `a9 = a8+e9`. There is no parallel structure
to fold into a log-depth tree without restructuring `build_moe_ffn` itself. CUDA does not build
a tree either — `ggml_cuda_op_fused_add` is an **N-ary linear fold**: one kernel reading up to
8 operand pointers (`fused_node.src[j + 2] = cgraph->nodes[i + j + 1]->src[1]`,
`ggml-cuda.cu:3613`) and writing the last node's `dst` once. Depth is irrelevant when the whole
run is 90 KB of f32 and the cost is the dispatch; **operand count per kernel is the only lever.**

**Implementation shape.** `ggml_sycl_op_add_add_fused` (`binbcast.cpp:594`) is hard-wired to
three operands — it destructures `add0->src[0]`, `add0->src[1]`, `add1->src[1]` and dispatches
`launch_bin_bcast3` through an explicit ~15-way type cross-product (F32, F16, mixed F16/F32,
I32, I16, BF16, mixed BF16/F32). Widening that entire matrix to N would be a large and mostly
dead diff. **Every one of the 158 runs in this graph is all-F32**, so the cheap version is an
N-ary F32-only `launch_bin_bcastN` fast path, leaving the existing 3-operand kernel untouched as
the fallback for every other type combination — additive, in the same spirit as
`IQ_TYPE_FUSION.md` §3.

Otherwise this is a straight port of a proven upstream pattern — no new maths.

### 6.2 `{MUL_MAT_ID, MUL_MAT_ID, SWIGLU}` (96 nodes) — the IQ3_XXS-specific one

The MoE block, verbatim from the dump (layer 1):

```
360 MUL_MAT_ID ffn_moe_gate-1     <- ffn_gate_exps  IQ3_XXS
361 MUL_MAT_ID ffn_moe_up-1       <- ffn_up_exps    IQ3_XXS   (same src1 as 360)
362 SWIGLU     ffn_moe_swiglu-1
363 MUL_MAT_ID ffn_moe_down-1     <- ffn_down_exps  IQ4_NL
364 MUL        ffn_moe_weighted-1
375-383 ADD x9                    <- see 6.1
```

Fusing 360-362 removes, per layer per token:

1. one `MUL_MAT_ID` dispatch,
2. one `SWIGLU` dispatch,
3. **one redundant `quantize_row_q8_1_sycl` over the identical activation** (§1b) — the only
   place in this whole document where a fusion deletes a genuine format conversion,
4. one `ggml_sycl_pool_alloc` for the duplicate q8_1 buffer,
5. two `[640 x 10]` f32 intermediates (25.6 KB each) written and read back.

x48 layers = **144 dispatches and ~2.5 MB of round-trip per token**.

Prerequisites are already in this tree: `IQ_TYPE_FUSION.md` added IQ3_XXS to
`ggml_sycl_mul_mat_vec_q_id`'s dispatch table (`mmvq.cpp:2868`), so the fused MoE GEMV accepts
this model's experts today. The gates in `ggml_sycl_mul_mat_id_mmvq_fused` (`ggml-sycl.cpp:5504`)
all pass at decode: `ne12 == 1`, `ne10 = 2560` and `640`, both `% QK8_1 == 0`, `ids->ne[1] == 1`,
`ne11` is 1 for gate/up and 10 (`== n_ids_per_group`) for down.

The work is to thread a second weight pointer + a GLU epilogue through
`mul_mat_vec_q_moe`, mirroring CUDA's `ggml_cuda_mm_fusion_args_host{ .gate, .glu_op, .glu_limit }`.
Structurally identical to what `ggml_sycl_mul_mat_glu_mmvq_fused` already does for the dense
case — but against the generic (non-reorder) `vec_dot`, which every IQ type has.

### 6.3 `{SCALE, UNARY, SCALE}` (192 nodes) and `{MUL, ADD}` (147 nodes)

Both live in `build_hc_combine` (`src/models/qwen4exp.cpp:314`), which runs twice per layer
(after attention and after FFN) — 96 calls:

```c
ggml_tensor * w = ggml_sigmoid(ctx0, ggml_scale(ctx0, inject, 1.0f / (float) hc));
w = ggml_scale(ctx0, w, 2.0f);
...
ggml_tensor * cur = ggml_add(ctx0, residual, ggml_mul(ctx0, b, w));
```

i.e. `2*sigmoid(x/4)` as three elementwise dispatches over a `[4]`-wide tensor, then a
multiply-add. CUDA already has this exact shape but gated to TANH
(`ggml-cuda.cu:4169`: `{ GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE }` with
`GGML_UNARY_OP_TANH`); generalising the unary is a switch-case, and `ggml_sycl_op_unary_mul_fused`
(`element_wise.cpp:1035`) already dispatches SILU/SIGMOID/SOFTPLUS by lambda, so the pattern
for doing so is in the file.

`{MUL, ADD}` -> fused multiply-add is a standalone elementwise kernel with no type subtleties.

### 6.4 `{SCALE, UNARY}` (97) and the hc-stream mean (~291)

`build_hc_mix` (`src/models/qwen4exp.cpp:266`) runs 97 times (2/layer + the final head mixer):

```
329 RMS_NORM              \ fused today
331 MUL       hc_norm-1   /
332 MUL_MAT   (hc_down)
333 SCALE                 \ {SCALE, UNARY} - not fused  (silu(x/hc))
334 SILU                  /
335 MUL_MAT   (hc_up)
336 SIGMOID   hc_gate-1   \ fused today ({UNARY, MUL})
337 MUL                   /
340 CONT                  \
342 ADD                    |
344 ADD                    |  the hc=4 stream mean: 5 dispatches to reduce
346 ADD                    |  [2560 x 4] -> [2560] and scale by 1/4
347 SCALE     hc_mixed-1  /
```

`{SCALE, UNARY}` is trivial (97 nodes). The mean is the bigger prize: `ggml_can_fuse` cannot
express it because `mixed` starts as a `ggml_cont` of a strided `ggml_view_2d` and the adds
consume further views of the same parent. A dedicated strided-mean kernel collapses 5 -> 1;
folding it into the preceding `{SIGMOID, MUL}` fusion collapses 7 -> 1, giving one kernel that
gates and reduces all four hyper-connection streams. This one is qwen4exp-specific and will
never be upstream-driven — but at 97 calls/token it is the single largest item in the table.

### 6.5 `{MUL_MAT, MUL_MAT, SWIGLU}` for non-Q4_K (96 nodes)

The shared expert, immediately after the MoE block:

```
384 MUL_MAT ffn_gate-1            <- ffn_gate_shexp  Q8_0
385 MUL_MAT ffn_up-1              <- ffn_up_shexp    Q8_0
386 SWIGLU  ffn_swiglu-1
387 MUL_MAT ffn_shexp-1           <- ffn_down_shexp  Q8_0
388 MUL_MAT shared_expert_gate-1  <- ffn_gate_inp_shexp  F32
389 SIGMOID
390 MUL     ffn_shexp_gated-1
391 ADD     ffn_out-1
```

384-386 is precisely the shape `ggml_sycl_mul_mat_glu_mmvq_fused` handles, declined only by the
Q4_K gate at `fusion.cpp:34`. Two ways forward:

- **Narrow:** Q8_0 is already in `ggml_sycl_supports_reorder_mmvq` (`ggml-sycl.cpp:4528`), so
  extend the reorder-SoA fused GEMV to Q8_0 super-blocks. Fastest kernel, most work.
- **General (preferred):** add a non-reorder fused variant built on the generic
  `vec_dot_*_q8_1`, which exists for **every** type including IQ3_XXS. Lower peak throughput
  than the SoA kernel, but it covers Q8_0 here *and* becomes the mechanism §6.2 needs for
  `MUL_MAT_ID`. Doing 6.5-general first makes 6.2 mostly a re-parameterisation.

---

## 7. What to measure before implementing any of this

The whole ranking rests on "a dispatch costs ~14-17 us here". Calibrate it first — it is two
runs and it decides whether the list is worth 20% or 5%:

```bash
# A/B the fusions that already exist (764 nodes' worth, §4)
GGML_SYCL_ENABLE_FUSION=1 llama-bench -m $M <production flags> -p 0 -n 32
GGML_SYCL_ENABLE_FUSION=0 llama-bench -m $M <production flags> -p 0 -n 32
```

`g_ggml_sycl_enable_fusion` (`ggml-sycl.cpp:103`, read at `:343`) gates every rule in §4,
including the `GATED_DELTA_NET`+`CPY` one. The delta divided by 764 is the measured value of
one eliminated dispatch on this hardware. Multiply by 881 (the §5 total) for a grounded
prediction, and only then decide.

Two operational cautions from `llama-b60-perf-findings.md` that apply to that run:
`-ncmoe 0 -ts 31,33,36` is the validated 3-card split (an "even" 16/16/16 puts SYCL0 135 MiB
over and collapses to 0.15 t/s), and never `kill -9` a wedged run without first confirming the
DRM engines are idle.

Also worth re-checking after any fusion lands: `graph splits @ bs=1` was **28** at `-ncmoe 0`,
and those residual splits are the host-side PLE gather — they do not shrink with fusion, so
they set a floor on what §5 can deliver.

---

## 8. Implementation plan — one fusion per branch

**Each item in §5 gets its own branch off the current HEAD, and its own A/B number.** Not one
"fusion work" branch. Reasons, in order of how much they have already cost on this tree:

1. **Attribution.** The §5 estimate is `nodes_saved x cost_per_dispatch`, and
   `cost_per_dispatch` is itself an estimate (§7). The only way to learn which items were worth
   doing is to measure each against the same baseline. Bundled, a +12% result cannot be split
   across six changes, and a *regression* in one silently eats the gain of another.
2. **Risk is not uniform.** §5 marks 1/3/4/6 low-risk (elementwise, no new maths, existing type
   plumbing) and 2/5/7 medium (new fused GEMV kernels, quantised weights, a GLU epilogue,
   numerics that differ from two separate `vec_dot` passes). Those must not share a bisect
   range: a numerical regression in the MoE gate+up kernel should never be tangled with a
   multiply-add on hyper-connections.
3. **Upstreamability differs per item.** 1, 3, 4 and 6 are CUDA-parity ports and are plausible
   standalone upstream PRs. 7 is qwen4exp-specific and will never be upstream-driven. 2 and 5
   are large enough to want their own review. Branch boundaries should match PR boundaries.
4. **`IQ_TYPE_FUSION.md` §3 already established the house style** and the argument applies one
   level up: a patch that only adds cases *cannot* regress anything, and mixing in an unrelated
   cleanup turns it into one that could.

Suggested order — cheapest and most independent first, so each lands against a baseline that is
still moving predictably:

| order | branch | §5 item | nodes | depends on |
|---|---|---|---|---|
| 1 | `fuse_add_n` | multi-`ADD` <= 8 | 253 | — |
| 2 | `fuse_scale_unary` | `{SCALE, UNARY}` + `{SCALE, UNARY, SCALE}` | 289 | — |
| 3 | `fuse_mul_add` | `{MUL, ADD}` | 147 | — |
| 4 | `fuse_glu_generic` | `{MUL_MAT, MUL_MAT, SWIGLU}`, non-Q4_K | 96 | — |
| 5 | `fuse_glu_mmid` | `{MUL_MAT_ID, MUL_MAT_ID, SWIGLU}` | 96 | 4 (reuses its generic `vec_dot` fused GEMV) |
| 6 | `fuse_hc_mean` | hc-stream mean | ~291 | 1 (re-measure: it subsumes the 97 runs of 3) |

Only 5-on-4 and 6-on-1 are real dependencies; the rest are independent and can be measured in
any order.

**Do not add the six numbers.** §5's -20.5% is not the sum of six independently-measured deltas:
items 1 and 6 overlap on the same 97 ADD runs, and greedy matching means an earlier rule can
consume nodes a later one would have claimed. Re-run the full greedy count (§0 method) against a
fresh dump after each merge instead.
