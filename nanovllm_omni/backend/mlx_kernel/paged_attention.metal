// Paged decode attention + KV store for the MLX track (ADR 0021 / 0023).
//
// THIS FILE IS THE SOURCE OF TRUTH, and it is a fragment, not a translation
// unit: it holds the kernel *bodies* and the shared helpers, never a
// signature. ``kernels.py`` reads it and hands each body to
// ``mx.fast.metal_kernel``; ``build.py`` compiles MLX's generated translation
// unit with ``xcrun metal`` so the syntax gate checks the bytes that run.
//
// Why no signature here: ``metal_kernel`` builds one from ``input_names`` /
// ``output_names``, appends whichever ``thread_*`` attributes the body
// mentions, and wraps the body in ``#include <metal_stdlib>`` +
// ``using namespace metal``. A hand-written copy would be a second source that
// can drift; leaving it out makes drift impossible.
//
// Why model constants are template ints: ``metal_kernel`` turns exactly
// ``input_names`` into buffers, so a scalar has nowhere else to live. Templates
// accept ints, dtypes and bools only -- no floats. Values that change every
// step (pool rows, token count, scale) therefore ride in small int32/float32
// arrays: a template int that changes retraces the kernel on every new value.

// ===========================================================================
// HELPERS  (shared; passed to ``metal_kernel`` as ``header=``)
// ===========================================================================
// HEADER_BEGIN

// One simdgroup per (sequence, head) = 32 lanes. ``simd_sum`` reduces across
// exactly these lanes, so this width is load-bearing, not a tuning knob.
constant int LANES = 32;

// Which KV head a query head reads. GQA groups are contiguous in both the fork
// and qwen3 (heads 0..G-1 read kv head 0, and so on).
inline int kv_head_of(int head, int num_heads, int num_kv_heads) {
    return head / (num_heads / num_kv_heads);
}

// HEADER_END

// ===========================================================================
// paged_decode_attention
// ===========================================================================
// One threadgroup per (sequence, query head). Each lane owns ``D / LANES``
// elements of the head dimension, so D must be a multiple of 32 (96 = 3 * 32).
//
//   q            [B, H,    D]  input dtype
//   k_cache      [S, H_kv, D]  input dtype, S = rows in the pool
//   v_cache      [S, H_kv, D]  input dtype
//   block_table  [B, NB]       int32, -1 marks a hole
//   context_lens [B]           int32, past keys this sequence has
//   scale        [1]           float32, D ** -0.5 in practice
//   meta         [1]           int32, block_table's row stride
//   out          [B, H,    D]  input dtype
//
// Template ints: D, NUM_HEADS, NUM_KV_HEADS, BLOCK_SIZE, and T (the dtype).
// Threads: ``threadgroup_position_in_grid`` (MLX's ``grid`` is a *thread*
// count, not a threadgroup count) and ``thread_position_in_threadgroup``.

// BODY_BEGIN paged_decode_attention
uint group = threadgroup_position_in_grid.x;
uint lane = thread_position_in_threadgroup.x;
int head = group % NUM_HEADS;
int seq = group / NUM_HEADS;
int kv_head = kv_head_of(head, NUM_HEADS, NUM_KV_HEADS);
int max_blocks = meta[0];
int context_len = context_lens[seq];
int out_row = (seq * NUM_HEADS + head) * D + lane;

if (context_len > 0) {
    // The slice of q this lane owns, held in registers across the whole loop.
    float q_lane[D / LANES];
    for (int i = 0; i < D / LANES; ++i) {
        q_lane[i] = (float)q[out_row + i * LANES];
    }

    // Online softmax state. All fp32: summing thousands of keys in fp16 loses
    // the tail long before the accumulator overflows.
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float acc[D / LANES];
    for (int i = 0; i < D / LANES; ++i) {
        acc[i] = 0.0f;
    }

    for (int pos = 0; pos < context_len; ++pos) {
        // pos -> pool row: one indirection through the block table.
        int block = block_table[seq * max_blocks + pos / BLOCK_SIZE];
        if (block != -1) {
            int row = (block * BLOCK_SIZE + pos % BLOCK_SIZE) * NUM_KV_HEADS * D
                    + kv_head * D;

            float partial = 0.0f;
            for (int i = 0; i < D / LANES; ++i) {
                partial += q_lane[i] * (float)k_cache[row + lane + i * LANES];
            }
            // Each lane scored its own slice; the dot product is their sum.
            float score = simd_sum(partial) * scale[0];

            // Rescale the accumulator onto the new maximum before adding this
            // key, so the running sum never overflows.
            float new_max = metal::max(running_max, score);
            float correction = metal::exp(running_max - new_max);
            float weight = metal::exp(score - new_max);
            running_sum = running_sum * correction + weight;
            running_max = new_max;

            for (int i = 0; i < D / LANES; ++i) {
                float v = (float)v_cache[row + lane + i * LANES];
                acc[i] = acc[i] * correction + weight * v;
            }
        }
        // block == -1 is a hole, not a block. Reading it would index pool row
        // -1, i.e. in front of the array; skipping keeps holes unreadable.
    }

    // Zero can only happen if every key scored -inf; 0 beats NaN there.
    float normaliser = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;
    for (int i = 0; i < D / LANES; ++i) {
        out[out_row + i * LANES] = (T)(acc[i] * normaliser);
    }
} else {
    // An empty context has no keys to average; the softmax of an empty set is
    // NaN, so write zero.
    for (int i = 0; i < D / LANES; ++i) {
        out[out_row + i * LANES] = (T)0;
    }
}
// BODY_END paged_decode_attention

// ===========================================================================
// store_kv
// ===========================================================================
// Copy the pool, replacing the rows this step's tokens map to. One thread per
// pool element; ``grid.x`` = pool_rows * NUM_KV_HEADS * D.
//
//   k_cache, v_cache  [S, H_kv, D]  input dtype, the pool as it stands
//   k, v              [T, H_kv, D]  input dtype, this step's fresh rows
//   slot_mapping      [T]           int32, -1 skips the token
//   meta              [2]           int32, [pool_rows, num_tokens]
//   k_out, v_out      [S, H_kv, D]  the pool with those rows replaced
//
// The whole pool is rewritten because ``mx.fast.metal_kernel`` cannot write
// in place. The pool is both an input and an output so the caller can rebind
// the returned array. ADR 0023 measured that ``mx.compile`` does not donate a
// still-referenced pool, so compilation does not remove this copy.

// BODY_BEGIN store_kv
int pool_rows = meta[0];
int num_tokens = meta[1];
uint per_token = (uint)(NUM_KV_HEADS * D);
uint tid = thread_position_in_grid.x;
if (tid < (uint)pool_rows * per_token) {
    uint row = tid / per_token;

    // Is this row one of the slots written this step? Tokens per step is tiny
    // (at most the batch size), so a linear scan beats an inverted index.
    int source = -1;
    for (int t = 0; t < num_tokens; ++t) {
        if (slot_mapping[t] == (int)row) {
            source = t;
            break;
        }
    }

    if (source < 0) {
        k_out[tid] = k_cache[tid];
        v_out[tid] = v_cache[tid];
    } else {
        uint from = (uint)source * per_token + tid % per_token;
        k_out[tid] = k[from];
        v_out[tid] = v[from];
    }
}
// BODY_END store_kv
