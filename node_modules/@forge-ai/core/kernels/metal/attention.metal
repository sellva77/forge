/**
 * Metal Attention Shaders
 * ========================
 * 
 * GPU shaders for Apple Silicon (M1/M2/M3).
 */

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// SOFTMAX
// =============================================================================

kernel void softmax_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    uint row = gid.y;
    if (row >= rows) return;
    
    // Find max for numerical stability
    float max_val = -INFINITY;
    for (uint i = 0; i < cols; i++) {
        max_val = max(max_val, input[row * cols + i]);
    }
    
    // Compute exp and sum
    float sum = 0.0;
    for (uint i = 0; i < cols; i++) {
        float exp_val = exp(input[row * cols + i] - max_val);
        output[row * cols + i] = exp_val;
        sum += exp_val;
    }
    
    // Normalize
    for (uint i = 0; i < cols; i++) {
        output[row * cols + i] /= sum;
    }
}

// =============================================================================
// MATRIX MULTIPLICATION
// =============================================================================

constant uint TILE_SIZE = 16;

kernel void matmul_kernel(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    // Tiled matrix multiplication
    threadgroup float As[TILE_SIZE][TILE_SIZE];
    threadgroup float Bs[TILE_SIZE][TILE_SIZE];
    
    uint row = tgid.y * TILE_SIZE + tid.y;
    uint col = tgid.x * TILE_SIZE + tid.x;
    
    float sum = 0.0;
    
    uint numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (uint t = 0; t < numTiles; t++) {
        // Load tiles into shared memory
        uint aCol = t * TILE_SIZE + tid.x;
        uint bRow = t * TILE_SIZE + tid.y;
        
        if (row < M && aCol < K) {
            As[tid.y][tid.x] = A[row * K + aCol];
        } else {
            As[tid.y][tid.x] = 0.0;
        }
        
        if (bRow < K && col < N) {
            Bs[tid.y][tid.x] = B[bRow * N + col];
        } else {
            Bs[tid.y][tid.x] = 0.0;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute partial dot product
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += As[tid.y][k] * Bs[k][tid.x];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// =============================================================================
// GELU ACTIVATION
// =============================================================================

kernel void gelu_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    float x = input[gid];
    // GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    float cdf = 0.5 * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
    output[gid] = x * cdf;
}

// =============================================================================
// LAYER NORMALIZATION
// =============================================================================

kernel void layer_norm_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* gamma [[buffer(2)]],
    device const float* beta [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    constant uint& hidden_size [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint batch = gid.y;
    if (batch >= batch_size) return;
    
    // Compute mean
    float mean = 0.0;
    for (uint i = 0; i < hidden_size; i++) {
        mean += input[batch * hidden_size + i];
    }
    mean /= hidden_size;
    
    // Compute variance
    float var = 0.0;
    for (uint i = 0; i < hidden_size; i++) {
        float diff = input[batch * hidden_size + i] - mean;
        var += diff * diff;
    }
    var /= hidden_size;
    
    // Normalize
    float std_inv = rsqrt(var + eps);
    for (uint i = 0; i < hidden_size; i++) {
        float normalized = (input[batch * hidden_size + i] - mean) * std_inv;
        output[batch * hidden_size + i] = gamma[i] * normalized + beta[i];
    }
}

// =============================================================================
// ATTENTION
// =============================================================================

kernel void attention_scores_kernel(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device float* scores [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    constant uint& num_heads [[buffer(4)]],
    constant uint& seq_len [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant float& scale [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint b = gid.z;
    uint h = gid.y;
    uint i = gid.x;
    
    if (b >= batch_size || h >= num_heads || i >= seq_len) return;
    
    uint offset = (b * num_heads + h) * seq_len * head_dim;
    
    for (uint j = 0; j < seq_len; j++) {
        float score = 0.0;
        for (uint d = 0; d < head_dim; d++) {
            score += Q[offset + i * head_dim + d] * K[offset + j * head_dim + d];
        }
        scores[(b * num_heads + h) * seq_len * seq_len + i * seq_len + j] = score * scale;
    }
}

kernel void attention_output_kernel(
    device const float* probs [[buffer(0)]],
    device const float* V [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    constant uint& num_heads [[buffer(4)]],
    constant uint& seq_len [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint b = gid.z;
    uint h = gid.y;
    uint i = gid.x;
    
    if (b >= batch_size || h >= num_heads || i >= seq_len) return;
    
    uint v_offset = (b * num_heads + h) * seq_len * head_dim;
    uint p_offset = (b * num_heads + h) * seq_len * seq_len;
    uint o_offset = (b * num_heads + h) * seq_len * head_dim;
    
    for (uint d = 0; d < head_dim; d++) {
        float sum = 0.0;
        for (uint j = 0; j < seq_len; j++) {
            sum += probs[p_offset + i * seq_len + j] * V[v_offset + j * head_dim + d];
        }
        output[o_offset + i * head_dim + d] = sum;
    }
}

// =============================================================================
// EMBEDDINGS
// =============================================================================

kernel void embedding_kernel(
    device const float* embeddings [[buffer(0)]],
    device const uint* input_ids [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    constant uint& seq_len [[buffer(4)]],
    constant uint& embed_dim [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint batch = gid.z;
    uint seq = gid.y;
    uint dim = gid.x;
    
    if (batch >= batch_size || seq >= seq_len || dim >= embed_dim) return;
    
    uint token_id = input_ids[batch * seq_len + seq];
    output[batch * seq_len * embed_dim + seq * embed_dim + dim] = embeddings[token_id * embed_dim + dim];
}

// =============================================================================
// ADAMW OPTIMIZER
// =============================================================================

kernel void adamw_kernel(
    device float* params [[buffer(0)]],
    device const float* grads [[buffer(1)]],
    device float* m [[buffer(2)]],
    device float* v [[buffer(3)]],
    constant float& lr [[buffer(4)]],
    constant float& beta1 [[buffer(5)]],
    constant float& beta2 [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    constant float& weight_decay [[buffer(8)]],
    constant uint& step [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    float g = grads[gid];
    
    // Update biased moments
    m[gid] = beta1 * m[gid] + (1.0 - beta1) * g;
    v[gid] = beta2 * v[gid] + (1.0 - beta2) * g * g;
    
    // Bias correction
    float m_hat = m[gid] / (1.0 - pow(beta1, float(step)));
    float v_hat = v[gid] / (1.0 - pow(beta2, float(step)));
    
    // Update with weight decay
    params[gid] -= lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * params[gid]);
}
