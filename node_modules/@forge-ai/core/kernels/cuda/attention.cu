/**
 * CUDA Attention Kernel
 * ======================
 * 
 * Flash Attention implementation for NVIDIA GPUs.
 */

// Constants
#define BLOCK_SIZE 256
#define WARP_SIZE 32

// Softmax kernel
__global__ void softmax_kernel(float* input, float* output, int rows, int cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    if (row >= rows) return;
    
    __shared__ float shared_max;
    __shared__ float shared_sum;
    
    // Find max for numerical stability
    float local_max = -INFINITY;
    for (int i = tid; i < cols; i += blockDim.x) {
        local_max = fmaxf(local_max, input[row * cols + i]);
    }
    
    // Reduce to find global max
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, stride));
        }
    }
    
    if (tid == 0) shared_max = local_max;
    __syncthreads();
    
    // Compute exp and sum
    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float val = expf(input[row * cols + i] - shared_max);
        output[row * cols + i] = val;
        local_sum += val;
    }
    
    // Reduce sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            local_sum += __shfl_down_sync(0xffffffff, local_sum, stride);
        }
    }
    
    if (tid == 0) shared_sum = local_sum;
    __syncthreads();
    
    // Normalize
    for (int i = tid; i < cols; i += blockDim.x) {
        output[row * cols + i] /= shared_sum;
    }
}

// Matrix multiplication kernel (tiled)
__global__ void matmul_kernel(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
        // Load tiles into shared memory
        if (row < M && t * BLOCK_SIZE + tx < K) {
            As[ty][tx] = A[row * K + t * BLOCK_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (t * BLOCK_SIZE + ty < K && col < N) {
            Bs[ty][tx] = B[(t * BLOCK_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// GELU activation kernel
__global__ void gelu_kernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float x = input[idx];
        // GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        output[idx] = x * cdf;
    }
}

// Layer normalization kernel
__global__ void layer_norm_kernel(
    float* input, float* output, float* gamma, float* beta,
    int batch_size, int hidden_size, float eps
) {
    int batch = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch >= batch_size) return;
    
    __shared__ float shared_mean;
    __shared__ float shared_var;
    
    // Compute mean
    float local_sum = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        local_sum += input[batch * hidden_size + i];
    }
    
    // Reduce
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            local_sum += __shfl_down_sync(0xffffffff, local_sum, stride);
        }
    }
    
    if (tid == 0) shared_mean = local_sum / hidden_size;
    __syncthreads();
    
    // Compute variance
    float local_var = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float diff = input[batch * hidden_size + i] - shared_mean;
        local_var += diff * diff;
    }
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            local_var += __shfl_down_sync(0xffffffff, local_var, stride);
        }
    }
    
    if (tid == 0) shared_var = local_var / hidden_size;
    __syncthreads();
    
    // Normalize
    float std_inv = rsqrtf(shared_var + eps);
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float normalized = (input[batch * hidden_size + i] - shared_mean) * std_inv;
        output[batch * hidden_size + i] = gamma[i] * normalized + beta[i];
    }
}

// Flash Attention forward kernel (simplified)
__global__ void flash_attention_forward(
    const float* Q, const float* K, const float* V,
    float* O, float* L, float* M,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale
) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int tid = threadIdx.x;
    
    extern __shared__ float shared[];
    float* s_q = shared;
    float* s_k = shared + head_dim;
    float* s_v = shared + 2 * head_dim;
    float* s_o = shared + 3 * head_dim;
    
    int offset = (b * num_heads + h) * seq_len * head_dim;
    
    // Simplified: compute attention for each query position
    for (int i = 0; i < seq_len; i++) {
        // Load query
        if (tid < head_dim) {
            s_q[tid] = Q[offset + i * head_dim + tid];
            s_o[tid] = 0.0f;
        }
        __syncthreads();
        
        float m_i = -INFINITY;
        float l_i = 0.0f;
        
        // Process keys
        for (int j = 0; j <= i; j++) { // Causal mask
            // Load key and value
            if (tid < head_dim) {
                s_k[tid] = K[offset + j * head_dim + tid];
                s_v[tid] = V[offset + j * head_dim + tid];
            }
            __syncthreads();
            
            // Compute attention score
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                score += s_q[d] * s_k[d];
            }
            score *= scale;
            
            // Update running max and sum
            float m_new = fmaxf(m_i, score);
            float l_new = expf(m_i - m_new) * l_i + expf(score - m_new);
            
            // Update output
            if (tid < head_dim) {
                s_o[tid] = (expf(m_i - m_new) * l_i * s_o[tid] + 
                           expf(score - m_new) * s_v[tid]) / l_new;
            }
            
            m_i = m_new;
            l_i = l_new;
            __syncthreads();
        }
        
        // Store output
        if (tid < head_dim) {
            O[offset + i * head_dim + tid] = s_o[tid];
        }
        
        // Store statistics
        if (tid == 0) {
            L[(b * num_heads + h) * seq_len + i] = l_i;
            M[(b * num_heads + h) * seq_len + i] = m_i;
        }
        __syncthreads();
    }
}
