/**
 * CUDA Training Kernels
 * ======================
 * 
 * Kernels for gradient computation and optimization.
 */

#define BLOCK_SIZE 256

// Cross-entropy loss kernel
__global__ void cross_entropy_loss_kernel(
    const float* logits,    // [batch, vocab_size]
    const int* targets,     // [batch]
    float* losses,          // [batch]
    int batch_size,
    int vocab_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size) {
        int target = targets[idx];
        
        // Compute log-softmax for stability
        float max_val = -INFINITY;
        for (int v = 0; v < vocab_size; v++) {
            max_val = fmaxf(max_val, logits[idx * vocab_size + v]);
        }
        
        float sum_exp = 0.0f;
        for (int v = 0; v < vocab_size; v++) {
            sum_exp += expf(logits[idx * vocab_size + v] - max_val);
        }
        
        float log_softmax = logits[idx * vocab_size + target] - max_val - logf(sum_exp);
        losses[idx] = -log_softmax;
    }
}

// Cross-entropy gradient kernel
__global__ void cross_entropy_grad_kernel(
    const float* logits,    // [batch, vocab_size]
    const int* targets,     // [batch]
    float* grad,            // [batch, vocab_size]
    int batch_size,
    int vocab_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size * vocab_size) {
        int batch = idx / vocab_size;
        int vocab = idx % vocab_size;
        int target = targets[batch];
        
        // Compute softmax
        float max_val = -INFINITY;
        for (int v = 0; v < vocab_size; v++) {
            max_val = fmaxf(max_val, logits[batch * vocab_size + v]);
        }
        
        float sum_exp = 0.0f;
        for (int v = 0; v < vocab_size; v++) {
            sum_exp += expf(logits[batch * vocab_size + v] - max_val);
        }
        
        float softmax = expf(logits[idx] - max_val) / sum_exp;
        
        // Gradient is softmax - one_hot(target)
        grad[idx] = softmax - (vocab == target ? 1.0f : 0.0f);
    }
}

// AdamW optimizer kernel
__global__ void adamw_kernel(
    float* params,          // Parameters to update
    const float* grads,     // Gradients
    float* m,               // First moment
    float* v,               // Second moment
    int size,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    int step                // Current step for bias correction
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float g = grads[idx];
        
        // Update biased moments
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;
        
        // Bias correction
        float m_hat = m[idx] / (1.0f - powf(beta1, step));
        float v_hat = v[idx] / (1.0f - powf(beta2, step));
        
        // Update with weight decay
        params[idx] -= lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * params[idx]);
    }
}

// SGD with momentum kernel
__global__ void sgd_momentum_kernel(
    float* params,
    const float* grads,
    float* velocity,
    int size,
    float lr,
    float momentum,
    float weight_decay
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float g = grads[idx] + weight_decay * params[idx];
        velocity[idx] = momentum * velocity[idx] + g;
        params[idx] -= lr * velocity[idx];
    }
}

// Gradient clipping kernel
__global__ void grad_clip_kernel(
    float* grads,
    int size,
    float max_norm,
    float* total_norm  // Output: pointer to pre-computed total norm
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float clip_coef = max_norm / ((*total_norm) + 1e-6f);
        if (clip_coef < 1.0f) {
            grads[idx] *= clip_coef;
        }
    }
}

// Compute L2 norm of gradient (reduction kernel)
__global__ void compute_grad_norm_kernel(
    const float* grads,
    float* partial_sums,
    int size
) {
    __shared__ float shared[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load and square
    float val = (idx < size) ? grads[idx] * grads[idx] : 0.0f;
    shared[tid] = val;
    __syncthreads();
    
    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        partial_sums[blockIdx.x] = shared[0];
    }
}

// Embedding lookup kernel
__global__ void embedding_kernel(
    const float* embeddings,    // [vocab_size, embed_dim]
    const int* input_ids,       // [batch, seq_len]
    float* output,              // [batch, seq_len, embed_dim]
    int batch_size,
    int seq_len,
    int embed_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * seq_len * embed_dim;
    
    if (idx < total) {
        int batch = idx / (seq_len * embed_dim);
        int seq = (idx / embed_dim) % seq_len;
        int dim = idx % embed_dim;
        
        int token_id = input_ids[batch * seq_len + seq];
        output[idx] = embeddings[token_id * embed_dim + dim];
    }
}

// Embedding gradient kernel
__global__ void embedding_grad_kernel(
    float* grad_embeddings,     // [vocab_size, embed_dim]
    const float* grad_output,   // [batch, seq_len, embed_dim]
    const int* input_ids,       // [batch, seq_len]
    int batch_size,
    int seq_len,
    int embed_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * seq_len * embed_dim;
    
    if (idx < total) {
        int batch = idx / (seq_len * embed_dim);
        int seq = (idx / embed_dim) % seq_len;
        int dim = idx % embed_dim;
        
        int token_id = input_ids[batch * seq_len + seq];
        atomicAdd(&grad_embeddings[token_id * embed_dim + dim], grad_output[idx]);
    }
}

// RoPE (Rotary Position Embedding) kernel
__global__ void rope_kernel(
    float* q,                   // [batch, seq_len, num_heads, head_dim]
    float* k,                   // [batch, seq_len, num_heads, head_dim]
    const float* cos_cached,    // [max_seq_len, head_dim/2]
    const float* sin_cached,    // [max_seq_len, head_dim/2]
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * seq_len * num_heads * head_dim;
    
    if (idx < total) {
        int batch = idx / (seq_len * num_heads * head_dim);
        int seq = (idx / (num_heads * head_dim)) % seq_len;
        int head = (idx / head_dim) % num_heads;
        int dim = idx % head_dim;
        
        int half_dim = head_dim / 2;
        
        if (dim < half_dim) {
            float cos_val = cos_cached[seq * half_dim + dim];
            float sin_val = sin_cached[seq * half_dim + dim];
            
            int q_idx = idx;
            int q_idx2 = idx + half_dim;
            
            float q1 = q[q_idx];
            float q2 = q[q_idx2];
            
            q[q_idx] = q1 * cos_val - q2 * sin_val;
            q[q_idx2] = q1 * sin_val + q2 * cos_val;
            
            float k1 = k[q_idx];
            float k2 = k[q_idx2];
            
            k[q_idx] = k1 * cos_val - k2 * sin_val;
            k[q_idx2] = k1 * sin_val + k2 * cos_val;
        }
    }
}
