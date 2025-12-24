//! Attention Module
//! =================
//!
//! Optimized multi-head attention implementation.

use rayon::prelude::*;

/// Compute scaled dot-product attention scores
pub fn compute_attention_scores(
    query: &[f32],
    key: &[f32],
    seq_len: usize,
    head_dim: usize,
    scale: f32,
) -> Vec<f32> {
    let mut scores = vec![0.0f32; seq_len * seq_len];
    
    scores.par_chunks_mut(seq_len).enumerate().for_each(|(i, row)| {
        for j in 0..seq_len {
            let mut dot = 0.0f32;
            for k in 0..head_dim {
                dot += query[i * head_dim + k] * key[j * head_dim + k];
            }
            row[j] = dot * scale;
        }
    });
    
    scores
}

/// Apply causal mask (for autoregressive models)
pub fn apply_causal_mask(scores: &mut [f32], seq_len: usize) {
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            scores[i * seq_len + j] = f32::NEG_INFINITY;
        }
    }
}

/// Softmax along rows
pub fn row_softmax(scores: &mut [f32], seq_len: usize) {
    scores.par_chunks_mut(seq_len).for_each(|row| {
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = row.iter_mut().map(|x| {
            *x = (*x - max).exp();
            *x
        }).sum();
        row.iter_mut().for_each(|x| *x /= sum);
    });
}

/// Compute attention output
pub fn compute_attention_output(
    weights: &[f32],
    value: &[f32],
    seq_len: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; seq_len * head_dim];
    
    output.par_chunks_mut(head_dim).enumerate().for_each(|(i, out_row)| {
        for j in 0..seq_len {
            let w = weights[i * seq_len + j];
            for k in 0..head_dim {
                out_row[k] += w * value[j * head_dim + k];
            }
        }
    });
    
    output
}

/// Flash Attention-style fused kernel (memory efficient)
/// Computes attention without materializing full attention matrix
pub fn flash_attention(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let head_size = seq_len * head_dim;
    let mut output = vec![0.0f32; batch_size * num_heads * seq_len * head_dim];
    
    // Process each head in parallel
    output.par_chunks_mut(head_size).enumerate().for_each(|(idx, out)| {
        let q_start = idx * head_size;
        let k_start = idx * head_size;
        let v_start = idx * head_size;
        
        // Block size for memory efficiency
        let block_size = 64.min(seq_len);
        
        for i in 0..seq_len {
            let q_row = &query[q_start + i * head_dim..q_start + (i + 1) * head_dim];
            
            // Compute attention for this query position
            let mut max_score = f32::NEG_INFINITY;
            let mut scores = vec![0.0f32; seq_len];
            
            // Compute scores
            for j in 0..=i { // Causal mask
                let k_row = &key[k_start + j * head_dim..k_start + (j + 1) * head_dim];
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_row[d] * k_row[d];
                }
                scores[j] = dot * scale;
                max_score = max_score.max(scores[j]);
            }
            
            // Softmax
            let mut sum = 0.0f32;
            for j in 0..=i {
                scores[j] = (scores[j] - max_score).exp();
                sum += scores[j];
            }
            for j in 0..=i {
                scores[j] /= sum;
            }
            
            // Compute output
            let out_row = &mut out[i * head_dim..(i + 1) * head_dim];
            for j in 0..=i {
                let v_row = &value[v_start + j * head_dim..v_start + (j + 1) * head_dim];
                let w = scores[j];
                for d in 0..head_dim {
                    out_row[d] += w * v_row[d];
                }
            }
        }
    });
    
    output
}

/// Rotary Position Embedding (RoPE)
pub fn apply_rope(
    x: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    base: f32,
) {
    let half_dim = head_dim / 2;
    
    for pos in 0..seq_len {
        for i in 0..half_dim {
            let freq = 1.0 / base.powf(2.0 * i as f32 / head_dim as f32);
            let angle = pos as f32 * freq;
            let cos = angle.cos();
            let sin = angle.sin();
            
            let idx1 = pos * head_dim + i;
            let idx2 = pos * head_dim + i + half_dim;
            
            let x1 = x[idx1];
            let x2 = x[idx2];
            
            x[idx1] = x1 * cos - x2 * sin;
            x[idx2] = x1 * sin + x2 * cos;
        }
    }
}

/// Grouped Query Attention (GQA) support
pub fn grouped_query_attention(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    num_q_heads: usize,
    num_kv_heads: usize,
    seq_len: usize,
    head_dim: usize,
) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let group_size = num_q_heads / num_kv_heads;
    let mut output = vec![0.0f32; num_q_heads * seq_len * head_dim];
    
    // Process query heads in parallel
    output.par_chunks_mut(seq_len * head_dim).enumerate().for_each(|(q_head, out)| {
        let kv_head = q_head / group_size;
        
        let q_start = q_head * seq_len * head_dim;
        let k_start = kv_head * seq_len * head_dim;
        let v_start = kv_head * seq_len * head_dim;
        
        for i in 0..seq_len {
            let mut scores = vec![0.0f32; seq_len];
            let mut max_score = f32::NEG_INFINITY;
            
            // Compute attention scores
            for j in 0..=i {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += query[q_start + i * head_dim + d] * key[k_start + j * head_dim + d];
                }
                scores[j] = dot * scale;
                max_score = max_score.max(scores[j]);
            }
            
            // Softmax
            let mut sum = 0.0f32;
            for j in 0..=i {
                scores[j] = (scores[j] - max_score).exp();
                sum += scores[j];
            }
            
            // Compute weighted sum
            for j in 0..=i {
                let w = scores[j] / sum;
                for d in 0..head_dim {
                    out[i * head_dim + d] += w * value[v_start + j * head_dim + d];
                }
            }
        }
    });
    
    output
}
