//! Optimizer Module
//! =================
//!
//! High-performance optimizers for training.

use rayon::prelude::*;

/// Adam optimizer state
pub struct AdamState {
    pub m: Vec<f32>,  // First moment
    pub v: Vec<f32>,  // Second moment
    pub t: usize,     // Time step
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
}

impl AdamState {
    pub fn new(size: usize, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) -> Self {
        AdamState {
            m: vec![0.0; size],
            v: vec![0.0; size],
            t: 0,
            beta1,
            beta2,
            eps,
            weight_decay,
        }
    }

    /// Perform Adam update step
    pub fn step(&mut self, params: &mut [f32], grads: &[f32], lr: f32) {
        self.t += 1;
        let t = self.t as f32;
        
        let bias_correction1 = 1.0 - self.beta1.powf(t);
        let bias_correction2 = 1.0 - self.beta2.powf(t);
        let step_size = lr / bias_correction1;
        
        // Parallel update
        params.par_iter_mut()
            .zip(grads.par_iter())
            .zip(self.m.par_iter_mut())
            .zip(self.v.par_iter_mut())
            .for_each(|(((param, &grad), m), v)| {
                // Weight decay
                let grad = grad + self.weight_decay * *param;
                
                // Update biased first moment estimate
                *m = self.beta1 * *m + (1.0 - self.beta1) * grad;
                
                // Update biased second moment estimate
                *v = self.beta2 * *v + (1.0 - self.beta2) * grad * grad;
                
                // Compute bias-corrected second moment
                let v_hat = *v / bias_correction2;
                
                // Update parameters
                *param -= step_size * *m / (v_hat.sqrt() + self.eps);
            });
    }
}

/// AdamW optimizer (decoupled weight decay)
pub struct AdamWState {
    pub m: Vec<f32>,
    pub v: Vec<f32>,
    pub t: usize,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
}

impl AdamWState {
    pub fn new(size: usize, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) -> Self {
        AdamWState {
            m: vec![0.0; size],
            v: vec![0.0; size],
            t: 0,
            beta1,
            beta2,
            eps,
            weight_decay,
        }
    }

    pub fn step(&mut self, params: &mut [f32], grads: &[f32], lr: f32) {
        self.t += 1;
        let t = self.t as f32;
        
        let bias_correction1 = 1.0 - self.beta1.powf(t);
        let bias_correction2 = 1.0 - self.beta2.powf(t);
        
        params.par_iter_mut()
            .zip(grads.par_iter())
            .zip(self.m.par_iter_mut())
            .zip(self.v.par_iter_mut())
            .for_each(|(((param, &grad), m), v)| {
                // Decoupled weight decay (applied directly to params)
                *param *= 1.0 - lr * self.weight_decay;
                
                // Update moments
                *m = self.beta1 * *m + (1.0 - self.beta1) * grad;
                *v = self.beta2 * *v + (1.0 - self.beta2) * grad * grad;
                
                // Bias correction
                let m_hat = *m / bias_correction1;
                let v_hat = *v / bias_correction2;
                
                // Update
                *param -= lr * m_hat / (v_hat.sqrt() + self.eps);
            });
    }
}

/// SGD with momentum
pub struct SGDState {
    pub velocity: Vec<f32>,
    pub momentum: f32,
    pub weight_decay: f32,
    pub nesterov: bool,
}

impl SGDState {
    pub fn new(size: usize, momentum: f32, weight_decay: f32, nesterov: bool) -> Self {
        SGDState {
            velocity: vec![0.0; size],
            momentum,
            weight_decay,
            nesterov,
        }
    }

    pub fn step(&mut self, params: &mut [f32], grads: &[f32], lr: f32) {
        params.par_iter_mut()
            .zip(grads.par_iter())
            .zip(self.velocity.par_iter_mut())
            .for_each(|((param, &grad), v)| {
                let grad = grad + self.weight_decay * *param;
                
                *v = self.momentum * *v + grad;
                
                if self.nesterov {
                    *param -= lr * (grad + self.momentum * *v);
                } else {
                    *param -= lr * *v;
                }
            });
    }
}

/// Lion optimizer (evolved sign momentum)
pub struct LionState {
    pub m: Vec<f32>,
    pub beta1: f32,
    pub beta2: f32,
    pub weight_decay: f32,
}

impl LionState {
    pub fn new(size: usize, beta1: f32, beta2: f32, weight_decay: f32) -> Self {
        LionState {
            m: vec![0.0; size],
            beta1,
            beta2,
            weight_decay,
        }
    }

    pub fn step(&mut self, params: &mut [f32], grads: &[f32], lr: f32) {
        params.par_iter_mut()
            .zip(grads.par_iter())
            .zip(self.m.par_iter_mut())
            .for_each(|((param, &grad), m)| {
                // Weight decay
                *param *= 1.0 - lr * self.weight_decay;
                
                // Compute update direction
                let update = self.beta1 * *m + (1.0 - self.beta1) * grad;
                
                // Sign update (main Lion innovation)
                *param -= lr * update.signum();
                
                // Update momentum
                *m = self.beta2 * *m + (1.0 - self.beta2) * grad;
            });
    }
}

/// Learning rate schedulers
pub mod schedulers {
    /// Constant learning rate
    pub fn constant(base_lr: f32, _step: usize) -> f32 {
        base_lr
    }

    /// Linear warmup then constant
    pub fn warmup_constant(base_lr: f32, step: usize, warmup_steps: usize) -> f32 {
        if step < warmup_steps {
            base_lr * (step as f32 / warmup_steps as f32)
        } else {
            base_lr
        }
    }

    /// Cosine decay
    pub fn cosine_decay(base_lr: f32, step: usize, total_steps: usize, min_lr: f32) -> f32 {
        let progress = (step as f32 / total_steps as f32).min(1.0);
        let decay = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
        min_lr + (base_lr - min_lr) * decay
    }

    /// Linear warmup + cosine decay
    pub fn warmup_cosine(
        base_lr: f32,
        step: usize,
        warmup_steps: usize,
        total_steps: usize,
        min_lr: f32,
    ) -> f32 {
        if step < warmup_steps {
            base_lr * (step as f32 / warmup_steps as f32)
        } else {
            let progress = ((step - warmup_steps) as f32 / (total_steps - warmup_steps) as f32).min(1.0);
            let decay = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
            min_lr + (base_lr - min_lr) * decay
        }
    }

    /// Exponential decay
    pub fn exponential_decay(base_lr: f32, step: usize, decay_rate: f32, decay_steps: usize) -> f32 {
        base_lr * decay_rate.powf((step / decay_steps) as f32)
    }

    /// Step decay
    pub fn step_decay(base_lr: f32, step: usize, step_size: usize, gamma: f32) -> f32 {
        base_lr * gamma.powf((step / step_size) as f32)
    }
}

/// Gradient clipping
pub fn clip_grad_norm(grads: &mut [f32], max_norm: f32) -> f32 {
    let total_norm: f32 = grads.par_iter().map(|&g| g * g).sum::<f32>().sqrt();
    
    if total_norm > max_norm {
        let scale = max_norm / (total_norm + 1e-6);
        grads.par_iter_mut().for_each(|g| *g *= scale);
    }
    
    total_norm
}

/// Gradient clipping by value
pub fn clip_grad_value(grads: &mut [f32], clip_value: f32) {
    grads.par_iter_mut().for_each(|g| {
        *g = g.clamp(-clip_value, clip_value);
    });
}
