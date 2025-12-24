# Forge Architecture

## Overview

Forge is a layered architecture designed for performance and simplicity.

```
User Code (JavaScript)
         │
         ▼
    forge (npm)          ← TypeScript API layer
         │
         ▼ N-API
  @forge-ai/core         ← Rust performance layer
         │
         ▼
    GPU Backend          ← CUDA / Metal / WebGPU
```

---

## Layer 1: JavaScript API (`forge`)

The user-facing layer that provides an Express-style API.

### Components

| File | Purpose |
|------|---------|
| `index.ts` | Main entry, Forge class |
| `model.ts` | Model presets and management |
| `tokenizer.ts` | Text tokenization |
| `trainer.ts` | Training orchestration |
| `generator.ts` | Text generation |
| `middleware.ts` | Middleware system |
| `server.ts` | HTTP server |
| `rag.ts` | RAG pattern |
| `agent.ts` | Agent system |

### Design Principles

1. **Chainable API** - All methods return `this` for chaining
2. **Event-driven** - Uses EventEmitter for progress updates
3. **Middleware pattern** - Like Express.js
4. **Async by default** - All operations are async

---

## Layer 2: Rust Core (`@forge-ai/core`)

High-performance operations exposed via N-API.

### Components

| File | Purpose |
|------|---------|
| `lib.rs` | Main entry, N-API exports |
| `attention.rs` | Multi-head attention |
| `optimizer.rs` | AdamW, LR schedulers |

### Tensor Operations

```rust
// From lib.rs
impl Tensor {
    fn matmul(&self, other: &Tensor) -> Tensor;
    fn softmax(&self) -> Tensor;
    fn gelu(&self) -> Tensor;
    fn layer_norm(&self, eps: f32) -> Tensor;
}
```

### N-API Bindings

```rust
use napi_derive::napi;

#[napi]
impl Model {
    #[napi]
    pub fn forward(&self, input: Vec<u32>) -> Vec<f64>;
    
    #[napi]
    pub fn generate(&self, tokens: Vec<u32>, max_tokens: u32) -> Vec<u32>;
}
```

---

## Layer 3: GPU Backend

Hardware acceleration for compute-intensive operations.

### CUDA (NVIDIA)

```
Kernel functions:
├── attention_forward
├── matmul_tiled
├── softmax
├── gelu_forward
└── layer_norm_forward
```

### Metal (Apple)

```
Shaders:
├── attention.metal
├── matmul.metal
└── activation.metal
```

### WebGPU (Browser)

```
WGSL shaders for browser deployment
```

---

## Data Flow

### Training

```
1. User provides: data, config
         │
         ▼
2. JavaScript: Tokenize text
         │
         ▼ (tokens)
3. Rust: Forward pass → loss
         │
         ▼
4. Rust: Backward pass → gradients
         │
         ▼
5. Rust: Optimizer step → update weights
         │
         ▼
6. JavaScript: Emit progress events
```

### Inference

```
1. User provides: prompt, config
         │
         ▼
2. JavaScript: Tokenize prompt
         │
         ▼ (tokens)
3. Rust: Forward pass → logits
         │
         ▼
4. Rust: Sample next token
         │
         ▼
5. Repeat until done
         │
         ▼
6. JavaScript: Decode tokens → text
```

---

## Middleware Pipeline

```
Input
  │
  ▼
┌─────────────────┐
│  Middleware 1   │  (e.g., logger)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Middleware 2   │  (e.g., timer)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Core Handler   │  (generation)
└────────┬────────┘
         │
         ▼
Output
```

---

## Performance Optimizations

### Memory

1. **Weight sharing** - Reuse embeddings
2. **Gradient checkpointing** - Trade compute for memory
3. **KV cache** - Cache key/value for generation

### Compute

1. **Flash Attention** - O(N) memory attention
2. **Fused kernels** - Combine operations
3. **Quantization** - INT8/INT4 inference

### I/O

1. **Streaming** - Token-by-token output
2. **Batching** - Process multiple requests
3. **Async** - Non-blocking operations

---

## File Structure

```
forge/
├── packages/
│   ├── forge/              # JS package
│   │   ├── src/
│   │   │   ├── index.ts
│   │   │   ├── model.ts
│   │   │   ├── tokenizer.ts
│   │   │   ├── trainer.ts
│   │   │   ├── generator.ts
│   │   │   ├── middleware.ts
│   │   │   ├── server.ts
│   │   │   ├── rag.ts
│   │   │   └── agent.ts
│   │   ├── bin/
│   │   │   └── cli.js
│   │   ├── package.json
│   │   └── tsconfig.json
│   │
│   └── core/               # Rust package
│       ├── src/
│       │   ├── lib.rs
│       │   ├── attention.rs
│       │   └── optimizer.rs
│       ├── Cargo.toml
│       └── build.rs
│
├── examples/
│   ├── quickstart.js
│   ├── training.js
│   ├── rag.js
│   ├── agent.js
│   └── server.js
│
├── docs/
│   ├── API.md
│   └── ARCHITECTURE.md
│
├── README.md
└── package.json
```

---

## Comparison with PyTorch

| Aspect | Forge | PyTorch |
|--------|-------|---------|
| Language | JavaScript | Python |
| Core | Rust + N-API | C++ + Python C API |
| GPU | CUDA/Metal/WebGPU | CUDA/ROCm |
| API Style | Express-like | Object-oriented |
| Package Manager | npm | pip/conda |
| Browser Support | Yes (WebGPU) | No |
