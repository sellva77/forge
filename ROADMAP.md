# Forge Roadmap

## Vision

Make JavaScript the #1 language for AI development.

---

## Current Status: v1.0.0 (Alpha)

### ✅ Completed

- [x] Express-style API
- [x] Model presets (tiny → 70b)
- [x] Training loop
- [x] Text generation
- [x] Middleware system
- [x] HTTP server
- [x] RAG pattern
- [x] Agent system
- [x] CLI (new, train, serve, generate)
- [x] Rust core (Tensor, Tokenizer, Model)
- [x] Flash Attention & RoPE
- [x] Documentation

---

## v1.1.0 - Performance (Q1 2025)

### GPU Support

- [ ] CUDA kernels for NVIDIA GPUs
- [ ] Metal shaders for Apple Silicon
- [ ] GPU memory management
- [ ] Multi-GPU support

### Optimizations

- [x] Flash Attention
- [x] KV Cache for generation
- [ ] Fused operations
- [ ] Quantization (INT8, INT4)

---

## v1.2.0 - Models (Q2 2025)

### Model Formats

- [ ] GGUF import/export
- [ ] ONNX export
- [ ] SafeTensors support

### Pre-trained Models

- [ ] Model hub / registry
- [ ] Download pre-trained weights
- [ ] Fine-tuning support

### Tokenizers

- [ ] BPE tokenizer
- [ ] SentencePiece integration
- [ ] HuggingFace tokenizers support

---

## v1.3.0 - Advanced Features (Q3 2025)

### Training

- [ ] Distributed training (DDP)
- [ ] FSDP (Fully Sharded Data Parallel)
- [ ] Gradient checkpointing
- [ ] Mixed precision (FP16, BF16)

### Generation

- [ ] Beam search
- [ ] Speculative decoding
- [ ] Constrained generation

### Patterns

- [ ] Chain-of-Thought
- [ ] Function calling (OpenAI style)
- [ ] Multi-modal (vision)

---

## v2.0.0 - Production (Q4 2025)

### Deployment

- [ ] Docker images
- [ ] Kubernetes operators
- [ ] Serverless support
- [ ] Edge deployment

### Observability

- [ ] Metrics (Prometheus)
- [ ] Tracing
- [ ] Logging

### Security

- [ ] Input validation
- [ ] Rate limiting
- [ ] Authentication

---

## v3.0.0 - Ecosystem (2026)

### Browser Support

- [ ] WebGPU backend
- [ ] WASM build
- [ ] React/Vue/Svelte bindings

### Mobile

- [ ] React Native support
- [ ] iOS/Android native

### Cloud

- [ ] Forge Cloud service
- [ ] Managed training
- [ ] Model marketplace

---

## Community

- [ ] Discord server
- [ ] Documentation site
- [ ] Tutorials / courses
- [ ] Contributor program

---

## How to Contribute

See [CONTRIBUTING.md](./CONTRIBUTING.md) for how to help!

Priority areas:
1. GPU kernels (CUDA/Metal)
2. Tokenizer implementations
3. Model format support
4. Testing and benchmarks

---

## Metrics Goals

| Metric | Target |
|--------|--------|
| npm downloads/week | 10,000+ |
| GitHub stars | 5,000+ |
| Active contributors | 50+ |
| Companies using | 100+ |

---

Let's build the future of AI together! 🔥
