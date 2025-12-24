# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-12-24

### Added

- 🔥 **Initial Release** - Forge AI Framework for JavaScript
- ✨ **Express-style API** - Familiar patterns for JS developers
- 🧠 **Transformer Implementation** - Full transformer model with attention, embeddings, and generation
- 📚 **Training System** - Train models on custom text data
- 💬 **RAG (Retrieval Augmented Generation)** - Built-in knowledge retrieval
- 🤖 **AI Agents** - Tool-calling agents with built-in tools
- 🔌 **Middleware System** - Logger, timer, cache, and custom middleware
- 🌐 **HTTP Server** - Built-in API server
- ⚡ **High Performance** - Rust core with N-API bindings
- 📦 **CLI** - Command-line interface for project scaffolding and generation
- 🎯 **Model Presets** - tiny, small, medium, large, 7b, 13b configurations

### Model Presets

| Preset | Parameters | Layers | Heads | Dimension |
|--------|------------|--------|-------|-----------|
| tiny   | ~17M       | 4      | 4     | 128       |
| small  | ~50M       | 6      | 8     | 256       |
| medium | ~150M      | 12     | 12    | 512       |
| large  | ~450M      | 24     | 16    | 768       |
| 7b     | ~7B        | 32     | 32    | 4096      |
| 13b    | ~13B       | 40     | 40    | 5120      |

### Performance Backends

- **CUDA** - NVIDIA GPU acceleration (when available)
- **Native** - Rust binaries via N-API
- **TypeScript** - Pure JavaScript fallback

### CLI Commands

- `forge new <name>` - Create new project
- `forge generate <prompt>` - Generate text
- `forge train <file>` - Train from data
- `forge serve` - Start API server
- `forge info` - Show system info
- `forge benchmark` - Run benchmarks

### Core Features

- Tensor operations (create, matmul, add, softmax, etc.)
- BPE Tokenization
- Rotary Position Embeddings (RoPE)
- KV Caching for efficient generation
- Multi-head self-attention
- RMSNorm normalization

---

## [Unreleased]

### Planned Features

- [ ] WebGPU backend
- [ ] Metal backend for Apple Silicon
- [ ] Model serialization/loading
- [ ] Fine-tuning support
- [ ] Quantization (INT8, INT4)
- [ ] Multi-GPU support
- [ ] ONNX export
- [ ] Vision transformer support
