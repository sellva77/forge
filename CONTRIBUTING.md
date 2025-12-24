# Contributing to Forge

Thank you for your interest in contributing to Forge!

---

## Development Setup

### Prerequisites

- Node.js >= 18.0.0
- Rust >= 1.70.0
- (Optional) CUDA Toolkit for GPU support

### Clone and Install

```bash
git clone https://github.com/forge-ai/forge.git
cd forge
npm install
```

### Build TypeScript

```bash
cd packages/forge
npm run build
```

### Build Rust Core

```bash
cd packages/core
npm run build
```

---

## Project Structure

```
forge/
├── packages/
│   ├── forge/          # TypeScript package
│   └── core/           # Rust package
├── examples/           # Example scripts
├── docs/               # Documentation
└── tests/              # Test files
```

---

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/my-feature
```

### 2. Make Changes

- TypeScript: `packages/forge/src/`
- Rust: `packages/core/src/`

### 3. Test

```bash
npm test
```

### 4. Commit

```bash
git commit -m "feat: add my feature"
```

Follow [Conventional Commits](https://www.conventionalcommits.org/).

### 5. Push and PR

```bash
git push origin feature/my-feature
```

---

## Code Style

### TypeScript

- Use ESLint + Prettier
- Prefer `async/await` over callbacks
- Use TypeScript types everywhere

### Rust

- Use `cargo fmt`
- Use `cargo clippy`
- Document public APIs

---

## Testing

### TypeScript Tests

```bash
cd packages/forge
npm test
```

### Rust Tests

```bash
cd packages/core
cargo test
```

---

## Areas to Contribute

### High Priority

1. **GPU Kernels** - CUDA/Metal implementations
2. **Tokenizers** - BPE, SentencePiece support
3. **Model Formats** - GGUF, ONNX import/export
4. **Benchmarks** - Performance comparisons

### Medium Priority

1. **Documentation** - Improve docs and examples
2. **Testing** - Increase test coverage
3. **CLI** - Add more commands
4. **Middleware** - New built-in middleware

### Good First Issues

Look for issues labeled `good first issue`.

---

## Pull Request Guidelines

1. **One feature per PR**
2. **Include tests**
3. **Update documentation**
4. **Follow code style**
5. **Write clear commit messages**

---

## Code of Conduct

Be respectful and inclusive. See CODE_OF_CONDUCT.md.

---

## Questions?

- Open an issue
- Join our Discord (coming soon)

---

Thank you for contributing! 🔥
