<p align="center">
  <img src="https://raw.githubusercontent.com/forge-ai/forge/main/docs/logo.svg" width="200" alt="Forge AI Logo">
</p>

<h1 align="center">🔥 Forge AI</h1>

<p align="center">
  <strong>AI Framework for JavaScript</strong><br>
  Express-style API for building AI applications.
</p>

<p align="center">
  <a href="https://www.npmjs.com/package/forge-ai"><img src="https://img.shields.io/npm/v/forge-ai.svg?style=flat-square" alt="npm version"></a>
  <a href="https://www.npmjs.com/package/forge-ai"><img src="https://img.shields.io/npm/dm/forge-ai.svg?style=flat-square" alt="npm downloads"></a>
  <a href="https://github.com/forge-ai/forge/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square" alt="license"></a>
  <a href="https://github.com/forge-ai/forge"><img src="https://img.shields.io/github/stars/forge-ai/forge?style=flat-square" alt="GitHub stars"></a>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-features">Features</a> •
  <a href="#-examples">Examples</a> •
  <a href="#-cli">CLI</a> •
  <a href="#-api">API</a> •
  <a href="#-performance">Performance</a>
</p>

---

## ⚡ Quick Start

```bash
npm install forge-ai
```

```javascript
const { forge } = require("forge-ai");

// Create app
const app = forge();

// Generate text
const output = await app.generate("Hello, world!");
console.log(output);
```

**That's it. No Python. No build steps. Just JavaScript.**

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🚀 **Express-style API** | Familiar patterns for JS developers |
| 🧠 **Built-in Transformer** | Full transformer model implementation |
| 📚 **Training** | Train on your own data with one line |
| 💬 **RAG** | Retrieval Augmented Generation built-in |
| 🤖 **Agents** | AI agents with tool calling |
| 🔌 **Middleware** | Logger, timer, cache, and custom middleware |
| 🌐 **HTTP Server** | Built-in API server |
| ⚡ **High Performance** | Rust core with GPU acceleration |
| 📦 **Zero Config** | Works out of the box |

---

## 📚 Examples

### Basic Generation

```javascript
const { forge } = require("forge-ai");

const app = forge({ model: "small" });

const output = await app.generate("Once upon a time", {
    maxTokens: 100,
    temperature: 0.8,
});

console.log(output);
```

### Streaming

```javascript
const app = forge();

for await (const token of app.stream("The meaning of life is")) {
    process.stdout.write(token);
}
```

### Training

```javascript
const app = forge({ model: "small" });

await app.train([
    "Forge is an AI framework",
    "It uses Express-style API",
    "Building AI is now simple",
], { epochs: 10 });

const output = await app.generate("Forge is");
console.log(output);
```

### API Server

```javascript
const { forge, logger, timer } = require("forge-ai");

const app = forge();

app.use(logger);
app.use(timer);

app.post("/chat", async (req, res) => {
    const output = await app.generate(req.body.message);
    res.json({ output });
});

app.listen(3000, () => {
    console.log("🔥 Server running on http://localhost:3000");
});
```

### RAG (Retrieval Augmented Generation)

```javascript
const app = forge();
const rag = app.rag();

// Add knowledge
rag.add("Forge was created in 2024");
rag.add("Forge is an AI framework for JavaScript");

// Query
const answer = await rag.query("When was Forge created?");
console.log(answer); // "Forge was created in 2024"
```

### Agents with Tools

```javascript
const { forge, createTool, calculatorTool } = require("forge-ai");

const app = forge();
const agent = app.agent();

agent.tool(calculatorTool);
agent.tool(createTool("weather", "Get weather", () => "Sunny, 25°C"));

await agent.run("What's 10 * 5?");
await agent.run("What's the weather?");
```

### Custom Middleware

```javascript
const app = forge();

app.use(async (ctx, next) => {
    console.log("Input:", ctx.input);
    const start = Date.now();
    
    await next();
    
    console.log("Output:", ctx.output);
    console.log(`Time: ${Date.now() - start}ms`);
});
```

---

## 🖥️ CLI

Forge comes with a powerful CLI for quick prototyping.

```bash
# Install globally
npm install -g forge-ai

# Create a new project
forge new my-ai-app

# Generate text
forge generate "Hello world" --model small

# Start a server
forge serve --port 3000 --model medium

# Train a model
forge train data.json --epochs 20

# Show system info
forge info

# Run benchmarks
forge benchmark --model small
```

### Project Templates

```bash
# Basic project
forge new my-app

# Server template
forge new my-server --template server

# RAG template
forge new my-rag --template rag

# Agent template
forge new my-agent --template agent

# TypeScript project
forge new my-ts-app --typescript
```

---

## 📖 API Reference

### `forge(config?)`

Create a new Forge application.

```javascript
const app = forge();
const app = forge({ model: "7b" });
const app = forge({ model: "small", modelConfig: { temperature: 0.9 } });
```

### Model Presets

| Preset | Parameters | Use Case |
|--------|------------|----------|
| `tiny` | ~17M | Testing, prototyping |
| `small` | ~50M | Development, small tasks |
| `medium` | ~150M | Production, general use |
| `large` | ~450M | Complex tasks |
| `7b` | ~7B | High-quality generation |
| `13b` | ~13B | Professional use |

### Methods

| Method | Description |
|--------|-------------|
| `app.generate(prompt, config?)` | Generate text |
| `app.stream(prompt, config?)` | Stream tokens |
| `app.train(data, config?)` | Train the model |
| `app.use(middleware)` | Add middleware |
| `app.rag()` | Create RAG instance |
| `app.agent(config?)` | Create agent instance |
| `app.listen(port)` | Start HTTP server |

### Built-in Middleware

```javascript
const { logger, timer, cache, normalize, lowercase } = require("forge-ai");

app.use(logger);           // Log input/output
app.use(timer);            // Track timing
app.use(cache(100));       // Cache last 100 responses
app.use(normalize);        // Normalize whitespace
app.use(lowercase);        // Lowercase input
```

---

## ⚡ Performance

Forge automatically selects the fastest available backend:

| Backend | Speed | When Used |
|---------|-------|-----------|
| **CUDA** | ⚡⚡⚡⚡⚡ | NVIDIA GPU available |
| **Native** | ⚡⚡⚡⚡ | Pre-built Rust binary |
| **TypeScript** | ⚡⚡ | Fallback (always works) |

```javascript
// Check what backend is active
const { getBackendInfo } = require("forge-ai");

const info = getBackendInfo();
console.log(info);
// { native: true, cuda: false, platform: "win32-x64-msvc" }
```

### Tensor Operations

```javascript
const { Tensor } = require("forge-ai");

const a = Tensor.randn([100, 100]);
const b = Tensor.randn([100, 100]);
const c = a.matmul(b);

console.log(c.shape); // [100, 100]
```

---

## 📁 Package Structure

```
forge-ai/
├── dist/           # Compiled JavaScript (CJS)
├── dist/esm/       # ES Modules build
├── bin/cli.js      # CLI executable
└── README.md
```

---

## 🔌 Optional Dependencies

For maximum performance, install the native Rust core:

```bash
npm install @forge-ai/core
```

The framework works without it (using TypeScript fallback), but native bindings provide 5-10x speedup.

---

## 🛠️ Development

```bash
# Clone
git clone https://github.com/forge-ai/forge.git
cd forge

# Install
npm install

# Build
npm run build

# Test
npm test

# Run examples
node examples/01_quickstart.js
```

---

## 📄 License

MIT © [Forge AI Team](https://github.com/forge-ai)

---

<p align="center">
  <strong>Built with ❤️ for the JavaScript community</strong>
</p>
