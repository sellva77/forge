<p align="center">
  <h1 align="center">🔥 Forge AI</h1>
</p>

<p align="center">
  <strong>AI Framework for JavaScript</strong><br>
  Express-style API for building AI applications with transformers, training, RAG, and agents.
</p>

<p align="center">
  <a href="https://www.npmjs.com/package/forge-ai"><img src="https://img.shields.io/npm/v/forge-ai.svg?style=flat-square&color=ff6b6b" alt="npm version"></a>
  <a href="https://www.npmjs.com/package/forge-ai"><img src="https://img.shields.io/npm/dm/forge-ai.svg?style=flat-square&color=4ecdc4" alt="npm downloads"></a>
  <a href="https://github.com/forge-ai/forge/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square" alt="license"></a>
  <a href="https://github.com/forge-ai/forge"><img src="https://img.shields.io/github/stars/forge-ai/forge?style=flat-square&color=ffe66d" alt="GitHub stars"></a>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-features">Features</a> •
  <a href="#-examples">Examples</a> •
  <a href="#-cli">CLI</a> •
  <a href="#-performance">Performance</a> •
  <a href="#-contributing">Contributing</a>
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
| ⚡ **High Performance** | Optional Rust core with GPU acceleration |
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

app.listen(3000);
```

### RAG

```javascript
const rag = app.rag();

rag.add("Forge was created in 2024");
rag.add("Forge is an AI framework for JavaScript");

const answer = await rag.query("When was Forge created?");
// "Forge was created in 2024"
```

### Agents

```javascript
const { forge, createTool, calculatorTool } = require("forge-ai");

const agent = app.agent();

agent.tool(calculatorTool);
agent.tool(createTool("weather", "Get weather", () => "Sunny, 25°C"));

await agent.run("What's 10 * 5?");
```

---

## 🖥️ CLI

```bash
# Install globally
npm install -g forge-ai

# Create a new project
forge new my-ai-app

# Generate text
forge generate "Hello world" --model small

# Start a server
forge serve --port 3000

# Train a model
forge train data.json --epochs 20

# System info
forge info
```

### Project Templates

```bash
forge new my-app                    # Basic
forge new my-app --template server  # API server
forge new my-app --template rag     # RAG application
forge new my-app --template agent   # AI agent
forge new my-app --typescript       # TypeScript
```

---

## 🏗️ Model Presets

```javascript
const app = forge({ model: "7b" });
```

| Preset | Parameters | Use Case |
|--------|------------|----------|
| `tiny` | ~17M | Testing, prototyping |
| `small` | ~50M | Development |
| `medium` | ~150M | Production |
| `large` | ~450M | Complex tasks |
| `7b` | ~7B | High-quality |
| `13b` | ~13B | Professional |

---

## ⚡ Performance

Forge automatically selects the fastest available backend:

| Backend | Speed | When Used |
|---------|-------|-----------|
| **CUDA** | ⚡⚡⚡⚡⚡ | NVIDIA GPU |
| **Native** | ⚡⚡⚡⚡ | Rust binaries |
| **TypeScript** | ⚡⚡ | Always works |

For maximum performance, install the optional Rust core:

```bash
npm install @forge-ai/core
```

---

## 📁 Repository Structure

```
forge/
├── packages/
│   ├── forge/          # Main TypeScript framework (forge-ai)
│   └── core/           # Rust native bindings (@forge-ai/core)
├── examples/           # Usage examples
├── docs/               # Documentation
└── first model/        # Starter project
```

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

## 🤝 Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

```bash
# Build TypeScript
npm run build:forge

# Build Rust core (requires Rust)
npm run build:core
```

---

## 📄 License

MIT © [Forge AI Team](https://github.com/forge-ai)

---

<p align="center">
  <strong>Built with ❤️ for the JavaScript community</strong>
</p>
