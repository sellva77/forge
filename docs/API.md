# Forge Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [API Reference](#api-reference)
5. [Architecture](#architecture)
6. [Examples](#examples)
7. [CLI Reference](#cli-reference)
8. [Contributing](#contributing)

---

## Introduction

**Forge** is an AI framework for JavaScript with an Express-style API.

### Why Forge?

| Problem | Solution |
|---------|----------|
| Python is required for AI | Forge is pure JavaScript |
| Complex setup | `npm install forge` |
| Verbose APIs | Express-style chaining |
| Slow | Rust core + GPU acceleration |

### Goals

1. **Simplicity** - Express-style API
2. **Performance** - Rust core with N-API bindings
3. **Completeness** - Training, inference, RAG, agents

---

## Installation

```bash
npm install forge
```

### Requirements

- Node.js >= 18.0.0
- (Optional) CUDA for GPU acceleration
- (Optional) Rust for building from source

---

## Quick Start

### Basic Usage

```javascript
const { forge } = require("forge");

// Create app
const app = forge();

// Generate text
const output = await app.generate("Hello, world!");
console.log(output);
```

### With Training

```javascript
const { forge } = require("forge");

const app = forge({ model: "small" });

// Training data
const data = [
  "Forge is an AI framework",
  "It uses an Express-style API",
  "Building AI is now simple",
];

// Train
await app.train(data, { epochs: 10 });

// Generate
const output = await app.generate("Forge is");
console.log(output);
```

### With Server

```javascript
const { forge, logger } = require("forge");

const app = forge();

app.use(logger);

app.post("/chat", async (req, res) => {
  const output = await app.generate(req.body.message);
  res.json({ output });
});

app.listen(3000);
```

---

## API Reference

### `forge(config?)`

Create a Forge application.

```javascript
// Default (small model)
const app = forge();

// With model preset
const app = forge("7b");

// With config object
const app = forge({
  model: "7b",
  modelConfig: {
    dim: 4096,
    layers: 32,
    heads: 32,
  }
});
```

#### Model Presets

| Preset | dim | layers | heads | Params |
|--------|-----|--------|-------|--------|
| `tiny` | 128 | 4 | 4 | ~17M |
| `small` | 256 | 6 | 8 | ~50M |
| `medium` | 512 | 12 | 8 | ~150M |
| `large` | 768 | 24 | 12 | ~450M |
| `7b` | 4096 | 32 | 32 | ~7B |
| `13b` | 5120 | 40 | 40 | ~13B |
| `70b` | 8192 | 80 | 64 | ~70B |

---

### `app.use(middleware)`

Add middleware to the processing pipeline.

```javascript
const { logger, timer, normalize } = require("forge");

app.use(logger);    // Log input/output
app.use(timer);     // Track timing
app.use(normalize); // Clean input

// Custom middleware
app.use(async (ctx, next) => {
  console.log("Before:", ctx.input);
  await next();
  console.log("After:", ctx.output);
});
```

#### Built-in Middleware

| Middleware | Description |
|------------|-------------|
| `logger` | Logs input and output |
| `timer` | Adds `ctx.duration` |
| `normalize` | Trims whitespace |
| `lowercase` | Converts to lowercase |
| `cache(size)` | Caches responses |
| `rateLimit(n, ms)` | Rate limiting |

---

### `app.train(data, config?)`

Train the model.

```javascript
await app.train(data, {
  epochs: 10,      // Number of epochs
  lr: 0.001,       // Learning rate
  batchSize: 8,    // Batch size
  warmupSteps: 100 // LR warmup
});
```

#### Events

```javascript
app.on("step", ({ step, loss }) => {
  console.log(`Step ${step}: loss=${loss}`);
});

app.on("epoch", ({ epoch, avgLoss }) => {
  console.log(`Epoch ${epoch}: avg_loss=${avgLoss}`);
});
```

---

### `app.generate(prompt, config?)`

Generate text.

```javascript
const output = await app.generate("Hello", {
  maxTokens: 100,    // Max tokens to generate
  temperature: 0.7,  // Sampling temperature
  topK: 50,          // Top-K sampling
  topP: 0.9,         // Top-P (nucleus) sampling
  stop: ["\n"]       // Stop sequences
});
```

#### Streaming

```javascript
for await (const token of app.stream("Hello")) {
  process.stdout.write(token);
}
```

---

### `app.rag()`

Create a RAG (Retrieval Augmented Generation) instance.

```javascript
const rag = app.rag();

// Add documents
rag.add("Forge is an AI framework");
rag.add(["Doc 1", "Doc 2", "Doc 3"]);

// Query
const answer = await rag.query("What is Forge?");
```

---

### `app.agent(config?)`

Create an agent with tools.

```javascript
const { createTool, calculatorTool } = require("forge");

const agent = app.agent();

// Add built-in tool
agent.tool(calculatorTool);

// Add custom tool
agent.tool(createTool(
  "weather",
  "Get weather for a location",
  (query) => "Sunny, 25°C"
));

// Run
await agent.run("Calculate 2 + 2");
await agent.run("What's the weather?");
```

---

### Server Methods

```javascript
// Routes
app.get("/path", handler);
app.post("/path", handler);
app.put("/path", handler);
app.delete("/path", handler);

// Handler signature
async (req, res) => {
  // req.body, req.query, req.params
  res.json({ data });
  res.status(404).send("Not found");
}

// Start server
app.listen(3000, () => {
  console.log("Server running");
});
```

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  User JavaScript                     │
│                                                     │
│  const app = forge("7b");                           │
│  await app.train(data);                             │
│  app.generate("Hello");                             │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│                  forge (npm)                         │
│                                                     │
│  TypeScript API                                     │
│  ├── Model management                               │
│  ├── Training orchestration                         │
│  ├── Inference pipeline                             │
│  ├── Middleware system                              │
│  ├── HTTP server                                    │
│  ├── RAG pattern                                    │
│  └── Agent system                                   │
└─────────────────────────────────────────────────────┘
                        │
                        ▼ N-API bindings
┌─────────────────────────────────────────────────────┐
│              @forge-ai/core (Rust)                   │
│                                                     │
│  High-performance operations                        │
│  ├── Tensor operations (matmul, softmax, etc.)     │
│  ├── Tokenization                                   │
│  ├── Attention mechanism                            │
│  ├── Optimizer (AdamW)                              │
│  └── Model forward/backward                         │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│                   GPU Backend                        │
│                                                     │
│  ├── CUDA (NVIDIA)                                  │
│  ├── Metal (Apple)                                  │
│  ├── ROCm (AMD)                                     │
│  └── WebGPU (Browser)                               │
└─────────────────────────────────────────────────────┘
```

---

## Examples

See `examples/` directory:

- `quickstart.js` - Basic usage
- `training.js` - Training with events
- `rag.js` - RAG pattern
- `agent.js` - Agent with tools
- `server.js` - HTTP server

---

## CLI Reference

### `forge new <name>`

Create a new project.

```bash
forge new my-app
forge new my-app --model 7b
```

### `forge train <file>`

Train from a JSON file.

```bash
forge train data.json
forge train data.json --epochs 20 --model medium
```

### `forge serve`

Start API server.

```bash
forge serve
forge serve --port 8080 --model 7b
```

### `forge generate <prompt>`

Generate text.

```bash
forge generate "Hello world"
forge generate "Once upon" --tokens 100
```

---

## Contributing

See CONTRIBUTING.md for guidelines.

---

## License

MIT
