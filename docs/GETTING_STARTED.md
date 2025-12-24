# Getting Started with Forge

Welcome to Forge! This guide will get you up and running in 5 minutes.

---

## Step 1: Install

```bash
npm install forge
```

---

## Step 2: Create Your First App

Create a file called `app.js`:

```javascript
const { forge } = require("forge");

async function main() {
  // Create a Forge app
  const app = forge();

  // Generate text
  const output = await app.generate("Hello, world!");
  
  console.log(output);
}

main();
```

Run it:

```bash
node app.js
```

---

## Step 3: Train a Model

```javascript
const { forge } = require("forge");

async function main() {
  const app = forge({ model: "small" });

  // Your training data
  const data = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning is a subset of artificial intelligence",
    "JavaScript is a versatile programming language",
  ];

  // Train
  await app.train(data, {
    epochs: 5,
    lr: 0.001,
  });

  // Test
  const output = await app.generate("Machine learning");
  console.log(output);
}

main();
```

---

## Step 4: Add Middleware

```javascript
const { forge, logger, timer } = require("forge");

async function main() {
  const app = forge();

  // Add middleware (like Express!)
  app.use(logger);
  app.use(timer);

  const output = await app.generate("Hello");
  console.log(output);
}

main();
```

---

## Step 5: Create an API Server

```javascript
const { forge, logger } = require("forge");

async function main() {
  const app = forge();

  app.use(logger);

  // Define routes
  app.get("/", (req, res) => {
    res.json({ message: "Welcome to Forge API" });
  });

  app.post("/chat", async (req, res) => {
    const output = await app.generate(req.body.message);
    res.json({ output });
  });

  // Start server
  app.listen(3000, () => {
    console.log("Server running on http://localhost:3000");
  });
}

main();
```

Test with curl:

```bash
curl http://localhost:3000/
curl -X POST http://localhost:3000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello"}'
```

---

## Step 6: Use RAG

```javascript
const { forge } = require("forge");

async function main() {
  const app = forge();

  // Create RAG instance
  const rag = app.rag();

  // Add knowledge
  rag.add("Forge was created in 2024");
  rag.add("Forge is an AI framework for JavaScript");
  rag.add("Forge uses Rust for performance");

  // Query
  const answer = await rag.query("When was Forge created?");
  console.log(answer);
}

main();
```

---

## Step 7: Use Agents

```javascript
const { forge, createTool, calculatorTool } = require("forge");

async function main() {
  const app = forge();

  // Create agent
  const agent = app.agent();

  // Add tools
  agent.tool(calculatorTool);
  agent.tool(createTool(
    "greet",
    "Greet someone",
    (name) => `Hello, ${name}!`
  ));

  // Run
  await agent.run("Calculate 10 * 5");
  await agent.run("Greet John");
}

main();
```

---

## Next Steps

1. Read the [API Reference](./API.md)
2. Explore [Examples](../examples/)
3. Understand the [Architecture](./ARCHITECTURE.md)
4. Check out the [CLI](./API.md#cli-reference)

---

## Common Patterns

### Pattern 1: Chat Application

```javascript
const { forge, logger } = require("forge");

const app = forge({ model: "7b" });
app.use(logger);

const history = [];

app.post("/chat", async (req, res) => {
  history.push({ role: "user", content: req.body.message });
  
  const context = history.map(m => `${m.role}: ${m.content}`).join("\n");
  const output = await app.generate(context + "\nassistant:");
  
  history.push({ role: "assistant", content: output });
  res.json({ output });
});

app.listen(3000);
```

### Pattern 2: Document Q&A

```javascript
const { forge } = require("forge");
const fs = require("fs");

const app = forge();
const rag = app.rag();

// Load documents
const docs = fs.readdirSync("./docs")
  .filter(f => f.endsWith(".txt"))
  .map(f => fs.readFileSync(`./docs/${f}`, "utf-8"));

rag.add(docs);

app.post("/ask", async (req, res) => {
  const answer = await rag.query(req.body.question);
  res.json({ answer });
});

app.listen(3000);
```

### Pattern 3: Custom Preprocessing

```javascript
const { forge } = require("forge");

const app = forge();

// Custom middleware
app.use(async (ctx, next) => {
  // Remove special characters
  ctx.input = ctx.input.replace(/[^\w\s]/g, "");
  await next();
});

app.use(async (ctx, next) => {
  // Add system prompt
  ctx.input = `You are a helpful assistant.\n\nUser: ${ctx.input}\n\nAssistant:`;
  await next();
});

const output = await app.generate("Hello!");
```

---

## Troubleshooting

### Issue: Module not found

```bash
npm install forge
```

### Issue: Out of memory

Use a smaller model:

```javascript
const app = forge({ model: "tiny" });
```

### Issue: Slow performance

Enable GPU (if available):

```bash
npm install @forge-ai/core-cuda  # NVIDIA
npm install @forge-ai/core-metal # Apple
```

---

Happy building! 🔥
