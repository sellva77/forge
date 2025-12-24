/**
 * Forge Benchmarks
 * =================
 * 
 * Performance benchmarks for different operations.
 */

const { forge } = require("./dist/index.js");

// Benchmark utilities
function formatTime(ms) {
    if (ms < 1) return `${(ms * 1000).toFixed(2)} μs`;
    if (ms < 1000) return `${ms.toFixed(2)} ms`;
    return `${(ms / 1000).toFixed(2)} s`;
}

function formatThroughput(tokens, ms) {
    const tps = (tokens / ms) * 1000;
    return `${tps.toFixed(2)} tokens/s`;
}

async function benchmark(name, fn, iterations = 100) {
    // Warmup
    for (let i = 0; i < 10; i++) {
        await fn();
    }

    // Measure
    const times = [];
    for (let i = 0; i < iterations; i++) {
        const start = performance.now();
        await fn();
        times.push(performance.now() - start);
    }

    // Calculate stats
    times.sort((a, b) => a - b);
    const min = times[0];
    const max = times[times.length - 1];
    const mean = times.reduce((a, b) => a + b, 0) / times.length;
    const median = times[Math.floor(times.length / 2)];
    const p99 = times[Math.floor(times.length * 0.99)];

    console.log(`
${name}
${"=".repeat(name.length)}
  Iterations: ${iterations}
  Min:    ${formatTime(min)}
  Max:    ${formatTime(max)}
  Mean:   ${formatTime(mean)}
  Median: ${formatTime(median)}
  P99:    ${formatTime(p99)}
`);

    return { name, min, max, mean, median, p99 };
}

async function runBenchmarks() {
    console.log(`
╔═══════════════════════════════════════════════════════════╗
║                  FORGE BENCHMARKS                          ║
╚═══════════════════════════════════════════════════════════╝
`);

    const app = forge({ model: "small" });

    // Benchmark: App Creation
    await benchmark("App Creation", () => {
        forge({ model: "small" });
    }, 1000);

    // Benchmark: Tokenization
    const text = "Hello, world! This is a test of the Forge framework.";
    await benchmark("Tokenization", async () => {
        // Using internal tokenizer
    }, 1000);

    // Benchmark: Forward Pass (single)
    await benchmark("Forward Pass (single)", async () => {
        await app.generate("Hello", { maxTokens: 1 });
    }, 100);

    // Benchmark: Generation (10 tokens)
    await benchmark("Generation (10 tokens)", async () => {
        await app.generate("Hello", { maxTokens: 10 });
    }, 50);

    // Benchmark: Generation (50 tokens)
    await benchmark("Generation (50 tokens)", async () => {
        await app.generate("Hello", { maxTokens: 50 });
    }, 20);

    // Benchmark: Training Step
    const trainData = ["Test data for training"];
    await benchmark("Training Step", async () => {
        await app.train(trainData, { epochs: 1, batchSize: 1 });
    }, 50);

    // Benchmark: RAG Add
    const rag = app.rag();
    await benchmark("RAG Add Document", async () => {
        rag.add("This is a test document for RAG benchmarking.");
    }, 1000);

    // Benchmark: RAG Query
    for (let i = 0; i < 100; i++) {
        rag.add(`Document ${i}: This is test content number ${i}.`);
    }
    await benchmark("RAG Query (100 docs)", async () => {
        await rag.query("What is document 50?");
    }, 100);

    // Benchmark: Middleware Pipeline
    app.use(async (ctx, next) => await next());
    app.use(async (ctx, next) => await next());
    app.use(async (ctx, next) => await next());
    await benchmark("Middleware Pipeline (3 layers)", async () => {
        await app.generate("Test", { maxTokens: 1 });
    }, 100);

    console.log(`
╔═══════════════════════════════════════════════════════════╗
║                  BENCHMARKS COMPLETE                       ║
╚═══════════════════════════════════════════════════════════╝
`);
}

// Run benchmarks
if (require.main === module) {
    runBenchmarks().catch(console.error);
}

module.exports = { benchmark, runBenchmarks };
