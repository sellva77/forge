/**
 * Forge - Test Script
 * ====================
 */

const { forge, logger, timer, createTool } = require("./dist/index.js");

async function test() {
    console.log("═══════════════════════════════════════════════════════");
    console.log("  🔥 Forge Framework - Test Suite");
    console.log("═══════════════════════════════════════════════════════\n");

    // Test 1: Create app
    console.log("1️⃣  Creating app...");
    const app = forge({ model: "small" });
    console.log("   ✓ App created\n");

    // Test 2: Middleware
    console.log("2️⃣  Adding middleware...");
    app.use(logger);
    app.use(timer);
    console.log("   ✓ Middleware added\n");

    // Test 3: Training
    console.log("3️⃣  Training model...");
    const data = [
        "Forge is an AI framework",
        "It has an Express-style API",
        "Building AI is now simple",
    ];
    await app.train(data, { epochs: 2, batchSize: 2 });
    console.log("   ✓ Training complete\n");

    // Test 4: Generation
    console.log("4️⃣  Generating text...");
    const output = await app.generate("Hello", { maxTokens: 10 });
    console.log(`   Output: ${output}`);
    console.log("   ✓ Generation complete\n");

    // Test 5: RAG
    console.log("5️⃣  Testing RAG...");
    const rag = app.rag();
    rag.add("Forge was created in 2024");
    rag.add("Forge uses Rust for performance");
    const answer = await rag.query("When was Forge created?");
    console.log(`   Answer: ${answer}`);
    console.log("   ✓ RAG complete\n");

    // Test 6: Agent
    console.log("6️⃣  Testing Agent...");
    const agent = app.agent();
    agent.tool(createTool("calculator", "Calculate math", (expr) => {
        const match = expr.match(/(\d+)\s*\*\s*(\d+)/);
        if (match) return parseInt(match[1]) * parseInt(match[2]);
        return "No expression found";
    }));
    const result = await agent.run("Calculate 25 * 4");
    console.log(`   Result: ${result}`);
    console.log("   ✓ Agent complete\n");

    // Test 7: Server (just create, don't listen)
    console.log("7️⃣  Testing server setup...");
    app.get("/health", (req, res) => {
        res.json({ status: "ok" });
    });
    app.post("/chat", async (req, res) => {
        const out = await app.generate(req.body.message);
        res.json({ output: out });
    });
    console.log("   ✓ Routes added\n");

    console.log("═══════════════════════════════════════════════════════");
    console.log("  ✅ All tests passed!");
    console.log("═══════════════════════════════════════════════════════\n");
}

test().catch(console.error);
