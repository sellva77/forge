/**
 * 🔥 Forge - Step 6: HTTP API Server
 * ===================================
 * 
 * Create an Express-style API server for your AI model.
 * Run: node examples/06_server.js
 * 
 * Then test with:
 *   curl http://localhost:3000/health
 *   curl -X POST http://localhost:3000/generate -H "Content-Type: application/json" -d '{"prompt":"Hello"}'
 */

const { forge, logger, timer } = require("../packages/forge/dist");

async function main() {
    console.log("\n" + "═".repeat(60));
    console.log("🔥 FORGE HTTP SERVER");
    console.log("═".repeat(60) + "\n");

    // Create app with model
    const app = forge({ model: "tiny" });

    // Add middleware
    app.use(logger);  // Log all requests
    app.use(timer);   // Track timing

    // =========================================================================
    // Route: Home
    // =========================================================================
    app.get("/", (req, res) => {
        res.json({
            name: "Forge AI Server",
            version: "1.0.0",
            endpoints: [
                "GET  /           - This info",
                "GET  /health     - Health check",
                "POST /generate   - Generate text",
                "POST /chat       - Chat interface",
                "POST /embed      - Get embeddings",
            ]
        });
    });

    // =========================================================================
    // Route: Health Check (built-in)
    // =========================================================================
    // Already registered by app.listen()

    // =========================================================================
    // Route: Chat
    // =========================================================================
    app.post("/chat", async (req, res) => {
        try {
            const { message, history = [] } = req.body;

            if (!message) {
                return res.status(400).json({ error: "Message required" });
            }

            // Build context from history
            let context = history
                .map(m => `${m.role}: ${m.content}`)
                .join("\n");

            if (context) context += "\n";
            context += `User: ${message}\nAssistant:`;

            // Generate response
            const response = await app.generate(context, {
                maxTokens: 50,
                temperature: 0.7,
                stop: ["\nUser:", "\n\n"],
            });

            res.json({
                response: response.trim(),
                model: app.model().name,
            });

        } catch (error) {
            res.status(500).json({ error: error.message });
        }
    });

    // =========================================================================
    // Route: Stream Generation (Server-Sent Events)
    // =========================================================================
    app.get("/stream", async (req, res) => {
        const prompt = req.query.prompt || "Hello";

        res.setHeader("Content-Type", "text/event-stream");
        res.setHeader("Cache-Control", "no-cache");
        res.setHeader("Connection", "keep-alive");

        try {
            for await (const token of app.stream(prompt, { maxTokens: 30 })) {
                res.write(`data: ${JSON.stringify({ token })}\n\n`);
            }
            res.write(`data: ${JSON.stringify({ done: true })}\n\n`);
        } catch (error) {
            res.write(`data: ${JSON.stringify({ error: error.message })}\n\n`);
        }

        res.end();
    });

    // =========================================================================
    // Route: Batch Generation
    // =========================================================================
    app.post("/batch", async (req, res) => {
        try {
            const { prompts, maxTokens = 30 } = req.body;

            if (!prompts || !Array.isArray(prompts)) {
                return res.status(400).json({ error: "Prompts array required" });
            }

            const results = await Promise.all(
                prompts.map(prompt =>
                    app.generate(prompt, { maxTokens })
                )
            );

            res.json({
                results: prompts.map((prompt, i) => ({
                    prompt,
                    completion: results[i],
                })),
            });

        } catch (error) {
            res.status(500).json({ error: error.message });
        }
    });

    // =========================================================================
    // Start Server
    // =========================================================================
    const PORT = process.env.PORT || 3000;

    app.listen(PORT, () => {
        console.log(`🚀 Server running at http://localhost:${PORT}`);
        console.log("");
        console.log("📡 Available endpoints:");
        console.log(`   GET  http://localhost:${PORT}/`);
        console.log(`   GET  http://localhost:${PORT}/health`);
        console.log(`   POST http://localhost:${PORT}/generate`);
        console.log(`   POST http://localhost:${PORT}/chat`);
        console.log(`   GET  http://localhost:${PORT}/stream?prompt=Hello`);
        console.log(`   POST http://localhost:${PORT}/batch`);
        console.log("");
        console.log("💡 Test with:");
        console.log(`   curl http://localhost:${PORT}/health`);
        console.log(`   curl -X POST http://localhost:${PORT}/generate -H "Content-Type: application/json" -d '{"prompt":"Hello"}'`);
        console.log("");
    });
}

main().catch(console.error);
