/**
 * 🔥 Forge - Step 3: Using Middleware
 * ====================================
 * 
 * Add middleware for logging, timing, caching, and custom processing.
 * Run: node examples/03_middleware.js
 */

const {
    forge,
    logger,
    timer,
    cache,
    rateLimit,
    normalize,
    lowercase
} = require("../packages/forge/dist");

async function main() {
    console.log("\n" + "═".repeat(60));
    console.log("🔥 FORGE MIDDLEWARE EXAMPLE");
    console.log("═".repeat(60) + "\n");

    // Create app
    const app = forge({ model: "tiny" });

    // =========================================================================
    // Example 1: Built-in Logger Middleware
    // =========================================================================
    console.log("📝 Example 1: Logger Middleware");
    console.log("-".repeat(40));

    const app1 = forge({ model: "tiny" });
    app1.use(logger);  // Logs input and output

    await app1.generate("Hello", { maxTokens: 10 });
    console.log("");

    // =========================================================================
    // Example 2: Timer Middleware
    // =========================================================================
    console.log("⏱️  Example 2: Timer Middleware");
    console.log("-".repeat(40));

    const app2 = forge({ model: "tiny" });
    app2.use(timer);  // Adds timing info

    await app2.generate("Test timing", { maxTokens: 10 });
    console.log("");

    // =========================================================================
    // Example 3: Custom Middleware
    // =========================================================================
    console.log("🔧 Example 3: Custom Middleware");
    console.log("-".repeat(40));

    const app3 = forge({ model: "tiny" });

    // Custom middleware that modifies input
    app3.use(async (ctx, next) => {
        console.log(`   Before: "${ctx.input}"`);

        // Modify input
        ctx.input = ctx.input.toUpperCase();
        console.log(`   Modified: "${ctx.input}"`);

        // Call next middleware
        await next();

        // Modify output
        console.log(`   After: "${ctx.output?.substring(0, 30)}..."`);
    });

    await app3.generate("hello world", { maxTokens: 10 });
    console.log("");

    // =========================================================================
    // Example 4: Chaining Multiple Middleware
    // =========================================================================
    console.log("🔗 Example 4: Chaining Middleware");
    console.log("-".repeat(40));

    const app4 = forge({ model: "tiny" });

    // Chain multiple middleware
    app4.use(normalize);    // Trim whitespace
    app4.use(lowercase);    // Convert to lowercase
    app4.use(async (ctx, next) => {
        console.log(`   Processed input: "${ctx.input}"`);
        await next();
    });

    await app4.generate("   HELLO WORLD   ", { maxTokens: 10 });
    console.log("");

    // =========================================================================
    // Example 5: System Prompt Middleware
    // =========================================================================
    console.log("💬 Example 5: System Prompt Middleware");
    console.log("-".repeat(40));

    const app5 = forge({ model: "tiny" });

    // Add system prompt to every request
    app5.use(async (ctx, next) => {
        const systemPrompt = "You are a helpful assistant. ";
        ctx.input = systemPrompt + "User: " + ctx.input + "\nAssistant:";
        console.log(`   Full prompt: "${ctx.input.substring(0, 50)}..."`);
        await next();
    });

    await app5.generate("What is AI?", { maxTokens: 15 });
    console.log("");

    console.log("═".repeat(60));
    console.log("✅ Middleware Example Complete!");
    console.log("═".repeat(60) + "\n");
}

main().catch(console.error);
