/**
 * 🔥 My First Forge Model
 * ========================
 * 
 * Example using the forge-ai npm package.
 * Run: node model.js
 */

// Import from the forge-ai package
import { forge, printBackendInfo, logger, timer } from "forge-ai";

async function main() {
    console.log("\n" + "═".repeat(50));
    console.log("🔥 MY FIRST FORGE MODEL");
    console.log("═".repeat(50) + "\n");

    // Show what backend we're using
    printBackendInfo();

    // Create a Forge app with tiny model
    const app = forge({ model: "tiny" });

    // Add middleware for logging and timing
    app.use(logger);
    app.use(timer);

    console.log(`📦 Model: ${app.model()}\n`);

    // Generate text
    console.log("✨ Generating text...");
    const output = await app.generate("Hello world", {
        maxTokens: 20,
        temperature: 0.8
    });
    console.log(`   Output: "${output}"\n`);

    // Stream generation
    console.log("📡 Streaming tokens...");
    process.stdout.write("   ");
    for await (const token of app.stream("AI is", { maxTokens: 15 })) {
        process.stdout.write(token);
    }
    console.log("\n");

    // Try RAG if you want
    console.log("📚 Testing RAG...");
    const rag = app.rag();
    rag.add("Forge was created in 2024");
    rag.add("Forge is an AI framework for JavaScript");
    const answer = await rag.query("When was Forge created?");
    console.log(`   Answer: "${answer}"\n`);

    console.log("═".repeat(50));
    console.log("✅ Done!");
    console.log("═".repeat(50) + "\n");
}

main().catch(err => {
    console.error("❌ Error:", err.message);
    console.log("\n💡 Tips:");
    console.log("   1. Make sure to run 'npm install' first");
    console.log("   2. Build the package with 'npm run build' in packages/forge");
    console.log("   3. Link the package with 'npm link' if developing locally");
});