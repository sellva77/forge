/**
 * 🔥 Forge - Step 1: Quick Start
 * ===============================
 * 
 * The simplest way to use Forge.
 * Run: node examples/01_quickstart.js
 */

// Import forge (ES Module)
import { forge, printBackendInfo } from "../packages/forge/dist/index.js";

async function main() {
    console.log("\n" + "═".repeat(60));
    console.log("🔥 FORGE QUICKSTART");
    console.log("═".repeat(60) + "\n");

    // Step 1: Check what backend is being used
    console.log("📦 Step 1: Check Backend");
    printBackendInfo();

    // Step 2: Create a Forge app
    console.log("🚀 Step 2: Create Forge App");
    const app = forge({ model: "tiny" });
    console.log(`   Created: ${app.model()}\n`);

    // Step 3: Generate text
    console.log("✨ Step 3: Generate Text");
    console.log("   Prompt: 'Hello'");

    const output = await app.generate("Hello", {
        maxTokens: 20,
        temperature: 0.8,
    });

    console.log(`   Generated: "${output}"\n`);

    // Step 4: Stream generation
    console.log("📡 Step 4: Stream Generation");
    process.stdout.write("   Streaming: ");

    for await (const token of app.stream("Hi", { maxTokens: 15 })) {
        process.stdout.write(token);
    }
    console.log("\n");

    console.log("═".repeat(60));
    console.log("✅ Quickstart Complete!");
    console.log("═".repeat(60) + "\n");
}

main().catch(console.error);
