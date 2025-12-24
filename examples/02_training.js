/**
 * 🔥 Forge - Step 2: Training a Model
 * ====================================
 * 
 * Train a model on your own data.
 * Run: node examples/02_training.js
 */

const { forge, Trainer, Model } = require("../packages/forge/dist");

async function main() {
    console.log("\n" + "═".repeat(60));
    console.log("🔥 FORGE TRAINING EXAMPLE");
    console.log("═".repeat(60) + "\n");

    // Step 1: Create a model
    console.log("📦 Step 1: Create Model");
    const app = forge({ model: "tiny" });
    console.log(`   Model: ${app.model()}\n`);

    // Step 2: Prepare training data
    console.log("📚 Step 2: Prepare Training Data");
    const trainingData = [
        "Forge is an AI framework for JavaScript",
        "It uses an Express-style API for simplicity",
        "Training AI models is now easy with Forge",
        "The framework includes real tensor operations",
        "Transformers power modern language models",
    ];
    console.log(`   Loaded ${trainingData.length} training samples\n`);

    // Step 3: Set up event listeners for progress
    console.log("📊 Step 3: Training with Progress");

    app.on("step", ({ step, loss, lr }) => {
        // Log every step
        // console.log(`   Step ${step}: loss=${loss.toFixed(4)}`);
    });

    app.on("epoch", ({ epoch, avgLoss }) => {
        console.log(`   ✓ Epoch ${epoch} complete - avg loss: ${avgLoss.toFixed(4)}`);
    });

    // Step 4: Train!
    const startTime = Date.now();

    await app.train(trainingData, {
        epochs: 3,
        lr: 0.0001,
        batchSize: 2,
        maxSeqLen: 32,
    });

    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    console.log(`\n   Training completed in ${elapsed}s\n`);

    // Step 5: Test the trained model
    console.log("✨ Step 5: Test Generation");

    const prompts = ["Forge is", "Training", "AI"];

    for (const prompt of prompts) {
        const output = await app.generate(prompt, { maxTokens: 15 });
        console.log(`   "${prompt}" → "${output}"`);
    }

    console.log("\n" + "═".repeat(60));
    console.log("✅ Training Example Complete!");
    console.log("═".repeat(60) + "\n");
}

main().catch(console.error);
