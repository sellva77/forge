/**
 * 🔥 Forge - Step 5: Transformer Architecture
 * ============================================
 * 
 * Deep dive into the transformer implementation.
 * Run: node examples/05_transformer.js
 */

const {
    Transformer,
    TRANSFORMER_PRESETS,
    BPETokenizer,
    Model,
} = require("../packages/forge/dist");

async function main() {
    console.log("\n" + "═".repeat(60));
    console.log("🔥 FORGE TRANSFORMER DEEP DIVE");
    console.log("═".repeat(60) + "\n");

    // =========================================================================
    // Available Model Presets
    // =========================================================================
    console.log("📦 Available Model Presets");
    console.log("-".repeat(40));

    for (const [name, config] of Object.entries(TRANSFORMER_PRESETS)) {
        const params = estimateParams(config);
        const size = formatSize(params);
        console.log(`   ${name.padEnd(8)} dim=${config.dim.toString().padEnd(5)} layers=${config.layers.toString().padEnd(3)} heads=${config.heads.toString().padEnd(3)} → ${size}`);
    }
    console.log("");

    // =========================================================================
    // Create a Transformer
    // =========================================================================
    console.log("🤖 Creating Transformer");
    console.log("-".repeat(40));

    const config = TRANSFORMER_PRESETS.tiny;
    console.log(`   Config: dim=${config.dim}, layers=${config.layers}, heads=${config.heads}`);

    const transformer = new Transformer(config);
    console.log(`   Created: ${transformer}`);
    console.log("");

    // =========================================================================
    // Tokenization
    // =========================================================================
    console.log("📝 Tokenization");
    console.log("-".repeat(40));

    const tokenizer = new BPETokenizer({ vocabSize: config.vocabSize });

    const text = "Hello world!";
    const tokens = tokenizer.encode(text);
    console.log(`   Text: "${text}"`);
    console.log(`   Tokens: [${tokens.join(", ")}]`);
    console.log(`   Decoded: "${tokenizer.decode(tokens, true)}"`);
    console.log("");

    // =========================================================================
    // Forward Pass
    // =========================================================================
    console.log("➡️  Forward Pass");
    console.log("-".repeat(40));

    const inputTokens = [1, 100, 200, 300];  // Example token IDs
    console.log(`   Input tokens: [${inputTokens.join(", ")}]`);

    const logits = transformer.forward(inputTokens);
    console.log(`   Output shape: [${logits.shape.join(", ")}]`);
    console.log(`   Vocab size: ${config.vocabSize}`);

    // Get top-5 predicted tokens for last position
    const lastLogits = getLastLogits(logits, config.vocabSize);
    const topTokens = getTopK(lastLogits, 5);
    console.log(`   Top 5 predictions: [${topTokens.join(", ")}]`);
    console.log("");

    // =========================================================================
    // Generation
    // =========================================================================
    console.log("✨ Generation");
    console.log("-".repeat(40));

    const promptTokens = tokenizer.encode("Hello");
    console.log(`   Prompt tokens: [${promptTokens.join(", ")}]`);

    transformer.clearCache();  // Clear KV cache before generation

    const generated = transformer.generate(promptTokens, 10, {
        temperature: 0.8,
        topK: 40,
        topP: 0.9,
    });

    console.log(`   Generated tokens: [${generated.join(", ")}]`);
    console.log(`   Decoded: "${tokenizer.decode(generated, true)}"`);
    console.log("");

    // =========================================================================
    // KV Cache (For Efficient Generation)
    // =========================================================================
    console.log("💾 KV Cache (Efficient Generation)");
    console.log("-".repeat(40));

    transformer.clearCache();

    // First pass: encode full prompt
    console.log("   Pass 1: Full prompt");
    const start1 = Date.now();
    transformer.forward([1, 100, 200, 300, 400, 500]);
    console.log(`     Time: ${Date.now() - start1}ms`);

    // Second pass: just the new token (uses cache)
    console.log("   Pass 2: Single token (cached)");
    const start2 = Date.now();
    transformer.forward([600], 6);  // startPos = 6
    console.log(`     Time: ${Date.now() - start2}ms`);

    console.log("   → Cached decoding is more efficient!\n");

    // =========================================================================
    // Model Class (High-Level Interface)
    // =========================================================================
    console.log("🎯 Model Class (High-Level)");
    console.log("-".repeat(40));

    const model = new Model("small");
    console.log(`   Model: ${model}`);
    console.log(`   Using real transformer: ${model.isUsingRealTransformer()}`);

    // Generate text
    const output = await model.generateText("Hello", 15, { temperature: 0.7 });
    console.log(`   Generated: "${output}"`);
    console.log("");

    console.log("═".repeat(60));
    console.log("✅ Transformer Deep Dive Complete!");
    console.log("═".repeat(60) + "\n");
}

// Helper functions
function estimateParams(config) {
    const embParams = config.vocabSize * config.dim;
    const attnParams = 4 * config.dim * config.heads * config.headDim * config.layers;
    const ffnParams = 3 * config.dim * config.hiddenDim * config.layers;
    return embParams + attnParams + ffnParams;
}

function formatSize(params) {
    if (params > 1e9) return `${(params / 1e9).toFixed(1)}B`;
    if (params > 1e6) return `${(params / 1e6).toFixed(1)}M`;
    return `${(params / 1e3).toFixed(1)}K`;
}

function getLastLogits(tensor, vocabSize) {
    const seq = tensor.shape[1];
    const offset = (seq - 1) * vocabSize;
    return tensor.data.slice(offset, offset + vocabSize);
}

function getTopK(logits, k) {
    const indices = Array.from({ length: logits.length }, (_, i) => i);
    indices.sort((a, b) => logits[b] - logits[a]);
    return indices.slice(0, k);
}

main().catch(console.error);
