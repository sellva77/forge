/**
 * Forge - Comprehensive Functionality Test
 * =========================================
 * 
 * Tests all core AI/ML capabilities of the framework.
 * Run: node test_functionality.js
 */

const {
    forge,
    Tensor,
    Transformer,
    TRANSFORMER_PRESETS,
    BPETokenizer,
    Model,
    Generator,
    Trainer,
    getBackendInfo,
} = require("./dist");

async function runTests() {
    console.log("\n" + "═".repeat(60));
    console.log("🔥 FORGE FRAMEWORK - FUNCTIONALITY TESTS");
    console.log("═".repeat(60) + "\n");

    let passed = 0;
    let failed = 0;

    function test(name, fn) {
        try {
            fn();
            console.log(`✅ ${name}`);
            passed++;
        } catch (e) {
            console.log(`❌ ${name}`);
            console.log(`   Error: ${e.message}`);
            failed++;
        }
    }

    async function asyncTest(name, fn) {
        try {
            await fn();
            console.log(`✅ ${name}`);
            passed++;
        } catch (e) {
            console.log(`❌ ${name}`);
            console.log(`   Error: ${e.message}`);
            failed++;
        }
    }

    // ===========================================================================
    // TENSOR OPERATIONS
    // ===========================================================================
    console.log("\n📐 TENSOR OPERATIONS\n" + "-".repeat(40));

    test("Tensor creation and shape", () => {
        const t = new Tensor([1, 2, 3, 4, 5, 6], [2, 3]);
        if (t.shape[0] !== 2 || t.shape[1] !== 3) throw new Error("Shape mismatch");
        if (t.size !== 6) throw new Error("Size mismatch");
    });

    test("Tensor.zeros()", () => {
        const t = Tensor.zeros([3, 3]);
        const sum = t.data.reduce((a, b) => a + b, 0);
        if (sum !== 0) throw new Error("Not all zeros");
    });

    test("Tensor.ones()", () => {
        const t = Tensor.ones([3, 3]);
        const sum = t.data.reduce((a, b) => a + b, 0);
        if (sum !== 9) throw new Error("Not all ones");
    });

    test("Tensor.randn() - normal distribution", () => {
        const t = Tensor.randn([1000]);
        const mean = t.data.reduce((a, b) => a + b, 0) / 1000;
        if (Math.abs(mean) > 0.2) throw new Error(`Mean too far from 0: ${mean}`);
    });

    test("Tensor.matmul()", () => {
        const a = new Tensor([1, 2, 3, 4, 5, 6], [2, 3]);
        const b = new Tensor([1, 2, 3, 4, 5, 6], [3, 2]);
        const c = a.matmul(b);
        if (c.shape[0] !== 2 || c.shape[1] !== 2) throw new Error("Result shape wrong");
        // [1,2,3] · [1,3,5] = 1+6+15 = 22
        if (Math.abs(c.data[0] - 22) > 0.01) throw new Error("Matmul result wrong");
    });

    test("Tensor.add()", () => {
        const a = Tensor.ones([3]);
        const b = Tensor.ones([3]);
        const c = a.add(b);
        if (c.data[0] !== 2) throw new Error("Add failed");
    });

    test("Tensor.mul()", () => {
        const a = new Tensor([2, 3, 4], [3]);
        const b = new Tensor([2, 2, 2], [3]);
        const c = a.mul(b);
        if (c.data[0] !== 4 || c.data[1] !== 6) throw new Error("Mul failed");
    });

    test("Tensor.softmax()", () => {
        const t = new Tensor([1, 2, 3, 4], [4]);
        const s = t.softmax();
        const sum = s.data.reduce((a, b) => a + b, 0);
        if (Math.abs(sum - 1) > 0.001) throw new Error(`Softmax sum is ${sum}, not 1`);
        // Check values are ordered
        if (s.data[3] <= s.data[0]) throw new Error("Softmax order wrong");
    });

    test("Tensor.gelu()", () => {
        const t = new Tensor([-1, 0, 1, 2], [4]);
        const g = t.gelu();
        // GELU(0) should be ~0
        if (Math.abs(g.data[1]) > 0.01) throw new Error("GELU(0) wrong");
        // GELU(2) should be positive
        if (g.data[3] < 1) throw new Error("GELU(2) wrong");
    });

    test("Tensor.layerNorm()", () => {
        const t = new Tensor([1, 2, 3, 4], [1, 4]);
        const n = t.layerNorm();
        const mean = n.data.reduce((a, b) => a + b, 0) / 4;
        if (Math.abs(mean) > 0.01) throw new Error("LayerNorm mean not 0");
    });

    test("Tensor.rmsNorm()", () => {
        const t = Tensor.randn([1, 64]);
        const n = t.rmsNorm();
        // RMS of output should be close to 1
        const rms = Math.sqrt(n.data.reduce((a, b) => a + b * b, 0) / 64);
        if (Math.abs(rms - 1) > 0.2) throw new Error(`RMS norm result: ${rms}`);
    });

    // ===========================================================================
    // TOKENIZER
    // ===========================================================================
    console.log("\n📝 TOKENIZER\n" + "-".repeat(40));

    test("BPETokenizer.encode()", () => {
        const tok = new BPETokenizer({ vocabSize: 32000 });
        const tokens = tok.encode("Hello world");
        if (tokens.length < 3) throw new Error("Too few tokens");
        // BOS token ID is 2 (not 1)
        if (tokens[0] !== 2) throw new Error(`Missing BOS token, got ${tokens[0]}`);
    });

    test("BPETokenizer.decode()", () => {
        const tok = new BPETokenizer({ vocabSize: 32000 });
        const tokens = tok.encode("Hello");
        const decoded = tok.decode(tokens, true);
        if (!decoded.includes("Hello")) throw new Error(`Decoded: "${decoded}"`);
    });

    test("BPETokenizer round-trip", () => {
        const tok = new BPETokenizer({ vocabSize: 32000 });
        const text = "The quick brown fox";
        const tokens = tok.encode(text, false, false);
        const decoded = tok.decode(tokens, true);
        if (decoded !== text) throw new Error(`"${decoded}" !== "${text}"`);
    });

    test("BPETokenizer.encodeBatch()", () => {
        const tok = new BPETokenizer({ vocabSize: 32000 });
        const batch = ["Hello", "World", "Forge AI"];
        const encoded = tok.encodeBatch(batch);
        if (encoded.length !== 3) throw new Error("Batch size wrong");
    });

    // ===========================================================================
    // TRANSFORMER ARCHITECTURE
    // ===========================================================================
    console.log("\n🤖 TRANSFORMER ARCHITECTURE\n" + "-".repeat(40));

    test("Transformer creation (tiny)", () => {
        const config = TRANSFORMER_PRESETS.tiny;
        const t = new Transformer(config);
        if (!t) throw new Error("Failed to create transformer");
    });

    test("Transformer.forward()", () => {
        const config = TRANSFORMER_PRESETS.tiny;
        const t = new Transformer(config);
        const output = t.forward([1, 2, 3, 4]);
        // Output is a tensor with shape [seq_len, vocab_size] or flattened
        if (!output.data || output.data.length === 0) throw new Error("No output data");
        // Should have vocabSize elements for last token
        if (output.data.length < config.vocabSize) throw new Error("Output too small");
    });

    test("Transformer.generate()", () => {
        const config = TRANSFORMER_PRESETS.tiny;
        const t = new Transformer(config);
        const output = t.generate([1, 100, 200], 5, { temperature: 0.8 });
        if (output.length !== 8) throw new Error(`Length should be 8, got ${output.length}`);
    });

    test("Transformer KV cache", () => {
        const config = TRANSFORMER_PRESETS.tiny;
        const t = new Transformer(config);

        // First forward pass
        t.forward([1, 2, 3]);

        // Second forward (should use cache)
        const out2 = t.forward([1, 2, 3, 4], 3);

        // Clear cache
        t.clearCache();

        // Should work after clearing
        const out3 = t.forward([1, 2]);
        if (!out3) throw new Error("Forward after clear failed");
    });

    test("Transformer.estimateParams()", () => {
        const t = new Transformer(TRANSFORMER_PRESETS.tiny);
        const params = t.estimateParams();
        if (params < 1000000) throw new Error(`Too few params: ${params}`);
    });

    // ===========================================================================
    // MODEL CLASS
    // ===========================================================================
    console.log("\n🎯 MODEL CLASS\n" + "-".repeat(40));

    test("Model creation", () => {
        const m = new Model("tiny");
        if (m.name !== "tiny") throw new Error("Wrong model name");
    });

    test("Model.forward()", () => {
        const m = new Model("tiny");
        const logits = m.forward([1, 2, 3]);
        if (!(logits instanceof Float32Array)) throw new Error("Should return Float32Array");
    });

    test("Model.generate()", () => {
        const m = new Model("tiny");
        const tokens = m.generate([1, 100], 10, { temperature: 0.7 });
        if (tokens.length !== 12) throw new Error(`Length should be 12, got ${tokens.length}`);
    });

    test("Model.encode() and decode()", () => {
        const m = new Model("tiny");
        const tokens = m.encode("Hello");
        const text = m.decode(tokens);
        if (typeof text !== "string") throw new Error("Decode should return string");
    });

    // ===========================================================================
    // GENERATOR
    // ===========================================================================
    console.log("\n✨ GENERATOR\n" + "-".repeat(40));

    await asyncTest("Generator.generate()", async () => {
        const m = new Model("tiny");
        const g = new Generator(m);
        const output = await g.generate("Hello", { maxTokens: 10 });
        if (typeof output !== "string") throw new Error("Should return string");
    });

    await asyncTest("Generator.stream()", async () => {
        const m = new Model("tiny");
        const g = new Generator(m);
        let tokens = 0;
        for await (const token of g.stream("Hi", { maxTokens: 5 })) {
            tokens++;
            if (typeof token !== "string") throw new Error("Token should be string");
        }
        if (tokens === 0) throw new Error("No tokens generated");
    });

    // ===========================================================================
    // FORGE APP
    // ===========================================================================
    console.log("\n🔥 FORGE APP\n" + "-".repeat(40));

    test("forge() factory", () => {
        const app = forge();
        if (!app) throw new Error("Factory failed");
    });

    test("forge({ model: 'tiny' })", () => {
        const app = forge({ model: "tiny" });
        const m = app.model();
        if (m.name !== "tiny") throw new Error("Wrong model");
    });

    test("app.model().isUsingRealTransformer()", () => {
        const app = forge({ model: "tiny" });
        const m = app.model();
        if (!m.isUsingRealTransformer()) throw new Error("Should use real transformer");
    });

    await asyncTest("app.generate()", async () => {
        const app = forge({ model: "tiny" });
        const output = await app.generate("Test", { maxTokens: 5 });
        if (typeof output !== "string") throw new Error("Should return string");
    });

    // ===========================================================================
    // TRAINING
    // ===========================================================================
    console.log("\n📚 TRAINING\n" + "-".repeat(40));

    await asyncTest("Trainer with small dataset", async () => {
        const m = new Model("tiny");
        const t = new Trainer(m, {
            epochs: 1,
            batchSize: 1,
            logInterval: 100,
            maxSeqLen: 8  // Very short for testing
        });

        const data = ["Hi"];  // Single very short entry
        const results = await t.train(data);

        if (results.length === 0) throw new Error("No training results");
        if (typeof results[0].loss !== "number") throw new Error("No loss value");
    });

    await asyncTest("Trainer.evaluate()", async () => {
        const m = new Model("tiny");
        const t = new Trainer(m, { maxSeqLen: 8 });

        const result = await t.evaluate(["Hi"]);

        if (typeof result.loss !== "number") throw new Error("No loss");
        if (typeof result.perplexity !== "number") throw new Error("No perplexity");
    });

    // ===========================================================================
    // SUMMARY
    // ===========================================================================
    console.log("\n" + "═".repeat(60));
    console.log(`📊 RESULTS: ${passed} passed, ${failed} failed`);
    console.log("═".repeat(60));

    if (failed === 0) {
        console.log("\n🎉 All tests passed! Framework is functional.\n");
    } else {
        console.log(`\n⚠️  ${failed} test(s) failed. See above for details.\n`);
        process.exit(1);
    }
}

// Run tests
runTests().catch(err => {
    console.error("Test runner error:", err);
    process.exit(1);
});
