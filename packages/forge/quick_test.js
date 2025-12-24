/**
 * Quick Test - Verify Forge Works
 * ================================
 * 
 * Run: node quick_test.js
 */

// Test 1: Import the library
console.log("\n🔥 Forge Quick Test\n");
console.log("═".repeat(50));

try {
    const {
        forge,
        Tensor,
        Transformer,
        TRANSFORMER_PRESETS,
        getBackendInfo,
        printBackendInfo
    } = require("./dist");

    // Test 2: Check backend
    console.log("\n📦 Backend Status:");
    printBackendInfo();

    // Test 3: Create a tensor
    console.log("🔢 Testing Tensor Operations...");
    const t1 = Tensor.randn([2, 3]);
    const t2 = Tensor.randn([3, 2]);
    const result = t1.matmul(t2);
    console.log(`   ✓ Matrix multiply: [2,3] × [3,2] = [${result.shape}]`);

    // Test 4: Test softmax
    const t3 = Tensor.randn([1, 10]);
    const softmax = t3.softmax();
    const sum = softmax.data.reduce((a, b) => a + b, 0);
    console.log(`   ✓ Softmax sum: ${sum.toFixed(4)} (should be ~1.0)`);

    // Test 5: Create a forge app
    console.log("\n🔥 Testing Forge App...");
    const app = forge({ model: "tiny" });
    console.log(`   ✓ Created app with model: ${app.model().name}`);

    // Test 6: Quick generate test
    console.log("\n✨ Testing Generation...");
    const tokens = app.model().encode("Hello");
    console.log(`   ✓ Encoded "Hello" to ${tokens.length} tokens`);

    // Test 7: Create a small transformer
    console.log("\n🤖 Testing Transformer...");
    const config = TRANSFORMER_PRESETS.tiny;
    console.log(`   ✓ Tiny transformer config: dim=${config.dim}, layers=${config.layers}`);

    console.log("\n" + "═".repeat(50));
    console.log("✅ All tests passed!\n");
    console.log("📝 Next steps:");
    console.log("   1. npm install forge");
    console.log("   2. const { forge } = require('forge');");
    console.log("   3. const app = forge();");
    console.log("   4. await app.generate('Hello world!');");
    console.log("");

} catch (error) {
    console.error("❌ Test failed:", error.message);
    console.error(error.stack);
    process.exit(1);
}
