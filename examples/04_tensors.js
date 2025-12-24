/**
 * 🔥 Forge - Step 4: Tensor Operations
 * =====================================
 * 
 * Direct tensor manipulation for custom ML operations.
 * Run: node examples/04_tensors.js
 */

const {
    Tensor,
    cat,
    stack,
} = require("../packages/forge/dist");

async function main() {
    console.log("\n" + "═".repeat(60));
    console.log("🔥 FORGE TENSOR OPERATIONS");
    console.log("═".repeat(60) + "\n");

    // =========================================================================
    // Creating Tensors
    // =========================================================================
    console.log("📦 Creating Tensors");
    console.log("-".repeat(40));

    // From array
    const t1 = new Tensor([1, 2, 3, 4, 5, 6], [2, 3]);
    console.log(`   From array [2, 3]: ${JSON.stringify(Array.from(t1.data))}`);

    // Zeros
    const zeros = Tensor.zeros([3, 3]);
    console.log(`   Zeros [3, 3]: ${zeros.data.slice(0, 5)}...`);

    // Ones
    const ones = Tensor.ones([2, 4]);
    console.log(`   Ones [2, 4]: ${Array.from(ones.data)}`);

    // Random normal
    const randn = Tensor.randn([4], 0, 1);
    console.log(`   Random [4]: ${Array.from(randn.data).map(x => x.toFixed(3))}`);
    console.log("");

    // =========================================================================
    // Basic Operations
    // =========================================================================
    console.log("🔢 Basic Operations");
    console.log("-".repeat(40));

    const a = new Tensor([1, 2, 3, 4], [2, 2]);
    const b = new Tensor([2, 2, 2, 2], [2, 2]);

    // Addition
    const sum = a.add(b);
    console.log(`   [1,2,3,4] + [2,2,2,2] = ${Array.from(sum.data)}`);

    // Multiplication
    const prod = a.mul(b);
    console.log(`   [1,2,3,4] * [2,2,2,2] = ${Array.from(prod.data)}`);

    // Scalar operations
    const scaled = a.mul(10);
    console.log(`   [1,2,3,4] * 10 = ${Array.from(scaled.data)}`);
    console.log("");

    // =========================================================================
    // Matrix Operations
    // =========================================================================
    console.log("📐 Matrix Operations");
    console.log("-".repeat(40));

    const m1 = new Tensor([1, 2, 3, 4, 5, 6], [2, 3]);  // 2x3
    const m2 = new Tensor([1, 2, 3, 4, 5, 6], [3, 2]);  // 3x2

    const result = m1.matmul(m2);  // 2x2
    console.log(`   [2,3] @ [3,2] = [${result.shape}]`);
    console.log(`   Result: ${Array.from(result.data)}`);

    // Transpose
    const transposed = m1.transpose();
    console.log(`   Transpose [2,3] → [${transposed.shape}]`);
    console.log("");

    // =========================================================================
    // Activation Functions
    // =========================================================================
    console.log("⚡ Activation Functions");
    console.log("-".repeat(40));

    const x = new Tensor([-2, -1, 0, 1, 2], [5]);

    // ReLU
    const relu = x.relu();
    console.log(`   ReLU([-2,-1,0,1,2]) = ${Array.from(relu.data)}`);

    // GELU (used in transformers)
    const gelu = x.gelu();
    console.log(`   GELU([-2,-1,0,1,2]) = ${Array.from(gelu.data).map(v => v.toFixed(3))}`);

    // Sigmoid
    const sigmoid = x.sigmoid();
    console.log(`   Sigmoid([-2,-1,0,1,2]) = ${Array.from(sigmoid.data).map(v => v.toFixed(3))}`);

    // SiLU (Swish)
    const silu = x.silu();
    console.log(`   SiLU([-2,-1,0,1,2]) = ${Array.from(silu.data).map(v => v.toFixed(3))}`);
    console.log("");

    // =========================================================================
    // Normalization
    // =========================================================================
    console.log("📊 Normalization");
    console.log("-".repeat(40));

    const data = new Tensor([1, 2, 3, 4, 5], [1, 5]);

    // Softmax
    const softmax = data.softmax();
    const softmaxSum = Array.from(softmax.data).reduce((a, b) => a + b, 0);
    console.log(`   Softmax sum: ${softmaxSum.toFixed(4)} (should be 1.0)`);
    console.log(`   Softmax: ${Array.from(softmax.data).map(v => v.toFixed(3))}`);

    // Layer Norm
    const layerNorm = data.layerNorm();
    const mean = Array.from(layerNorm.data).reduce((a, b) => a + b, 0) / 5;
    console.log(`   LayerNorm mean: ${mean.toFixed(4)} (should be ~0)`);

    // RMS Norm
    const rmsNorm = data.rmsNorm();
    const rms = Math.sqrt(Array.from(rmsNorm.data).reduce((a, b) => a + b * b, 0) / 5);
    console.log(`   RMSNorm RMS: ${rms.toFixed(4)} (should be ~1)`);
    console.log("");

    // =========================================================================
    // Reductions
    // =========================================================================
    console.log("📉 Reductions");
    console.log("-".repeat(40));

    const vals = new Tensor([1, 5, 3, 9, 2, 7], [2, 3]);

    console.log(`   Sum: ${vals.sum()}`);
    console.log(`   Mean: ${vals.mean()}`);
    console.log(`   Max: ${vals.max()}`);
    console.log(`   Min: ${vals.min()}`);
    console.log(`   Argmax: ${vals.argmax()}`);
    console.log("");

    console.log("═".repeat(60));
    console.log("✅ Tensor Operations Complete!");
    console.log("═".repeat(60) + "\n");
}

main().catch(console.error);
