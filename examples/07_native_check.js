import { forge } from "../packages/forge/dist/index.js";
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const core = require("../packages/core/index.js");

async function main() {
    console.log("\n🧪 Checking Forge Native Core Status...\n");

    const info = core.getInfo();
    core.printInfo();

    if (info.native) {
        console.log("✅ NATIVE CORE ACTIVE");
        console.log("   Performance should be optimal.");
        console.log(`   Running on: ${info.platform}`);
        console.log(`   Threads: ${info.features.threads}`);
        if (info.features.simd) console.log("   SIMD: Enabled 🚀");
        if (info.features.cuda) console.log("   CUDA: Enabled 🏎️");
    } else {
        console.log("⚠️ USING JAVASCRIPT FALLBACK");
        console.log("   Native binary not loaded.");
        if (info.loadReason) {
            console.log(`   Reason: ${info.loadReason}`);
        }
        console.log("   Performance will be slower but functional.");
    }

    console.log("\n📊 Running Benchmarks...");
    const results = await core.benchmark({ size: 512, iterations: 10 }); // Reduced iterations for quick check

    console.log("\n-------------------------------------------");
    console.log("Benchmark Results:");
    console.log(JSON.stringify(results, null, 2));
    console.log("-------------------------------------------\n");
}

main().catch(console.error);
