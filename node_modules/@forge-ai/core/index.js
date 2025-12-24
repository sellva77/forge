/**
 * @forge-ai/core - Native Binding Loader
 * =======================================
 * 
 * High-performance Rust core with automatic fallback to JavaScript.
 * 
 * Features:
 * - Platform-specific native binary loading
 * - Graceful fallback to JS implementation
 * - CUDA/Metal GPU detection
 * - Performance benchmarking utilities
 */

const { existsSync } = require('fs');
const { join } = require('path');
const os = require('os');

// =============================================================================
// PLATFORM DETECTION
// =============================================================================

const PLATFORM_MAP = {
    'win32-x64-msvc': 'forge-core.win32-x64-msvc.node',
    'darwin-x64': 'forge-core.darwin-x64.node',
    'darwin-arm64': 'forge-core.darwin-arm64.node',
    'linux-x64-gnu': 'forge-core.linux-x64-gnu.node',
    'linux-arm64-gnu': 'forge-core.linux-arm64-gnu.node',
    'linux-x64-musl': 'forge-core.linux-x64-musl.node',
    'linux-arm64-musl': 'forge-core.linux-arm64-musl.node',
};

/**
 * Detect the current platform triple
 */
function getPlatformTriple() {
    const platform = process.platform;
    const arch = process.arch;

    if (platform === 'win32') {
        return arch === 'x64' ? 'win32-x64-msvc' : null;
    }

    if (platform === 'darwin') {
        return arch === 'arm64' ? 'darwin-arm64' : 'darwin-x64';
    }

    if (platform === 'linux') {
        const isMusl = (() => {
            try {
                const { execSync } = require('child_process');
                const output = execSync('ldd --version 2>&1', { encoding: 'utf-8' });
                return output.toLowerCase().includes('musl');
            } catch {
                // If ldd fails, try checking /lib
                try {
                    const files = require('fs').readdirSync('/lib');
                    return files.some(f => f.includes('musl'));
                } catch {
                    return false;
                }
            }
        })();

        return `linux-${arch}-${isMusl ? 'musl' : 'gnu'}`;
    }

    return null;
}

/**
 * Get detailed system information
 */
function getSystemInfo() {
    return {
        platform: process.platform,
        arch: process.arch,
        cpus: os.cpus().length,
        cpuModel: os.cpus()[0]?.model || 'Unknown',
        totalMemory: Math.round(os.totalmem() / (1024 * 1024 * 1024)) + ' GB',
        freeMemory: Math.round(os.freemem() / (1024 * 1024 * 1024)) + ' GB',
        nodeVersion: process.version,
    };
}

// =============================================================================
// NATIVE BINARY LOADING
// =============================================================================

/**
 * Attempt to load the native binary
 */
function loadNative() {
    const triple = getPlatformTriple();

    if (!triple) {
        return { native: null, reason: 'Unsupported platform' };
    }

    const binaryName = PLATFORM_MAP[triple];

    if (!binaryName) {
        return { native: null, reason: `No binary for ${triple}` };
    }

    // Search paths for the native binary
    const searchPaths = [
        // Local build (development)
        join(__dirname, binaryName),
        join(__dirname, `forge-core.${triple}.node`),

        // Target directory (after cargo build)
        join(__dirname, 'target', 'release', 'forge_core.node'),

        // Platform-specific npm package
        join(__dirname, '..', `@forge-ai-core-${triple}`, binaryName),

        // Node modules location
        join(__dirname, '..', '..', `@forge-ai`, `core-${triple}`, binaryName),
    ];

    for (const path of searchPaths) {
        if (existsSync(path)) {
            try {
                const native = require(path);
                return { native, reason: null, path };
            } catch (err) {
                // Continue to next path
            }
        }
    }

    // Try loading platform-specific npm package
    try {
        const native = require(`@forge-ai/core-${triple}`);
        return { native, reason: null, path: `@forge-ai/core-${triple}` };
    } catch {
        // Not installed
    }

    return { native: null, reason: 'Native binary not found' };
}

/**
 * Load the JavaScript fallback implementation
 */
function loadFallback() {
    try {
        return require('./fallback.js');
    } catch (err) {
        throw new Error(`Failed to load fallback: ${err.message}`);
    }
}

// =============================================================================
// INITIALIZATION
// =============================================================================

let bindings = null;
let usingNative = false;
let loadedFrom = 'unknown';
let loadReason = null;

// Try native first
const nativeResult = loadNative();
if (nativeResult.native) {
    bindings = nativeResult.native;
    usingNative = true;
    loadedFrom = nativeResult.path || 'native';
} else {
    loadReason = nativeResult.reason;
    bindings = loadFallback();
    usingNative = false;
    loadedFrom = 'fallback.js';
}

// =============================================================================
// EXPORTS
// =============================================================================

// Re-export all bindings
module.exports = {
    // Core classes and functions from native/fallback
    ...bindings,

    // Metadata
    isNative: usingNative,
    platform: getPlatformTriple(),
    loadedFrom,

    // Package info
    version: require('./package.json').version,

    /**
     * Check if using native bindings
     */
    checkNative() {
        return usingNative;
    },

    /**
     * Get comprehensive info about the core
     */
    getInfo() {
        const sys = getSystemInfo();
        return {
            version: this.version,
            native: usingNative,
            platform: getPlatformTriple(),
            loadedFrom,
            loadReason: usingNative ? null : loadReason,
            system: sys,
            features: {
                simd: usingNative ? (bindings.hasSIMD?.() ?? 'unknown') : false,
                cuda: usingNative ? (bindings.hasCUDA?.() ?? false) : false,
                metal: usingNative ? (bindings.hasMetal?.() ?? false) : false,
                threads: usingNative ? (bindings.numThreads?.() ?? sys.cpus) : 1,
            },
        };
    },

    /**
     * Print formatted info to console
     */
    printInfo() {
        const info = this.getInfo();
        console.log('\n╔═══════════════════════════════════════════════════════╗');
        console.log('║           @forge-ai/core - System Info                ║');
        console.log('╠═══════════════════════════════════════════════════════╣');
        console.log(`║  Version:     ${info.version.padEnd(40)}║`);
        console.log(`║  Native:      ${(info.native ? '✓ Yes' : '✗ No (fallback)').padEnd(40)}║`);
        console.log(`║  Platform:    ${(info.platform || 'unknown').padEnd(40)}║`);
        console.log(`║  Loaded from: ${info.loadedFrom.slice(0, 38).padEnd(40)}║`);
        console.log('╠═══════════════════════════════════════════════════════╣');
        console.log(`║  CPU:         ${info.system.cpuModel.slice(0, 38).padEnd(40)}║`);
        console.log(`║  Cores:       ${String(info.system.cpus).padEnd(40)}║`);
        console.log(`║  Memory:      ${info.system.totalMemory.padEnd(40)}║`);
        console.log(`║  Node:        ${info.system.nodeVersion.padEnd(40)}║`);
        console.log('╠═══════════════════════════════════════════════════════╣');
        console.log(`║  Threads:     ${String(info.features.threads).padEnd(40)}║`);
        console.log(`║  SIMD:        ${String(info.features.simd).padEnd(40)}║`);
        console.log(`║  CUDA:        ${(info.features.cuda ? '✓ Available' : '✗ Not available').padEnd(40)}║`);
        console.log('╚═══════════════════════════════════════════════════════╝\n');
    },

    /**
     * Run a simple benchmark
     */
    async benchmark(opts = {}) {
        const { size = 512, iterations = 100 } = opts;
        const results = {};

        console.log(`\n🔥 Running benchmarks (${size}x${size}, ${iterations} iterations)...\n`);

        // Tensor creation
        const startCreate = performance.now();
        for (let i = 0; i < iterations; i++) {
            bindings.randn([size, size]);
        }
        results.tensorCreate = (performance.now() - startCreate) / iterations;
        console.log(`  Tensor creation:  ${results.tensorCreate.toFixed(3)}ms`);

        // Matrix multiplication
        const A = bindings.randn([size, size]);
        const B = bindings.randn([size, size]);

        // Warmup
        A.matmul(B);

        const startMatmul = performance.now();
        for (let i = 0; i < iterations / 10; i++) {
            A.matmul(B);
        }
        results.matmul = (performance.now() - startMatmul) / (iterations / 10);
        console.log(`  Matrix multiply:  ${results.matmul.toFixed(3)}ms`);

        // Softmax
        const C = bindings.randn([1, size * size]);
        const startSoftmax = performance.now();
        for (let i = 0; i < iterations; i++) {
            C.softmax();
        }
        results.softmax = (performance.now() - startSoftmax) / iterations;
        console.log(`  Softmax:          ${results.softmax.toFixed(3)}ms`);

        // GELU
        const startGelu = performance.now();
        for (let i = 0; i < iterations; i++) {
            C.gelu();
        }
        results.gelu = (performance.now() - startGelu) / iterations;
        console.log(`  GELU:             ${results.gelu.toFixed(3)}ms`);

        console.log(`\n  Backend: ${usingNative ? 'Native Rust' : 'JavaScript Fallback'}`);
        console.log(`  Platform: ${getPlatformTriple()}\n`);

        return results;
    },
};
