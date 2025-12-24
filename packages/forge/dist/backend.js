"use strict";
/**
 * Forge - Backend Abstraction Layer
 * ==================================
 *
 * Automatically selects the best available backend:
 * 1. CUDA (NVIDIA GPU) - fastest
 * 2. Native Rust - fast
 * 3. TypeScript - always works
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.getBackendType = getBackendType;
exports.getBackendInfo = getBackendInfo;
exports.setBackend = setBackend;
exports.createTensor = createTensor;
exports.zeros = zeros;
exports.ones = ones;
exports.randn = randn;
exports.createTransformer = createTransformer;
exports.createTokenizer = createTokenizer;
exports.benchmark = benchmark;
exports.printBackendInfo = printBackendInfo;
const tensor_1 = require("./tensor");
const transformer_1 = require("./transformer");
const bpe_tokenizer_1 = require("./bpe_tokenizer");
// =============================================================================
// BACKEND DETECTION
// =============================================================================
let _nativeCore = null;
let _cudaCore = null;
let _detectedBackend = "javascript";
let _backendInfo = null;
/**
 * Try to load native core
 */
function tryLoadNative() {
    if (_nativeCore !== null) {
        return _nativeCore !== false;
    }
    try {
        _nativeCore = require("@forge-ai/core");
        if (_nativeCore.isNative) {
            console.log("✓ Forge: Using native Rust backend");
            return true;
        }
        // Core loaded but using JS fallback internally
        _nativeCore = false;
        return false;
    }
    catch (e) {
        // Native core not available
        _nativeCore = false;
        return false;
    }
}
/**
 * Try to load CUDA core
 */
function tryLoadCuda() {
    if (_cudaCore !== null) {
        return _cudaCore !== false;
    }
    try {
        _cudaCore = require("@forge-ai/cuda");
        console.log("✓ Forge: Using CUDA GPU backend");
        return true;
    }
    catch (e) {
        _cudaCore = false;
        return false;
    }
}
/**
 * Detect and select the best available backend
 */
function detectBackend() {
    // Priority 1: CUDA (GPU)
    if (tryLoadCuda()) {
        _detectedBackend = "cuda";
        return "cuda";
    }
    // Priority 2: Native Rust
    if (tryLoadNative()) {
        _detectedBackend = "native";
        return "native";
    }
    // Priority 3: JavaScript (always available)
    console.log("ℹ Forge: Using TypeScript backend (install @forge-ai/core for better performance)");
    _detectedBackend = "javascript";
    return "javascript";
}
// =============================================================================
// BACKEND API
// =============================================================================
/**
 * Get current backend type
 */
function getBackendType() {
    if (!_backendInfo) {
        detectBackend();
    }
    return _detectedBackend;
}
/**
 * Get detailed backend info
 */
function getBackendInfo() {
    if (_backendInfo) {
        return _backendInfo;
    }
    detectBackend();
    if (_detectedBackend === "cuda" && _cudaCore) {
        _backendInfo = {
            type: "cuda",
            name: "CUDA GPU Backend",
            version: _cudaCore.version || "1.0.0",
            platform: process.platform,
            gpu: true,
            simd: true,
        };
    }
    else if (_detectedBackend === "native" && _nativeCore) {
        const info = _nativeCore.getInfo?.() || {};
        _backendInfo = {
            type: "native",
            name: "Native Rust Backend",
            version: info.version || "1.0.0",
            platform: info.platform || process.platform,
            gpu: false,
            simd: true,
        };
    }
    else {
        _backendInfo = {
            type: "javascript",
            name: "TypeScript Backend",
            version: "1.0.0",
            platform: null,
            gpu: false,
            simd: false,
        };
    }
    return _backendInfo;
}
/**
 * Force a specific backend (for testing)
 */
function setBackend(type) {
    _detectedBackend = type;
    _backendInfo = null;
}
// =============================================================================
// TENSOR FACTORY
// =============================================================================
/**
 * Create a tensor using the best available backend
 */
function createTensor(data, shape) {
    // Currently using JS Tensor for all backends
    // Native backend integration will be added with proper Tensor interop
    return new tensor_1.Tensor(data, shape);
}
/**
 * Create zeros tensor
 */
function zeros(shape) {
    return tensor_1.Tensor.zeros(shape);
}
/**
 * Create ones tensor
 */
function ones(shape) {
    return tensor_1.Tensor.ones(shape);
}
/**
 * Create random tensor
 */
function randn(shape, mean = 0, std = 1) {
    return tensor_1.Tensor.randn(shape, mean, std);
}
// =============================================================================
// MODEL FACTORY
// =============================================================================
/**
 * Create a transformer model using the best available backend
 */
function createTransformer(config) {
    const resolvedConfig = typeof config === "string"
        ? transformer_1.TRANSFORMER_PRESETS[config] || transformer_1.TRANSFORMER_PRESETS.small
        : config;
    return new transformer_1.Transformer(resolvedConfig);
}
// =============================================================================
// TOKENIZER FACTORY
// =============================================================================
/**
 * Create a tokenizer using the best available backend
 */
function createTokenizer(vocabSize = 32000) {
    return new bpe_tokenizer_1.BPETokenizer({ vocabSize });
}
// =============================================================================
// PERFORMANCE UTILITIES
// =============================================================================
/**
 * Benchmark the current backend
 */
async function benchmark(size = 512) {
    const a = randn([size, size]);
    const b = randn([size, size]);
    // Benchmark matmul
    const startMatmul = performance.now();
    const c = a.matmul(b);
    const matmulMs = performance.now() - startMatmul;
    // Benchmark softmax
    const startSoftmax = performance.now();
    c.softmax();
    const softmaxMs = performance.now() - startSoftmax;
    // Calculate ops/sec (2 * N^3 for matmul)
    const flops = 2 * size * size * size;
    const opsPerSecond = flops / (matmulMs / 1000);
    return {
        backend: getBackendType(),
        matmulMs: Math.round(matmulMs * 100) / 100,
        softmaxMs: Math.round(softmaxMs * 100) / 100,
        opsPerSecond: Math.round(opsPerSecond),
    };
}
/**
 * Print backend info to console
 */
function printBackendInfo() {
    const info = getBackendInfo();
    console.log("\n🔥 Forge Backend Info");
    console.log("═".repeat(40));
    console.log(`  Type:     ${info.type.toUpperCase()}`);
    console.log(`  Name:     ${info.name}`);
    console.log(`  Version:  ${info.version}`);
    console.log(`  Platform: ${info.platform || "universal"}`);
    console.log(`  GPU:      ${info.gpu ? "✓ Yes" : "✗ No"}`);
    console.log(`  SIMD:     ${info.simd ? "✓ Yes" : "✗ No"}`);
    console.log("═".repeat(40) + "\n");
}
// =============================================================================
// AUTO-DETECT ON IMPORT
// =============================================================================
// Detect backend when this module is first imported
detectBackend();
exports.default = {
    getBackendType,
    getBackendInfo,
    setBackend,
    createTensor,
    createTransformer,
    createTokenizer,
    zeros,
    ones,
    randn,
    benchmark,
    printBackendInfo,
};
//# sourceMappingURL=backend.js.map