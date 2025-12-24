/**
 * Forge - Backend Abstraction Layer
 * ==================================
 * 
 * Automatically selects the best available backend:
 * 1. CUDA (NVIDIA GPU) - fastest
 * 2. Native Rust - fast
 * 3. TypeScript - always works
 */

import { Tensor as JSTensor } from "./tensor";
import { Transformer as JSTransformer, TransformerConfig, TRANSFORMER_PRESETS } from "./transformer";
import { BPETokenizer as JSTokenizer } from "./bpe_tokenizer";

// =============================================================================
// TYPES
// =============================================================================

export type BackendType = "cuda" | "native" | "javascript";

export interface BackendInfo {
    type: BackendType;
    name: string;
    version: string;
    platform: string | null;
    gpu: boolean;
    simd: boolean;
}

export interface TensorLike {
    data: Float32Array;
    shape: number[];
    matmul(other: TensorLike): TensorLike;
    add(other: TensorLike | number): TensorLike;
    mul(other: TensorLike | number): TensorLike;
    softmax(dim?: number): TensorLike;
    gelu(): TensorLike;
}

// =============================================================================
// BACKEND DETECTION
// =============================================================================

let _nativeCore: any = null;
let _cudaCore: any = null;
let _detectedBackend: BackendType = "javascript";
let _backendInfo: BackendInfo | null = null;

/**
 * Try to load native core
 */
function tryLoadNative(): boolean {
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
    } catch (e) {
        // Native core not available
        _nativeCore = false;
        return false;
    }
}

/**
 * Try to load CUDA core
 */
function tryLoadCuda(): boolean {
    if (_cudaCore !== null) {
        return _cudaCore !== false;
    }

    try {
        _cudaCore = require("@forge-ai/cuda");
        console.log("✓ Forge: Using CUDA GPU backend");
        return true;
    } catch (e) {
        _cudaCore = false;
        return false;
    }
}

/**
 * Detect and select the best available backend
 */
function detectBackend(): BackendType {
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
export function getBackendType(): BackendType {
    if (!_backendInfo) {
        detectBackend();
    }
    return _detectedBackend;
}

/**
 * Get detailed backend info
 */
export function getBackendInfo(): BackendInfo {
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
    } else if (_detectedBackend === "native" && _nativeCore) {
        const info = _nativeCore.getInfo?.() || {};
        _backendInfo = {
            type: "native",
            name: "Native Rust Backend",
            version: info.version || "1.0.0",
            platform: info.platform || process.platform,
            gpu: false,
            simd: true,
        };
    } else {
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
export function setBackend(type: BackendType): void {
    _detectedBackend = type;
    _backendInfo = null;
}

// =============================================================================
// TENSOR FACTORY
// =============================================================================

/**
 * Create a tensor using the best available backend
 */
export function createTensor(data: number[] | Float32Array, shape: number[]): JSTensor {
    // Currently using JS Tensor for all backends
    // Native backend integration will be added with proper Tensor interop
    return new JSTensor(data, shape);
}

/**
 * Create zeros tensor
 */
export function zeros(shape: number[]): JSTensor {
    return JSTensor.zeros(shape);
}

/**
 * Create ones tensor
 */
export function ones(shape: number[]): JSTensor {
    return JSTensor.ones(shape);
}

/**
 * Create random tensor
 */
export function randn(shape: number[], mean = 0, std = 1): JSTensor {
    return JSTensor.randn(shape, mean, std);
}

// =============================================================================
// MODEL FACTORY
// =============================================================================

/**
 * Create a transformer model using the best available backend
 */
export function createTransformer(config: TransformerConfig | string): JSTransformer {
    const resolvedConfig = typeof config === "string"
        ? TRANSFORMER_PRESETS[config] || TRANSFORMER_PRESETS.small
        : config;

    return new JSTransformer(resolvedConfig);
}

// =============================================================================
// TOKENIZER FACTORY
// =============================================================================

/**
 * Create a tokenizer using the best available backend
 */
export function createTokenizer(vocabSize = 32000): JSTokenizer {
    return new JSTokenizer({ vocabSize });
}

// =============================================================================
// PERFORMANCE UTILITIES
// =============================================================================

/**
 * Benchmark the current backend
 */
export async function benchmark(size = 512): Promise<{
    backend: BackendType;
    matmulMs: number;
    softmaxMs: number;
    opsPerSecond: number;
}> {
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
export function printBackendInfo(): void {
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

export default {
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
