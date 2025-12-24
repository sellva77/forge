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
import { Transformer as JSTransformer, TransformerConfig } from "./transformer";
import { BPETokenizer as JSTokenizer } from "./bpe_tokenizer";
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
/**
 * Get current backend type
 */
export declare function getBackendType(): BackendType;
/**
 * Get detailed backend info
 */
export declare function getBackendInfo(): BackendInfo;
/**
 * Force a specific backend (for testing)
 */
export declare function setBackend(type: BackendType): void;
/**
 * Create a tensor using the best available backend
 */
export declare function createTensor(data: number[] | Float32Array, shape: number[]): JSTensor;
/**
 * Create zeros tensor
 */
export declare function zeros(shape: number[]): JSTensor;
/**
 * Create ones tensor
 */
export declare function ones(shape: number[]): JSTensor;
/**
 * Create random tensor
 */
export declare function randn(shape: number[], mean?: number, std?: number): JSTensor;
/**
 * Create a transformer model using the best available backend
 */
export declare function createTransformer(config: TransformerConfig | string): JSTransformer;
/**
 * Create a tokenizer using the best available backend
 */
export declare function createTokenizer(vocabSize?: number): JSTokenizer;
/**
 * Benchmark the current backend
 */
export declare function benchmark(size?: number): Promise<{
    backend: BackendType;
    matmulMs: number;
    softmaxMs: number;
    opsPerSecond: number;
}>;
/**
 * Print backend info to console
 */
export declare function printBackendInfo(): void;
declare const _default: {
    getBackendType: typeof getBackendType;
    getBackendInfo: typeof getBackendInfo;
    setBackend: typeof setBackend;
    createTensor: typeof createTensor;
    createTransformer: typeof createTransformer;
    createTokenizer: typeof createTokenizer;
    zeros: typeof zeros;
    ones: typeof ones;
    randn: typeof randn;
    benchmark: typeof benchmark;
    printBackendInfo: typeof printBackendInfo;
};
export default _default;
//# sourceMappingURL=backend.d.ts.map