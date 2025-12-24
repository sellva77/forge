/**
 * Forge - Transformer Implementation
 * ===================================
 *
 * Real transformer model with attention, embeddings, and generation.
 */
import { Tensor } from "./tensor";
import { EventEmitter } from "events";
export interface TransformerConfig {
    vocabSize: number;
    dim: number;
    layers: number;
    heads: number;
    headDim: number;
    hiddenDim: number;
    maxSeqLen: number;
    dropout: number;
    ropeTheta: number;
}
export declare const TRANSFORMER_PRESETS: Record<string, TransformerConfig>;
/**
 * Linear layer (y = xW + b)
 */
export declare class Linear {
    weight: Tensor;
    bias: Tensor | null;
    constructor(inFeatures: number, outFeatures: number, useBias?: boolean);
    forward(x: Tensor): Tensor;
}
/**
 * RMSNorm (Root Mean Square Normalization)
 */
export declare class RMSNorm {
    weight: Tensor;
    eps: number;
    constructor(dim: number, eps?: number);
    forward(x: Tensor): Tensor;
}
/**
 * Rotary Position Embedding (RoPE)
 */
export declare class RotaryEmbedding {
    dim: number;
    maxSeqLen: number;
    theta: number;
    cosCache: Float32Array;
    sinCache: Float32Array;
    constructor(dim: number, maxSeqLen: number, theta?: number);
    apply(x: Tensor, startPos: number): Tensor;
}
/**
 * Multi-Head Self Attention
 */
export declare class Attention {
    config: TransformerConfig;
    wq: Linear;
    wk: Linear;
    wv: Linear;
    wo: Linear;
    rope: RotaryEmbedding;
    cacheK: Tensor | null;
    cacheV: Tensor | null;
    constructor(config: TransformerConfig);
    forward(x: Tensor, startPos: number, mask?: Tensor): Tensor;
    clearCache(): void;
    private _concatCache;
    private _transposeForAttention;
    private _transposeFromAttention;
    private _transposeLastTwo;
    private _batchedMatmul;
    private _applyCausalMask;
    private _batchedSoftmax;
}
/**
 * Feed-Forward Network (SwiGLU variant)
 */
export declare class FeedForward {
    w1: Linear;
    w2: Linear;
    w3: Linear;
    constructor(dim: number, hiddenDim: number);
    forward(x: Tensor): Tensor;
}
/**
 * Transformer Block (Pre-norm architecture)
 */
export declare class TransformerBlock {
    attention: Attention;
    ffn: FeedForward;
    attnNorm: RMSNorm;
    ffnNorm: RMSNorm;
    constructor(config: TransformerConfig);
    forward(x: Tensor, startPos: number): Tensor;
    clearCache(): void;
}
/**
 * Full Transformer Model
 */
export declare class Transformer extends EventEmitter {
    config: TransformerConfig;
    tokenEmbedding: Tensor;
    layers: TransformerBlock[];
    norm: RMSNorm;
    output: Linear;
    constructor(config: TransformerConfig);
    forward(tokens: number[], startPos?: number): Tensor;
    generate(tokens: number[], maxTokens: number, options?: {
        temperature?: number;
        topK?: number;
        topP?: number;
        repetitionPenalty?: number;
    }): number[];
    clearCache(): void;
    private _getLastLogits;
    private _applyRepetitionPenalty;
    private _sample;
    estimateParams(): number;
    toString(): string;
}
export default Transformer;
//# sourceMappingURL=transformer.d.ts.map