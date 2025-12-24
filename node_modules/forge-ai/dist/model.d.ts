/**
 * Forge - Model Class
 * ====================
 *
 * Updated to use real transformer implementation.
 */
import { EventEmitter } from "events";
import { Transformer } from "./transformer";
import { BPETokenizer } from "./bpe_tokenizer";
export interface ModelConfig {
    dim: number;
    layers: number;
    heads: number;
    vocabSize: number;
    maxSeqLen: number;
}
export declare const PRESETS: Record<string, ModelConfig>;
/**
 * Model class that wraps the Transformer implementation
 */
export declare class Model extends EventEmitter {
    config: ModelConfig;
    name: string;
    private transformer;
    private tokenizer;
    private _useRealTransformer;
    private weights;
    constructor(name?: string);
    private _getTransformerConfig;
    private _estimateParams;
    /**
     * Get estimated parameter count
     */
    estimateParams(): number;
    /**
     * Forward pass
     */
    forward(tokens: number[]): Float32Array;
    /**
     * Generate tokens from a prompt
     */
    generate(promptTokens: number[], maxTokens: number, options?: {
        temperature?: number;
        topK?: number;
        topP?: number;
        repetitionPenalty?: number;
    }): number[];
    /**
     * Generate text from a prompt string
     */
    generateText(prompt: string, maxTokens?: number, options?: {
        temperature?: number;
        topK?: number;
        topP?: number;
    }): Promise<string>;
    /**
     * Encode text to tokens
     */
    encode(text: string): number[];
    /**
     * Decode tokens to text
     */
    decode(tokens: number[]): string;
    /**
     * Clear KV cache (for generation)
     */
    clearCache(): void;
    /**
     * Training step (simplified - real training would need autograd)
     */
    backward(loss: number, lr: number): void;
    /**
     * Get model info string
     */
    toString(): string;
    /**
     * Check if using real transformer
     */
    isUsingRealTransformer(): boolean;
    /**
     * Get the underlying transformer (if available)
     */
    getTransformer(): Transformer | null;
    /**
     * Get the tokenizer
     */
    getTokenizer(): BPETokenizer;
}
//# sourceMappingURL=model.d.ts.map