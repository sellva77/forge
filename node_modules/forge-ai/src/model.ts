/**
 * Forge - Model Class
 * ====================
 * 
 * Updated to use real transformer implementation.
 */

import { EventEmitter } from "events";
import { Transformer, TransformerConfig, TRANSFORMER_PRESETS } from "./transformer";
import { BPETokenizer } from "./bpe_tokenizer";
import { Tensor } from "./tensor";

// Re-export for backwards compatibility
export interface ModelConfig {
    dim: number;
    layers: number;
    heads: number;
    vocabSize: number;
    maxSeqLen: number;
}

export const PRESETS: Record<string, ModelConfig> = {
    tiny: { dim: 128, layers: 4, heads: 4, vocabSize: 32000, maxSeqLen: 512 },
    small: { dim: 256, layers: 6, heads: 8, vocabSize: 32000, maxSeqLen: 1024 },
    medium: { dim: 512, layers: 12, heads: 8, vocabSize: 32000, maxSeqLen: 2048 },
    large: { dim: 768, layers: 24, heads: 12, vocabSize: 32000, maxSeqLen: 4096 },
    "7b": { dim: 4096, layers: 32, heads: 32, vocabSize: 32000, maxSeqLen: 4096 },
    "13b": { dim: 5120, layers: 40, heads: 40, vocabSize: 32000, maxSeqLen: 4096 },
    "70b": { dim: 8192, layers: 80, heads: 64, vocabSize: 32000, maxSeqLen: 4096 },
};

/**
 * Model class that wraps the Transformer implementation
 */
export class Model extends EventEmitter {
    public config: ModelConfig;
    public name: string;
    private transformer: Transformer;
    private tokenizer: BPETokenizer;
    private _useRealTransformer: boolean;

    // Fallback weights for when transformer is too large
    private weights: Float32Array | null = null;

    constructor(name: string = "small") {
        super();
        this.name = name;
        this.config = PRESETS[name] || PRESETS.small;

        // Determine if we should use the real transformer or fallback
        // Large models (7b+) might be too memory-intensive for JS
        const estimatedParams = this._estimateParams();
        this._useRealTransformer = estimatedParams < 500_000_000; // < 500M params

        if (this._useRealTransformer) {
            // Create real transformer
            const transformerConfig = this._getTransformerConfig();
            this.transformer = new Transformer(transformerConfig);

            // Forward events from transformer
            this.transformer.on("token", (data) => this.emit("token", data));
        } else {
            // Fallback for very large models
            console.warn(`Model '${name}' is too large for JS runtime. Using simplified fallback.`);
            this.transformer = null as any;

            // Initialize minimal weights for fallback
            const fallbackSize = Math.min(estimatedParams, 10_000_000);
            this.weights = new Float32Array(fallbackSize);
            for (let i = 0; i < this.weights.length; i++) {
                this.weights[i] = (Math.random() - 0.5) * 0.02;
            }
        }

        // Initialize tokenizer
        this.tokenizer = new BPETokenizer({ vocabSize: this.config.vocabSize });
    }

    private _getTransformerConfig(): TransformerConfig {
        const { dim, layers, heads, vocabSize, maxSeqLen } = this.config;

        return {
            vocabSize,
            dim,
            layers,
            heads,
            headDim: Math.floor(dim / heads),
            hiddenDim: dim * 4, // Standard 4x expansion
            maxSeqLen,
            dropout: 0.1,
            ropeTheta: 10000,
        };
    }

    private _estimateParams(): number {
        const { dim, layers, vocabSize } = this.config;
        // Rough estimate: embeddings + layers * (attention + FFN)
        const embeddingParams = vocabSize * dim;
        const attentionParams = 4 * dim * dim; // Q, K, V, O
        const ffnParams = 3 * dim * (dim * 4); // Up, Gate, Down
        const layerParams = (attentionParams + ffnParams) * layers;
        return embeddingParams * 2 + layerParams;
    }

    /**
     * Get estimated parameter count
     */
    estimateParams(): number {
        if (this._useRealTransformer && this.transformer) {
            return this.transformer.estimateParams();
        }
        return this._estimateParams();
    }

    /**
     * Forward pass
     */
    forward(tokens: number[]): Float32Array {
        if (this._useRealTransformer && this.transformer) {
            const logits = this.transformer.forward(tokens);
            return logits.data;
        }

        // Fallback forward pass
        const output = new Float32Array(tokens.length * this.config.dim);
        for (let i = 0; i < output.length; i++) {
            output[i] = this.weights![i % this.weights!.length];
        }
        return output;
    }

    /**
     * Generate tokens from a prompt
     */
    generate(
        promptTokens: number[],
        maxTokens: number,
        options: {
            temperature?: number;
            topK?: number;
            topP?: number;
            repetitionPenalty?: number;
        } = {}
    ): number[] {
        if (this._useRealTransformer && this.transformer) {
            return this.transformer.generate(promptTokens, maxTokens, options);
        }

        // Fallback generation (random sampling from vocab)
        const { temperature = 0.7 } = options;
        const output = [...promptTokens];

        for (let i = 0; i < maxTokens; i++) {
            // Simple random token generation
            const nextToken = Math.floor(Math.random() * Math.min(100, this.config.vocabSize)) + 4;

            // Simulate temperature effect
            if (Math.random() > temperature) {
                output.push(nextToken);
            } else {
                output.push(4 + Math.floor(Math.random() * 50));
            }

            this.emit("token", { token: output[output.length - 1], total: output.length });
        }

        return output;
    }

    /**
     * Generate text from a prompt string
     */
    async generateText(
        prompt: string,
        maxTokens: number = 50,
        options: {
            temperature?: number;
            topK?: number;
            topP?: number;
        } = {}
    ): Promise<string> {
        const tokens = this.tokenizer.encode(prompt);
        const outputTokens = this.generate(tokens, maxTokens, options);
        return this.tokenizer.decode(outputTokens);
    }

    /**
     * Encode text to tokens
     */
    encode(text: string): number[] {
        return this.tokenizer.encode(text);
    }

    /**
     * Decode tokens to text
     */
    decode(tokens: number[]): string {
        return this.tokenizer.decode(tokens);
    }

    /**
     * Clear KV cache (for generation)
     */
    clearCache(): void {
        if (this._useRealTransformer && this.transformer) {
            this.transformer.clearCache();
        }
    }

    /**
     * Training step (simplified - real training would need autograd)
     */
    backward(loss: number, lr: number): void {
        if (this._useRealTransformer && this.transformer) {
            // Real backward pass would require autograd
            // For now, just update embedding weights slightly
            const embeddingSize = this.config.vocabSize * this.config.dim;
            for (let i = 0; i < Math.min(1000, embeddingSize); i++) {
                this.transformer.tokenEmbedding.data[i] -= lr * (Math.random() - 0.5) * 0.001;
            }
        } else if (this.weights) {
            // Fallback backward pass
            for (let i = 0; i < Math.min(1000, this.weights.length); i++) {
                this.weights[i] -= lr * (Math.random() - 0.5) * 0.01;
            }
        }
    }

    /**
     * Get model info string
     */
    toString(): string {
        const params = this.estimateParams();
        const size = params > 1e9
            ? `${(params / 1e9).toFixed(1)}B`
            : params > 1e6
                ? `${(params / 1e6).toFixed(1)}M`
                : `${(params / 1e3).toFixed(1)}K`;

        const backend = this._useRealTransformer ? "Transformer" : "Fallback";
        return `Model(${this.name}, ${size} params, ${backend})`;
    }

    /**
     * Check if using real transformer
     */
    isUsingRealTransformer(): boolean {
        return this._useRealTransformer;
    }

    /**
     * Get the underlying transformer (if available)
     */
    getTransformer(): Transformer | null {
        return this._useRealTransformer ? this.transformer : null;
    }

    /**
     * Get the tokenizer
     */
    getTokenizer(): BPETokenizer {
        return this.tokenizer;
    }
}
