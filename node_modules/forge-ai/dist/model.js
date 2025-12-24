"use strict";
/**
 * Forge - Model Class
 * ====================
 *
 * Updated to use real transformer implementation.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.Model = exports.PRESETS = void 0;
const events_1 = require("events");
const transformer_1 = require("./transformer");
const bpe_tokenizer_1 = require("./bpe_tokenizer");
exports.PRESETS = {
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
class Model extends events_1.EventEmitter {
    config;
    name;
    transformer;
    tokenizer;
    _useRealTransformer;
    // Fallback weights for when transformer is too large
    weights = null;
    constructor(name = "small") {
        super();
        this.name = name;
        this.config = exports.PRESETS[name] || exports.PRESETS.small;
        // Determine if we should use the real transformer or fallback
        // Large models (7b+) might be too memory-intensive for JS
        const estimatedParams = this._estimateParams();
        this._useRealTransformer = estimatedParams < 500_000_000; // < 500M params
        if (this._useRealTransformer) {
            // Create real transformer
            const transformerConfig = this._getTransformerConfig();
            this.transformer = new transformer_1.Transformer(transformerConfig);
            // Forward events from transformer
            this.transformer.on("token", (data) => this.emit("token", data));
        }
        else {
            // Fallback for very large models
            console.warn(`Model '${name}' is too large for JS runtime. Using simplified fallback.`);
            this.transformer = null;
            // Initialize minimal weights for fallback
            const fallbackSize = Math.min(estimatedParams, 10_000_000);
            this.weights = new Float32Array(fallbackSize);
            for (let i = 0; i < this.weights.length; i++) {
                this.weights[i] = (Math.random() - 0.5) * 0.02;
            }
        }
        // Initialize tokenizer
        this.tokenizer = new bpe_tokenizer_1.BPETokenizer({ vocabSize: this.config.vocabSize });
    }
    _getTransformerConfig() {
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
    _estimateParams() {
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
    estimateParams() {
        if (this._useRealTransformer && this.transformer) {
            return this.transformer.estimateParams();
        }
        return this._estimateParams();
    }
    /**
     * Forward pass
     */
    forward(tokens) {
        if (this._useRealTransformer && this.transformer) {
            const logits = this.transformer.forward(tokens);
            return logits.data;
        }
        // Fallback forward pass
        const output = new Float32Array(tokens.length * this.config.dim);
        for (let i = 0; i < output.length; i++) {
            output[i] = this.weights[i % this.weights.length];
        }
        return output;
    }
    /**
     * Generate tokens from a prompt
     */
    generate(promptTokens, maxTokens, options = {}) {
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
            }
            else {
                output.push(4 + Math.floor(Math.random() * 50));
            }
            this.emit("token", { token: output[output.length - 1], total: output.length });
        }
        return output;
    }
    /**
     * Generate text from a prompt string
     */
    async generateText(prompt, maxTokens = 50, options = {}) {
        const tokens = this.tokenizer.encode(prompt);
        const outputTokens = this.generate(tokens, maxTokens, options);
        return this.tokenizer.decode(outputTokens);
    }
    /**
     * Encode text to tokens
     */
    encode(text) {
        return this.tokenizer.encode(text);
    }
    /**
     * Decode tokens to text
     */
    decode(tokens) {
        return this.tokenizer.decode(tokens);
    }
    /**
     * Clear KV cache (for generation)
     */
    clearCache() {
        if (this._useRealTransformer && this.transformer) {
            this.transformer.clearCache();
        }
    }
    /**
     * Training step (simplified - real training would need autograd)
     */
    backward(loss, lr) {
        if (this._useRealTransformer && this.transformer) {
            // Real backward pass would require autograd
            // For now, just update embedding weights slightly
            const embeddingSize = this.config.vocabSize * this.config.dim;
            for (let i = 0; i < Math.min(1000, embeddingSize); i++) {
                this.transformer.tokenEmbedding.data[i] -= lr * (Math.random() - 0.5) * 0.001;
            }
        }
        else if (this.weights) {
            // Fallback backward pass
            for (let i = 0; i < Math.min(1000, this.weights.length); i++) {
                this.weights[i] -= lr * (Math.random() - 0.5) * 0.01;
            }
        }
    }
    /**
     * Get model info string
     */
    toString() {
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
    isUsingRealTransformer() {
        return this._useRealTransformer;
    }
    /**
     * Get the underlying transformer (if available)
     */
    getTransformer() {
        return this._useRealTransformer ? this.transformer : null;
    }
    /**
     * Get the tokenizer
     */
    getTokenizer() {
        return this.tokenizer;
    }
}
exports.Model = Model;
//# sourceMappingURL=model.js.map