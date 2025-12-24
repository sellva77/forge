/**
 * Forge - Transformer Implementation
 * ===================================
 * 
 * Real transformer model with attention, embeddings, and generation.
 */

import { Tensor } from "./tensor";
import { EventEmitter } from "events";

// =========================================================================
// CONFIGURATION
// =========================================================================

export interface TransformerConfig {
    vocabSize: number;
    dim: number;
    layers: number;
    heads: number;
    headDim: number;
    hiddenDim: number;  // FFN hidden dimension (usually 4 * dim)
    maxSeqLen: number;
    dropout: number;
    ropeTheta: number;  // RoPE base frequency
}

export const TRANSFORMER_PRESETS: Record<string, TransformerConfig> = {
    tiny: {
        vocabSize: 32000,
        dim: 128,
        layers: 4,
        heads: 4,
        headDim: 32,
        hiddenDim: 512,
        maxSeqLen: 512,
        dropout: 0.1,
        ropeTheta: 10000,
    },
    small: {
        vocabSize: 32000,
        dim: 256,
        layers: 6,
        heads: 8,
        headDim: 32,
        hiddenDim: 1024,
        maxSeqLen: 1024,
        dropout: 0.1,
        ropeTheta: 10000,
    },
    medium: {
        vocabSize: 32000,
        dim: 512,
        layers: 12,
        heads: 8,
        headDim: 64,
        hiddenDim: 2048,
        maxSeqLen: 2048,
        dropout: 0.1,
        ropeTheta: 10000,
    },
    large: {
        vocabSize: 32000,
        dim: 768,
        layers: 24,
        heads: 12,
        headDim: 64,
        hiddenDim: 3072,
        maxSeqLen: 4096,
        dropout: 0.1,
        ropeTheta: 10000,
    },
    "7b": {
        vocabSize: 32000,
        dim: 4096,
        layers: 32,
        heads: 32,
        headDim: 128,
        hiddenDim: 11008,
        maxSeqLen: 4096,
        dropout: 0.0,
        ropeTheta: 10000,
    },
    "13b": {
        vocabSize: 32000,
        dim: 5120,
        layers: 40,
        heads: 40,
        headDim: 128,
        hiddenDim: 13824,
        maxSeqLen: 4096,
        dropout: 0.0,
        ropeTheta: 10000,
    },
};

// =========================================================================
// LAYER IMPLEMENTATIONS
// =========================================================================

/**
 * Linear layer (y = xW + b)
 */
export class Linear {
    weight: Tensor;
    bias: Tensor | null;

    constructor(inFeatures: number, outFeatures: number, useBias = true) {
        // Xavier initialization
        const scale = Math.sqrt(2 / (inFeatures + outFeatures));
        this.weight = Tensor.randn([inFeatures, outFeatures], 0, scale);
        this.bias = useBias ? Tensor.zeros([outFeatures]) : null;
    }

    forward(x: Tensor): Tensor {
        // x: [batch, seq, in] * weight: [in, out] -> [batch, seq, out]
        let result: Tensor;

        if (x.shape.length === 2) {
            result = x.matmul(this.weight);
        } else if (x.shape.length === 3) {
            // Reshape for batched matmul
            const [batch, seq, dim] = x.shape;
            const flat = x.reshape([batch * seq, dim]);
            const out = flat.matmul(this.weight);
            result = out.reshape([batch, seq, this.weight.shape[1]]);
        } else {
            throw new Error(`Linear layer expects 2D or 3D input, got ${x.shape.length}D`);
        }

        if (this.bias) {
            // Broadcast bias addition
            const biasData = new Float32Array(result.size);
            for (let i = 0; i < result.size; i++) {
                biasData[i] = result.data[i] + this.bias.data[i % this.bias.size];
            }
            return new Tensor(biasData, result.shape);
        }

        return result;
    }
}

/**
 * RMSNorm (Root Mean Square Normalization)
 */
export class RMSNorm {
    weight: Tensor;
    eps: number;

    constructor(dim: number, eps = 1e-5) {
        this.weight = Tensor.ones([dim]);
        this.eps = eps;
    }

    forward(x: Tensor): Tensor {
        return x.rmsNorm(this.weight, this.eps);
    }
}

/**
 * Rotary Position Embedding (RoPE)
 */
export class RotaryEmbedding {
    dim: number;
    maxSeqLen: number;
    theta: number;
    cosCache: Float32Array;
    sinCache: Float32Array;

    constructor(dim: number, maxSeqLen: number, theta = 10000) {
        this.dim = dim;
        this.maxSeqLen = maxSeqLen;
        this.theta = theta;

        // Precompute cos/sin tables
        this.cosCache = new Float32Array(maxSeqLen * dim / 2);
        this.sinCache = new Float32Array(maxSeqLen * dim / 2);

        for (let pos = 0; pos < maxSeqLen; pos++) {
            for (let i = 0; i < dim / 2; i++) {
                const freq = 1 / Math.pow(theta, (2 * i) / dim);
                const angle = pos * freq;
                this.cosCache[pos * (dim / 2) + i] = Math.cos(angle);
                this.sinCache[pos * (dim / 2) + i] = Math.sin(angle);
            }
        }
    }

    apply(x: Tensor, startPos: number): Tensor {
        // x: [batch, seq, heads, headDim]
        const [batch, seq, heads, headDim] = x.shape;
        const result = new Float32Array(x.size);

        for (let b = 0; b < batch; b++) {
            for (let s = 0; s < seq; s++) {
                const pos = startPos + s;
                for (let h = 0; h < heads; h++) {
                    for (let i = 0; i < headDim / 2; i++) {
                        const idx = b * seq * heads * headDim + s * heads * headDim + h * headDim;
                        const x0 = x.data[idx + i];
                        const x1 = x.data[idx + headDim / 2 + i];

                        const cos = this.cosCache[pos * (headDim / 2) + i];
                        const sin = this.sinCache[pos * (headDim / 2) + i];

                        result[idx + i] = x0 * cos - x1 * sin;
                        result[idx + headDim / 2 + i] = x0 * sin + x1 * cos;
                    }
                }
            }
        }

        return new Tensor(result, x.shape);
    }
}

/**
 * Multi-Head Self Attention
 */
export class Attention {
    config: TransformerConfig;
    wq: Linear;
    wk: Linear;
    wv: Linear;
    wo: Linear;
    rope: RotaryEmbedding;

    // KV Cache for generation
    cacheK: Tensor | null = null;
    cacheV: Tensor | null = null;

    constructor(config: TransformerConfig) {
        this.config = config;
        const dim = config.dim;
        const headDim = config.headDim;
        const numHeads = config.heads;

        this.wq = new Linear(dim, numHeads * headDim, false);
        this.wk = new Linear(dim, numHeads * headDim, false);
        this.wv = new Linear(dim, numHeads * headDim, false);
        this.wo = new Linear(numHeads * headDim, dim, false);

        this.rope = new RotaryEmbedding(headDim, config.maxSeqLen, config.ropeTheta);
    }

    forward(x: Tensor, startPos: number, mask?: Tensor): Tensor {
        const [batch, seq, dim] = x.shape;
        const { heads, headDim } = this.config;

        // Project Q, K, V
        let q = this.wq.forward(x).reshape([batch, seq, heads, headDim]);
        let k = this.wk.forward(x).reshape([batch, seq, heads, headDim]);
        let v = this.wv.forward(x).reshape([batch, seq, heads, headDim]);

        // Apply RoPE
        q = this.rope.apply(q, startPos);
        k = this.rope.apply(k, startPos);

        // Update KV cache for generation
        if (this.cacheK && this.cacheV && startPos > 0) {
            // Concatenate with cache
            k = this._concatCache(this.cacheK, k);
            v = this._concatCache(this.cacheV, v);
        }
        this.cacheK = k;
        this.cacheV = v;

        // Reshape for batched attention: [batch, heads, seq, headDim]
        const qT = this._transposeForAttention(q);
        const kT = this._transposeForAttention(k);
        const vT = this._transposeForAttention(v);

        // Compute attention scores: Q @ K^T / sqrt(d)
        const scale = 1 / Math.sqrt(headDim);
        const scores = this._batchedMatmul(qT, this._transposeLastTwo(kT)).mul(scale);

        // Apply causal mask
        const maskedScores = this._applyCausalMask(scores, seq, k.shape[1]);

        // Softmax
        const attnWeights = this._batchedSoftmax(maskedScores);

        // Attention output: weights @ V
        const attnOutput = this._batchedMatmul(attnWeights, vT);

        // Reshape back: [batch, seq, heads * headDim]
        const output = this._transposeFromAttention(attnOutput, batch, seq, heads, headDim);

        // Output projection
        return this.wo.forward(output);
    }

    clearCache(): void {
        this.cacheK = null;
        this.cacheV = null;
    }

    private _concatCache(cache: Tensor, current: Tensor): Tensor {
        const [batch, cacheSeq, heads, headDim] = cache.shape;
        const [_, curSeq, __, ___] = current.shape;

        const newData = new Float32Array((cacheSeq + curSeq) * batch * heads * headDim);
        newData.set(cache.data, 0);
        newData.set(current.data, cache.size);

        return new Tensor(newData, [batch, cacheSeq + curSeq, heads, headDim]);
    }

    private _transposeForAttention(x: Tensor): Tensor {
        // [batch, seq, heads, headDim] -> [batch, heads, seq, headDim]
        const [batch, seq, heads, headDim] = x.shape;
        const result = new Float32Array(x.size);

        for (let b = 0; b < batch; b++) {
            for (let s = 0; s < seq; s++) {
                for (let h = 0; h < heads; h++) {
                    for (let d = 0; d < headDim; d++) {
                        const srcIdx = b * seq * heads * headDim + s * heads * headDim + h * headDim + d;
                        const dstIdx = b * heads * seq * headDim + h * seq * headDim + s * headDim + d;
                        result[dstIdx] = x.data[srcIdx];
                    }
                }
            }
        }

        return new Tensor(result, [batch, heads, seq, headDim]);
    }

    private _transposeFromAttention(x: Tensor, batch: number, seq: number, heads: number, headDim: number): Tensor {
        // [batch, heads, seq, headDim] -> [batch, seq, heads * headDim]
        const result = new Float32Array(batch * seq * heads * headDim);

        for (let b = 0; b < batch; b++) {
            for (let h = 0; h < heads; h++) {
                for (let s = 0; s < seq; s++) {
                    for (let d = 0; d < headDim; d++) {
                        const srcIdx = b * heads * seq * headDim + h * seq * headDim + s * headDim + d;
                        const dstIdx = b * seq * heads * headDim + s * heads * headDim + h * headDim + d;
                        result[dstIdx] = x.data[srcIdx];
                    }
                }
            }
        }

        return new Tensor(result, [batch, seq, heads * headDim]);
    }

    private _transposeLastTwo(x: Tensor): Tensor {
        // [..., m, n] -> [..., n, m]
        const shape = x.shape;
        const m = shape[shape.length - 2];
        const n = shape[shape.length - 1];
        const batchSize = x.size / (m * n);

        const result = new Float32Array(x.size);

        for (let b = 0; b < batchSize; b++) {
            for (let i = 0; i < m; i++) {
                for (let j = 0; j < n; j++) {
                    result[b * n * m + j * m + i] = x.data[b * m * n + i * n + j];
                }
            }
        }

        const newShape = [...shape];
        newShape[shape.length - 2] = n;
        newShape[shape.length - 1] = m;

        return new Tensor(result, newShape);
    }

    private _batchedMatmul(a: Tensor, b: Tensor): Tensor {
        // a: [batch, heads, m, k], b: [batch, heads, k, n] -> [batch, heads, m, n]
        const [batch, heads, m, k] = a.shape;
        const n = b.shape[3];

        const result = new Float32Array(batch * heads * m * n);

        for (let i = 0; i < batch * heads; i++) {
            const aOffset = i * m * k;
            const bOffset = i * k * n;
            const cOffset = i * m * n;

            for (let mi = 0; mi < m; mi++) {
                for (let ni = 0; ni < n; ni++) {
                    let sum = 0;
                    for (let ki = 0; ki < k; ki++) {
                        sum += a.data[aOffset + mi * k + ki] * b.data[bOffset + ki * n + ni];
                    }
                    result[cOffset + mi * n + ni] = sum;
                }
            }
        }

        return new Tensor(result, [batch, heads, m, n]);
    }

    private _applyCausalMask(scores: Tensor, qLen: number, kLen: number): Tensor {
        // Apply causal mask: positions can only attend to previous positions
        const [batch, heads, _, __] = scores.shape;
        const result = new Float32Array(scores.size);

        for (let b = 0; b < batch; b++) {
            for (let h = 0; h < heads; h++) {
                for (let i = 0; i < qLen; i++) {
                    for (let j = 0; j < kLen; j++) {
                        const idx = b * heads * qLen * kLen + h * qLen * kLen + i * kLen + j;
                        // For incremental decoding, allow attending to all cached + current
                        if (j > (kLen - qLen) + i) {
                            result[idx] = -Infinity;
                        } else {
                            result[idx] = scores.data[idx];
                        }
                    }
                }
            }
        }

        return new Tensor(result, scores.shape);
    }

    private _batchedSoftmax(x: Tensor): Tensor {
        const [batch, heads, m, n] = x.shape;
        const result = new Float32Array(x.size);

        for (let b = 0; b < batch; b++) {
            for (let h = 0; h < heads; h++) {
                for (let i = 0; i < m; i++) {
                    const offset = b * heads * m * n + h * m * n + i * n;

                    // Find max
                    let max = -Infinity;
                    for (let j = 0; j < n; j++) {
                        max = Math.max(max, x.data[offset + j]);
                    }

                    // Exp and sum
                    let sum = 0;
                    for (let j = 0; j < n; j++) {
                        const exp = Math.exp(x.data[offset + j] - max);
                        result[offset + j] = exp;
                        sum += exp;
                    }

                    // Normalize
                    for (let j = 0; j < n; j++) {
                        result[offset + j] /= sum;
                    }
                }
            }
        }

        return new Tensor(result, x.shape);
    }
}

/**
 * Feed-Forward Network (SwiGLU variant)
 */
export class FeedForward {
    w1: Linear;  // Gate projection
    w2: Linear;  // Down projection
    w3: Linear;  // Up projection

    constructor(dim: number, hiddenDim: number) {
        this.w1 = new Linear(dim, hiddenDim, false);
        this.w2 = new Linear(hiddenDim, dim, false);
        this.w3 = new Linear(dim, hiddenDim, false);
    }

    forward(x: Tensor): Tensor {
        // SwiGLU: w2(SiLU(w1(x)) * w3(x))
        const gate = this.w1.forward(x).silu();
        const up = this.w3.forward(x);
        const hidden = gate.mul(up);
        return this.w2.forward(hidden);
    }
}

/**
 * Transformer Block (Pre-norm architecture)
 */
export class TransformerBlock {
    attention: Attention;
    ffn: FeedForward;
    attnNorm: RMSNorm;
    ffnNorm: RMSNorm;

    constructor(config: TransformerConfig) {
        this.attention = new Attention(config);
        this.ffn = new FeedForward(config.dim, config.hiddenDim);
        this.attnNorm = new RMSNorm(config.dim);
        this.ffnNorm = new RMSNorm(config.dim);
    }

    forward(x: Tensor, startPos: number): Tensor {
        // Pre-norm attention + residual
        const attnInput = this.attnNorm.forward(x);
        const attnOutput = this.attention.forward(attnInput, startPos);
        const h = x.add(attnOutput);

        // Pre-norm FFN + residual
        const ffnInput = this.ffnNorm.forward(h);
        const ffnOutput = this.ffn.forward(ffnInput);
        return h.add(ffnOutput);
    }

    clearCache(): void {
        this.attention.clearCache();
    }
}

/**
 * Full Transformer Model
 */
export class Transformer extends EventEmitter {
    config: TransformerConfig;
    tokenEmbedding: Tensor;
    layers: TransformerBlock[];
    norm: RMSNorm;
    output: Linear;

    constructor(config: TransformerConfig) {
        super();
        this.config = config;

        // Token embeddings
        const embScale = Math.sqrt(1 / config.dim);
        this.tokenEmbedding = Tensor.randn([config.vocabSize, config.dim], 0, embScale);

        // Transformer blocks
        this.layers = [];
        for (let i = 0; i < config.layers; i++) {
            this.layers.push(new TransformerBlock(config));
        }

        // Final norm and output projection
        this.norm = new RMSNorm(config.dim);
        this.output = new Linear(config.dim, config.vocabSize, false);
    }

    forward(tokens: number[], startPos = 0): Tensor {
        const batch = 1;
        const seq = tokens.length;

        // Get embeddings
        const embedData = new Float32Array(seq * this.config.dim);
        for (let i = 0; i < seq; i++) {
            const tokenId = Math.min(tokens[i], this.config.vocabSize - 1);
            for (let j = 0; j < this.config.dim; j++) {
                embedData[i * this.config.dim + j] = this.tokenEmbedding.data[tokenId * this.config.dim + j];
            }
        }
        let h = new Tensor(embedData, [batch, seq, this.config.dim]);

        // Forward through layers
        for (let i = 0; i < this.layers.length; i++) {
            h = this.layers[i].forward(h, startPos);
        }

        // Final norm and logits
        h = this.norm.forward(h);
        return this.output.forward(h);
    }

    generate(
        tokens: number[],
        maxTokens: number,
        options: {
            temperature?: number;
            topK?: number;
            topP?: number;
            repetitionPenalty?: number;
        } = {}
    ): number[] {
        const {
            temperature = 0.7,
            topK = 40,
            topP = 0.9,
            repetitionPenalty = 1.1,
        } = options;

        const output = [...tokens];
        this.clearCache();

        // Initial forward pass with all tokens
        let logits = this.forward(tokens, 0);

        for (let i = 0; i < maxTokens; i++) {
            // Get last token logits
            const lastLogits = this._getLastLogits(logits);

            // Apply repetition penalty
            const penalizedLogits = this._applyRepetitionPenalty(lastLogits, output, repetitionPenalty);

            // Sample next token
            const nextToken = this._sample(penalizedLogits, temperature, topK, topP);

            // Check for EOS
            if (nextToken === 3) break; // EOS token

            output.push(nextToken);

            // Emit progress
            this.emit("token", { token: nextToken, total: output.length });

            // Forward with just the new token
            logits = this.forward([nextToken], output.length - 1);
        }

        return output;
    }

    clearCache(): void {
        for (const layer of this.layers) {
            layer.clearCache();
        }
    }

    private _getLastLogits(logits: Tensor): Float32Array {
        const [batch, seq, vocab] = logits.shape;
        const lastLogits = new Float32Array(vocab);
        const offset = (seq - 1) * vocab;

        for (let i = 0; i < vocab; i++) {
            lastLogits[i] = logits.data[offset + i];
        }

        return lastLogits;
    }

    private _applyRepetitionPenalty(logits: Float32Array, tokens: number[], penalty: number): Float32Array {
        const result = new Float32Array(logits);
        const seen = new Set(tokens);

        for (const token of seen) {
            if (token < result.length) {
                if (result[token] > 0) {
                    result[token] /= penalty;
                } else {
                    result[token] *= penalty;
                }
            }
        }

        return result;
    }

    private _sample(logits: Float32Array, temperature: number, topK: number, topP: number): number {
        // Apply temperature
        const scaledLogits = new Float32Array(logits.length);
        for (let i = 0; i < logits.length; i++) {
            scaledLogits[i] = logits[i] / temperature;
        }

        // Top-K filtering
        const indices = Array.from({ length: logits.length }, (_, i) => i);
        indices.sort((a, b) => scaledLogits[b] - scaledLogits[a]);

        const topKIndices = indices.slice(0, topK);
        const topKLogits = topKIndices.map(i => scaledLogits[i]);

        // Softmax
        const max = Math.max(...topKLogits);
        const exp = topKLogits.map(x => Math.exp(x - max));
        const sum = exp.reduce((a, b) => a + b, 0);
        const probs = exp.map(x => x / sum);

        // Top-P (nucleus) filtering
        let cumSum = 0;
        let cutoffIdx = probs.length;
        for (let i = 0; i < probs.length; i++) {
            cumSum += probs[i];
            if (cumSum > topP) {
                cutoffIdx = i + 1;
                break;
            }
        }

        // Renormalize
        const nucleusProbs = probs.slice(0, cutoffIdx);
        const nucleusSum = nucleusProbs.reduce((a, b) => a + b, 0);
        for (let i = 0; i < nucleusProbs.length; i++) {
            nucleusProbs[i] /= nucleusSum;
        }

        // Sample
        const r = Math.random();
        let cumulative = 0;
        for (let i = 0; i < nucleusProbs.length; i++) {
            cumulative += nucleusProbs[i];
            if (r < cumulative) {
                return topKIndices[i];
            }
        }

        return topKIndices[0];
    }

    estimateParams(): number {
        const { vocabSize, dim, layers, hiddenDim, heads, headDim } = this.config;

        // Embeddings
        const embedParams = vocabSize * dim;

        // Each layer: attention (4 * dim * heads * headDim) + FFN (3 * dim * hiddenDim) + norms
        const attnParams = 4 * dim * heads * headDim;
        const ffnParams = 3 * dim * hiddenDim;
        const normParams = 2 * dim;
        const layerParams = (attnParams + ffnParams + normParams) * layers;

        // Output
        const outputParams = dim * vocabSize + dim; // + final norm

        return embedParams + layerParams + outputParams;
    }

    toString(): string {
        const params = this.estimateParams();
        const size = params > 1e9
            ? `${(params / 1e9).toFixed(1)}B`
            : params > 1e6
                ? `${(params / 1e6).toFixed(1)}M`
                : `${(params / 1e3).toFixed(1)}K`;

        return `Transformer(dim=${this.config.dim}, layers=${this.config.layers}, heads=${this.config.heads}, params=${size})`;
    }
}

export default Transformer;
