"use strict";
/**
 * Forge - Transformer Implementation
 * ===================================
 *
 * Real transformer model with attention, embeddings, and generation.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.Transformer = exports.TransformerBlock = exports.FeedForward = exports.Attention = exports.RotaryEmbedding = exports.RMSNorm = exports.Linear = exports.TRANSFORMER_PRESETS = void 0;
const tensor_1 = require("./tensor");
const events_1 = require("events");
exports.TRANSFORMER_PRESETS = {
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
class Linear {
    weight;
    bias;
    constructor(inFeatures, outFeatures, useBias = true) {
        // Xavier initialization
        const scale = Math.sqrt(2 / (inFeatures + outFeatures));
        this.weight = tensor_1.Tensor.randn([inFeatures, outFeatures], 0, scale);
        this.bias = useBias ? tensor_1.Tensor.zeros([outFeatures]) : null;
    }
    forward(x) {
        // x: [batch, seq, in] * weight: [in, out] -> [batch, seq, out]
        let result;
        if (x.shape.length === 2) {
            result = x.matmul(this.weight);
        }
        else if (x.shape.length === 3) {
            // Reshape for batched matmul
            const [batch, seq, dim] = x.shape;
            const flat = x.reshape([batch * seq, dim]);
            const out = flat.matmul(this.weight);
            result = out.reshape([batch, seq, this.weight.shape[1]]);
        }
        else {
            throw new Error(`Linear layer expects 2D or 3D input, got ${x.shape.length}D`);
        }
        if (this.bias) {
            // Broadcast bias addition
            const biasData = new Float32Array(result.size);
            for (let i = 0; i < result.size; i++) {
                biasData[i] = result.data[i] + this.bias.data[i % this.bias.size];
            }
            return new tensor_1.Tensor(biasData, result.shape);
        }
        return result;
    }
}
exports.Linear = Linear;
/**
 * RMSNorm (Root Mean Square Normalization)
 */
class RMSNorm {
    weight;
    eps;
    constructor(dim, eps = 1e-5) {
        this.weight = tensor_1.Tensor.ones([dim]);
        this.eps = eps;
    }
    forward(x) {
        return x.rmsNorm(this.weight, this.eps);
    }
}
exports.RMSNorm = RMSNorm;
/**
 * Rotary Position Embedding (RoPE)
 */
class RotaryEmbedding {
    dim;
    maxSeqLen;
    theta;
    cosCache;
    sinCache;
    constructor(dim, maxSeqLen, theta = 10000) {
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
    apply(x, startPos) {
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
        return new tensor_1.Tensor(result, x.shape);
    }
}
exports.RotaryEmbedding = RotaryEmbedding;
/**
 * Multi-Head Self Attention
 */
class Attention {
    config;
    wq;
    wk;
    wv;
    wo;
    rope;
    // KV Cache for generation
    cacheK = null;
    cacheV = null;
    constructor(config) {
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
    forward(x, startPos, mask) {
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
    clearCache() {
        this.cacheK = null;
        this.cacheV = null;
    }
    _concatCache(cache, current) {
        const [batch, cacheSeq, heads, headDim] = cache.shape;
        const [_, curSeq, __, ___] = current.shape;
        const newData = new Float32Array((cacheSeq + curSeq) * batch * heads * headDim);
        newData.set(cache.data, 0);
        newData.set(current.data, cache.size);
        return new tensor_1.Tensor(newData, [batch, cacheSeq + curSeq, heads, headDim]);
    }
    _transposeForAttention(x) {
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
        return new tensor_1.Tensor(result, [batch, heads, seq, headDim]);
    }
    _transposeFromAttention(x, batch, seq, heads, headDim) {
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
        return new tensor_1.Tensor(result, [batch, seq, heads * headDim]);
    }
    _transposeLastTwo(x) {
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
        return new tensor_1.Tensor(result, newShape);
    }
    _batchedMatmul(a, b) {
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
        return new tensor_1.Tensor(result, [batch, heads, m, n]);
    }
    _applyCausalMask(scores, qLen, kLen) {
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
                        }
                        else {
                            result[idx] = scores.data[idx];
                        }
                    }
                }
            }
        }
        return new tensor_1.Tensor(result, scores.shape);
    }
    _batchedSoftmax(x) {
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
        return new tensor_1.Tensor(result, x.shape);
    }
}
exports.Attention = Attention;
/**
 * Feed-Forward Network (SwiGLU variant)
 */
class FeedForward {
    w1; // Gate projection
    w2; // Down projection
    w3; // Up projection
    constructor(dim, hiddenDim) {
        this.w1 = new Linear(dim, hiddenDim, false);
        this.w2 = new Linear(hiddenDim, dim, false);
        this.w3 = new Linear(dim, hiddenDim, false);
    }
    forward(x) {
        // SwiGLU: w2(SiLU(w1(x)) * w3(x))
        const gate = this.w1.forward(x).silu();
        const up = this.w3.forward(x);
        const hidden = gate.mul(up);
        return this.w2.forward(hidden);
    }
}
exports.FeedForward = FeedForward;
/**
 * Transformer Block (Pre-norm architecture)
 */
class TransformerBlock {
    attention;
    ffn;
    attnNorm;
    ffnNorm;
    constructor(config) {
        this.attention = new Attention(config);
        this.ffn = new FeedForward(config.dim, config.hiddenDim);
        this.attnNorm = new RMSNorm(config.dim);
        this.ffnNorm = new RMSNorm(config.dim);
    }
    forward(x, startPos) {
        // Pre-norm attention + residual
        const attnInput = this.attnNorm.forward(x);
        const attnOutput = this.attention.forward(attnInput, startPos);
        const h = x.add(attnOutput);
        // Pre-norm FFN + residual
        const ffnInput = this.ffnNorm.forward(h);
        const ffnOutput = this.ffn.forward(ffnInput);
        return h.add(ffnOutput);
    }
    clearCache() {
        this.attention.clearCache();
    }
}
exports.TransformerBlock = TransformerBlock;
/**
 * Full Transformer Model
 */
class Transformer extends events_1.EventEmitter {
    config;
    tokenEmbedding;
    layers;
    norm;
    output;
    constructor(config) {
        super();
        this.config = config;
        // Token embeddings
        const embScale = Math.sqrt(1 / config.dim);
        this.tokenEmbedding = tensor_1.Tensor.randn([config.vocabSize, config.dim], 0, embScale);
        // Transformer blocks
        this.layers = [];
        for (let i = 0; i < config.layers; i++) {
            this.layers.push(new TransformerBlock(config));
        }
        // Final norm and output projection
        this.norm = new RMSNorm(config.dim);
        this.output = new Linear(config.dim, config.vocabSize, false);
    }
    forward(tokens, startPos = 0) {
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
        let h = new tensor_1.Tensor(embedData, [batch, seq, this.config.dim]);
        // Forward through layers
        for (let i = 0; i < this.layers.length; i++) {
            h = this.layers[i].forward(h, startPos);
        }
        // Final norm and logits
        h = this.norm.forward(h);
        return this.output.forward(h);
    }
    generate(tokens, maxTokens, options = {}) {
        const { temperature = 0.7, topK = 40, topP = 0.9, repetitionPenalty = 1.1, } = options;
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
            if (nextToken === 3)
                break; // EOS token
            output.push(nextToken);
            // Emit progress
            this.emit("token", { token: nextToken, total: output.length });
            // Forward with just the new token
            logits = this.forward([nextToken], output.length - 1);
        }
        return output;
    }
    clearCache() {
        for (const layer of this.layers) {
            layer.clearCache();
        }
    }
    _getLastLogits(logits) {
        const [batch, seq, vocab] = logits.shape;
        const lastLogits = new Float32Array(vocab);
        const offset = (seq - 1) * vocab;
        for (let i = 0; i < vocab; i++) {
            lastLogits[i] = logits.data[offset + i];
        }
        return lastLogits;
    }
    _applyRepetitionPenalty(logits, tokens, penalty) {
        const result = new Float32Array(logits);
        const seen = new Set(tokens);
        for (const token of seen) {
            if (token < result.length) {
                if (result[token] > 0) {
                    result[token] /= penalty;
                }
                else {
                    result[token] *= penalty;
                }
            }
        }
        return result;
    }
    _sample(logits, temperature, topK, topP) {
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
    estimateParams() {
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
    toString() {
        const params = this.estimateParams();
        const size = params > 1e9
            ? `${(params / 1e9).toFixed(1)}B`
            : params > 1e6
                ? `${(params / 1e6).toFixed(1)}M`
                : `${(params / 1e3).toFixed(1)}K`;
        return `Transformer(dim=${this.config.dim}, layers=${this.config.layers}, heads=${this.config.heads}, params=${size})`;
    }
}
exports.Transformer = Transformer;
exports.default = Transformer;
//# sourceMappingURL=transformer.js.map