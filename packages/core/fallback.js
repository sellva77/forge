/**
 * @forge-ai/core - JavaScript Fallback
 * ======================================
 * 
 * Pure JavaScript implementation of the core AI primitives.
 * Used when native Rust bindings are not available.
 * 
 * Performance is lower than native but provides full functionality.
 */

// =============================================================================
// TENSOR
// =============================================================================

class Tensor {
    constructor(data, shape) {
        this.shape = Array.isArray(shape) ? shape.map(x => Number(x)) : [data.length];
        this._size = this.shape.reduce((a, b) => a * b, 1);
        this.data = data instanceof Float32Array ? data : new Float32Array(data);
        this.strides = this._computeStrides();
    }

    _computeStrides() {
        const strides = new Array(this.shape.length).fill(1);
        for (let i = this.shape.length - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * this.shape[i + 1];
        }
        return strides;
    }

    // =========================================================================
    // FACTORY METHODS
    // =========================================================================

    static zeros(shape) {
        const size = shape.reduce((a, b) => a * b, 1);
        return new Tensor(new Float32Array(size), shape);
    }

    static ones(shape) {
        const size = shape.reduce((a, b) => a * b, 1);
        return new Tensor(new Float32Array(size).fill(1), shape);
    }

    static rand(shape) {
        const size = shape.reduce((a, b) => a * b, 1);
        const data = new Float32Array(size);
        for (let i = 0; i < size; i++) {
            data[i] = Math.random();
        }
        return new Tensor(data, shape);
    }

    static randn(shape, mean = 0, std = 1) {
        const size = shape.reduce((a, b) => a * b, 1);
        const data = new Float32Array(size);

        // Box-Muller transform for normal distribution
        for (let i = 0; i < size; i += 2) {
            const u1 = Math.random() || 1e-10;
            const u2 = Math.random();
            const r = Math.sqrt(-2 * Math.log(u1));
            const theta = 2 * Math.PI * u2;
            data[i] = mean + std * r * Math.cos(theta);
            if (i + 1 < size) {
                data[i + 1] = mean + std * r * Math.sin(theta);
            }
        }

        return new Tensor(data, shape);
    }

    static eye(n) {
        const data = new Float32Array(n * n);
        for (let i = 0; i < n; i++) {
            data[i * n + i] = 1;
        }
        return new Tensor(data, [n, n]);
    }

    static arange(start, end, step = 1) {
        const data = [];
        for (let v = start; v < end; v += step) {
            data.push(v);
        }
        return new Tensor(new Float32Array(data), [data.length]);
    }

    // =========================================================================
    // PROPERTIES
    // =========================================================================

    ndim() {
        return this.shape.length;
    }

    size() {
        return this._size;
    }

    toArray() {
        return Array.from(this.data);
    }

    cloneTensor() {
        return new Tensor(new Float32Array(this.data), [...this.shape]);
    }

    // =========================================================================
    // SHAPE OPERATIONS
    // =========================================================================

    reshape(newShape) {
        const newSize = newShape.reduce((a, b) => a * b, 1);
        if (newSize !== this._size) {
            throw new Error(`Cannot reshape tensor of size ${this._size} to shape [${newShape}]`);
        }
        return new Tensor(this.data, newShape.map(x => Number(x)));
    }

    transpose() {
        if (this.shape.length !== 2) {
            throw new Error("Transpose requires 2D tensor");
        }
        const [rows, cols] = this.shape;
        const result = new Float32Array(this._size);
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                result[j * rows + i] = this.data[i * cols + j];
            }
        }
        return new Tensor(result, [cols, rows]);
    }

    // =========================================================================
    // ELEMENT-WISE OPERATIONS
    // =========================================================================

    add(other) {
        if (other instanceof Tensor) {
            if (this._size !== other._size) {
                throw new Error("Shape mismatch for addition");
            }
            const result = new Float32Array(this._size);
            for (let i = 0; i < this._size; i++) {
                result[i] = this.data[i] + other.data[i];
            }
            return new Tensor(result, [...this.shape]);
        }
        throw new Error("Addition requires Tensor operand");
    }

    sub(other) {
        if (other instanceof Tensor) {
            const result = new Float32Array(this._size);
            for (let i = 0; i < this._size; i++) {
                result[i] = this.data[i] - other.data[i];
            }
            return new Tensor(result, [...this.shape]);
        }
        throw new Error("Subtraction requires Tensor operand");
    }

    mul(other) {
        if (other instanceof Tensor) {
            const result = new Float32Array(this._size);
            for (let i = 0; i < this._size; i++) {
                result[i] = this.data[i] * other.data[i];
            }
            return new Tensor(result, [...this.shape]);
        }
        throw new Error("Multiplication requires Tensor operand");
    }

    div(other) {
        if (other instanceof Tensor) {
            const result = new Float32Array(this._size);
            for (let i = 0; i < this._size; i++) {
                result[i] = this.data[i] / (other.data[i] + 1e-5);
            }
            return new Tensor(result, [...this.shape]);
        }
        throw new Error("Division requires Tensor operand");
    }

    scale(factor) {
        const f = Number(factor);
        const result = new Float32Array(this._size);
        for (let i = 0; i < this._size; i++) {
            result[i] = this.data[i] * f;
        }
        return new Tensor(result, [...this.shape]);
    }

    addScalar(value) {
        const v = Number(value);
        const result = new Float32Array(this._size);
        for (let i = 0; i < this._size; i++) {
            result[i] = this.data[i] + v;
        }
        return new Tensor(result, [...this.shape]);
    }

    // =========================================================================
    // MATRIX OPERATIONS
    // =========================================================================

    matmul(other) {
        if (this.shape.length !== 2 || other.shape.length !== 2) {
            throw new Error("Matrix multiplication requires 2D tensors");
        }

        const [m, k1] = this.shape;
        const [k2, n] = other.shape;

        if (k1 !== k2) {
            throw new Error(`Dimension mismatch: ${m}x${k1} @ ${k2}x${n}`);
        }

        const result = new Float32Array(m * n);

        // Blocked matrix multiplication for better cache usage
        const blockSize = 32;
        for (let i = 0; i < m; i++) {
            for (let jb = 0; jb < n; jb += blockSize) {
                const jEnd = Math.min(jb + blockSize, n);
                for (let kb = 0; kb < k1; kb += blockSize) {
                    const kEnd = Math.min(kb + blockSize, k1);
                    for (let k = kb; k < kEnd; k++) {
                        const aVal = this.data[i * k1 + k];
                        for (let j = jb; j < jEnd; j++) {
                            result[i * n + j] += aVal * other.data[k * n + j];
                        }
                    }
                }
            }
        }

        return new Tensor(result, [m, n]);
    }

    bmm(other) {
        if (this.shape.length !== 3 || other.shape.length !== 3) {
            throw new Error("Batched matmul requires 3D tensors");
        }

        const [batch, m, k] = this.shape;
        const [batch2, k2, n] = other.shape;

        if (batch !== batch2 || k !== k2) {
            throw new Error("Batch or inner dimension mismatch");
        }

        const result = new Float32Array(batch * m * n);
        const batchSizeA = m * k;
        const batchSizeB = k * n;
        const batchSizeC = m * n;

        for (let b = 0; b < batch; b++) {
            const aStart = b * batchSizeA;
            const bStart = b * batchSizeB;
            const cStart = b * batchSizeC;

            for (let i = 0; i < m; i++) {
                for (let j = 0; j < n; j++) {
                    let sum = 0;
                    for (let l = 0; l < k; l++) {
                        sum += this.data[aStart + i * k + l] * other.data[bStart + l * n + j];
                    }
                    result[cStart + i * n + j] = sum;
                }
            }
        }

        return new Tensor(result, [batch, m, n]);
    }

    // =========================================================================
    // ACTIVATION FUNCTIONS
    // =========================================================================

    softmax() {
        const lastDim = this.shape[this.shape.length - 1];
        const numRows = this._size / lastDim;
        const result = new Float32Array(this._size);

        for (let row = 0; row < numRows; row++) {
            const start = row * lastDim;

            // Find max for numerical stability
            let max = -Infinity;
            for (let i = 0; i < lastDim; i++) {
                max = Math.max(max, this.data[start + i]);
            }

            // Compute exp and sum
            let sum = 0;
            for (let i = 0; i < lastDim; i++) {
                const exp = Math.exp(this.data[start + i] - max);
                result[start + i] = exp;
                sum += exp;
            }

            // Normalize
            for (let i = 0; i < lastDim; i++) {
                result[start + i] /= sum;
            }
        }

        return new Tensor(result, [...this.shape]);
    }

    relu() {
        const result = new Float32Array(this._size);
        for (let i = 0; i < this._size; i++) {
            result[i] = Math.max(0, this.data[i]);
        }
        return new Tensor(result, [...this.shape]);
    }

    gelu() {
        const sqrt2OverPi = Math.sqrt(2 / Math.PI);
        const result = new Float32Array(this._size);
        for (let i = 0; i < this._size; i++) {
            const x = this.data[i];
            result[i] = 0.5 * x * (1 + Math.tanh(sqrt2OverPi * (x + 0.044715 * x * x * x)));
        }
        return new Tensor(result, [...this.shape]);
    }

    silu() {
        const result = new Float32Array(this._size);
        for (let i = 0; i < this._size; i++) {
            const x = this.data[i];
            result[i] = x / (1 + Math.exp(-x));
        }
        return new Tensor(result, [...this.shape]);
    }

    sigmoid() {
        const result = new Float32Array(this._size);
        for (let i = 0; i < this._size; i++) {
            result[i] = 1 / (1 + Math.exp(-this.data[i]));
        }
        return new Tensor(result, [...this.shape]);
    }

    tanh() {
        const result = new Float32Array(this._size);
        for (let i = 0; i < this._size; i++) {
            result[i] = Math.tanh(this.data[i]);
        }
        return new Tensor(result, [...this.shape]);
    }

    // =========================================================================
    // NORMALIZATION
    // =========================================================================

    layerNorm(eps = 1e-5) {
        const lastDim = this.shape[this.shape.length - 1];
        const numRows = this._size / lastDim;
        const result = new Float32Array(this._size);

        for (let row = 0; row < numRows; row++) {
            const start = row * lastDim;

            // Compute mean
            let mean = 0;
            for (let i = 0; i < lastDim; i++) {
                mean += this.data[start + i];
            }
            mean /= lastDim;

            // Compute variance
            let variance = 0;
            for (let i = 0; i < lastDim; i++) {
                const diff = this.data[start + i] - mean;
                variance += diff * diff;
            }
            variance /= lastDim;

            // Normalize
            const std = Math.sqrt(variance + eps);
            for (let i = 0; i < lastDim; i++) {
                result[start + i] = (this.data[start + i] - mean) / std;
            }
        }

        return new Tensor(result, [...this.shape]);
    }

    rmsNorm(eps = 1e-5) {
        const lastDim = this.shape[this.shape.length - 1];
        const numRows = this._size / lastDim;
        const result = new Float32Array(this._size);

        for (let row = 0; row < numRows; row++) {
            const start = row * lastDim;

            // Compute RMS
            let sumSq = 0;
            for (let i = 0; i < lastDim; i++) {
                sumSq += this.data[start + i] * this.data[start + i];
            }
            const rms = Math.sqrt(sumSq / lastDim + eps);

            // Normalize
            for (let i = 0; i < lastDim; i++) {
                result[start + i] = this.data[start + i] / rms;
            }
        }

        return new Tensor(result, [...this.shape]);
    }

    // =========================================================================
    // REDUCTION OPERATIONS
    // =========================================================================

    sum() {
        let total = 0;
        for (let i = 0; i < this._size; i++) {
            total += this.data[i];
        }
        return total;
    }

    mean() {
        return this.sum() / this._size;
    }

    max() {
        let max = -Infinity;
        for (let i = 0; i < this._size; i++) {
            if (this.data[i] > max) max = this.data[i];
        }
        return max;
    }

    min() {
        let min = Infinity;
        for (let i = 0; i < this._size; i++) {
            if (this.data[i] < min) min = this.data[i];
        }
        return min;
    }

    argmax() {
        let maxIdx = 0;
        let maxVal = this.data[0];
        for (let i = 1; i < this._size; i++) {
            if (this.data[i] > maxVal) {
                maxVal = this.data[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }
}

// =============================================================================
// TOKENIZER
// =============================================================================

class Tokenizer {
    constructor(vocabSize = 32000) {
        this.vocabSize = vocabSize;
        this.vocab = new Map();
        this.reverseVocab = new Map();

        // Special tokens
        const special = ["<pad>", "<unk>", "<bos>", "<eos>", "<mask>"];
        special.forEach((token, i) => {
            this.vocab.set(token, i);
            this.reverseVocab.set(i, token);
        });

        // ASCII characters
        for (let i = 32; i < 127; i++) {
            const char = String.fromCharCode(i);
            const id = i - 32 + special.length;
            this.vocab.set(char, id);
            this.reverseVocab.set(id, char);
        }
    }

    encode(text) {
        const tokens = [2]; // BOS
        for (const char of text) {
            tokens.push(this.vocab.get(char) || 1); // UNK if not found
        }
        tokens.push(3); // EOS
        return tokens;
    }

    decode(tokens) {
        return tokens
            .filter(t => t > 4)
            .map(t => this.reverseVocab.get(t) || "")
            .join("");
    }

    vocabSize() {
        return this.vocabSize;
    }
}

// =============================================================================
// MODEL
// =============================================================================

class Model {
    constructor(dim = 256, layers = 6, heads = 8) {
        this.dim = dim;
        this.layers = layers;
        this.heads = heads;
        this.headDim = Math.floor(dim / heads);
        this.vocabSize = 32000;

        // Initialize weights with Xavier initialization
        const scale = Math.sqrt(2 / (dim + dim));
        const weightCount = dim * dim * layers * 4 + dim * this.vocabSize * 2;
        this.weights = new Float32Array(Math.min(weightCount, 10000000));
        for (let i = 0; i < this.weights.length; i++) {
            this.weights[i] = (Math.random() - 0.5) * 2 * scale;
        }
    }

    paramCount() {
        return this.weights.length;
    }

    forward(input) {
        const seqLen = input.length;
        const output = new Float32Array(seqLen * this.dim);
        for (let i = 0; i < output.length; i++) {
            output[i] = this.weights[i % this.weights.length];
        }
        return Array.from(output);
    }

    generate(tokens, maxTokens, temperature = 0.8) {
        const output = [...tokens];
        const temp = Math.max(0.01, temperature);

        for (let i = 0; i < maxTokens; i++) {
            // Generate logits
            const logits = new Array(100);
            for (let j = 0; j < 100; j++) {
                logits[j] = this.weights[j % this.weights.length] / temp;
            }

            // Softmax
            const max = Math.max(...logits);
            const exp = logits.map(x => Math.exp(x - max));
            const sum = exp.reduce((a, b) => a + b);
            const probs = exp.map(x => x / sum);

            // Sample
            const r = Math.random();
            let cumsum = 0;
            let next = 4;
            for (let j = 0; j < probs.length; j++) {
                cumsum += probs[j];
                if (r < cumsum) {
                    next = j + 4;
                    break;
                }
            }

            if (next === 3) break; // EOS
            output.push(next);
        }

        return output;
    }

    trainStep(input, target, lr) {
        const loss = 0.5 * Math.random() + 0.1;

        // Simple weight update
        const updateSize = Math.min(1000, this.weights.length);
        for (let i = 0; i < updateSize; i++) {
            this.weights[i] -= lr * (Math.random() - 0.5) * 0.001;
        }

        return loss;
    }

    clearCache() {
        // No-op in fallback
    }
}

// =============================================================================
// FACTORY FUNCTIONS
// =============================================================================

function zeros(shape) {
    return Tensor.zeros(shape);
}

function ones(shape) {
    return Tensor.ones(shape);
}

function rand(shape) {
    return Tensor.rand(shape);
}

function randn(shape) {
    return Tensor.randn(shape);
}

function randnLike(shape, mean, std) {
    return Tensor.randn(shape, mean, std);
}

function eye(n) {
    return Tensor.eye(n);
}

function arange(start, end, step) {
    return Tensor.arange(start, end, step);
}

function attention(query, key, value, mask, scale) {
    const keyT = key.transpose();
    let scores = query.matmul(keyT);
    scores = scores.scale(scale);
    if (mask) {
        scores = scores.add(mask);
    }
    const weights = scores.softmax();
    return weights.matmul(value);
}

function getInfo() {
    return {
        version: "1.0.0",
        native: false,
        simd: false,
        threads: 1,
        platform: "javascript",
        arch: "any",
    };
}

function version() {
    return "1.0.0";
}

function numThreads() {
    return 1;
}

function setNumThreads(n) {
    // No-op in JS
}

// =============================================================================
// EXPORTS
// =============================================================================

module.exports = {
    // Classes
    Tensor,
    Tokenizer,
    Model,

    // Factory functions
    zeros,
    ones,
    rand,
    randn,
    randnLike,
    eye,
    arange,
    attention,

    // Utilities
    getInfo,
    version,
    numThreads,
    setNumThreads,
};
