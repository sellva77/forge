"use strict";
/**
 * Forge - Tensor Operations
 * ==========================
 *
 * Real tensor operations implemented in JavaScript.
 * This provides actual ML operations when Rust core is not available.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.Tensor = void 0;
exports.cat = cat;
exports.stack = stack;
class Tensor {
    data;
    shape;
    size;
    strides;
    constructor(data, shape) {
        this.shape = shape;
        this.size = shape.reduce((a, b) => a * b, 1);
        this.data = data instanceof Float32Array ? data : new Float32Array(data);
        // Calculate strides for indexing
        this.strides = new Array(shape.length);
        let stride = 1;
        for (let i = shape.length - 1; i >= 0; i--) {
            this.strides[i] = stride;
            stride *= shape[i];
        }
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
        const data = new Float32Array(size).fill(1);
        return new Tensor(data, shape);
    }
    static randn(shape, mean = 0, std = 1) {
        const size = shape.reduce((a, b) => a * b, 1);
        const data = new Float32Array(size);
        // Box-Muller transform for normal distribution
        for (let i = 0; i < size; i += 2) {
            const u1 = Math.random();
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
    static fromArray(arr, shape) {
        return new Tensor(new Float32Array(arr), shape);
    }
    // =========================================================================
    // BASIC OPERATIONS
    // =========================================================================
    clone() {
        return new Tensor(new Float32Array(this.data), [...this.shape]);
    }
    reshape(newShape) {
        const newSize = newShape.reduce((a, b) => a * b, 1);
        if (newSize !== this.size) {
            throw new Error(`Cannot reshape tensor of size ${this.size} to shape ${newShape}`);
        }
        return new Tensor(this.data, newShape);
    }
    transpose() {
        if (this.shape.length !== 2) {
            throw new Error("Transpose only supported for 2D tensors");
        }
        const [rows, cols] = this.shape;
        const result = new Float32Array(this.size);
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
        const result = new Float32Array(this.size);
        if (typeof other === "number") {
            for (let i = 0; i < this.size; i++) {
                result[i] = this.data[i] + other;
            }
        }
        else {
            if (this.size !== other.size) {
                throw new Error("Tensor sizes must match for addition");
            }
            for (let i = 0; i < this.size; i++) {
                result[i] = this.data[i] + other.data[i];
            }
        }
        return new Tensor(result, [...this.shape]);
    }
    sub(other) {
        const result = new Float32Array(this.size);
        if (typeof other === "number") {
            for (let i = 0; i < this.size; i++) {
                result[i] = this.data[i] - other;
            }
        }
        else {
            for (let i = 0; i < this.size; i++) {
                result[i] = this.data[i] - other.data[i];
            }
        }
        return new Tensor(result, [...this.shape]);
    }
    mul(other) {
        const result = new Float32Array(this.size);
        if (typeof other === "number") {
            for (let i = 0; i < this.size; i++) {
                result[i] = this.data[i] * other;
            }
        }
        else {
            for (let i = 0; i < this.size; i++) {
                result[i] = this.data[i] * other.data[i];
            }
        }
        return new Tensor(result, [...this.shape]);
    }
    div(other) {
        const result = new Float32Array(this.size);
        if (typeof other === "number") {
            for (let i = 0; i < this.size; i++) {
                result[i] = this.data[i] / other;
            }
        }
        else {
            for (let i = 0; i < this.size; i++) {
                result[i] = this.data[i] / other.data[i];
            }
        }
        return new Tensor(result, [...this.shape]);
    }
    neg() {
        return this.mul(-1);
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
            throw new Error(`Matrix dimensions incompatible: ${this.shape} x ${other.shape}`);
        }
        const result = new Float32Array(m * n);
        // Standard matrix multiplication O(n³)
        for (let i = 0; i < m; i++) {
            for (let j = 0; j < n; j++) {
                let sum = 0;
                for (let k = 0; k < k1; k++) {
                    sum += this.data[i * k1 + k] * other.data[k * n + j];
                }
                result[i * n + j] = sum;
            }
        }
        return new Tensor(result, [m, n]);
    }
    // Batched matrix multiplication [B, M, K] x [B, K, N] -> [B, M, N]
    bmm(other) {
        if (this.shape.length !== 3 || other.shape.length !== 3) {
            throw new Error("Batched matmul requires 3D tensors");
        }
        const [b1, m, k1] = this.shape;
        const [b2, k2, n] = other.shape;
        if (b1 !== b2 || k1 !== k2) {
            throw new Error(`Batch dimensions incompatible: ${this.shape} x ${other.shape}`);
        }
        const result = new Float32Array(b1 * m * n);
        for (let b = 0; b < b1; b++) {
            for (let i = 0; i < m; i++) {
                for (let j = 0; j < n; j++) {
                    let sum = 0;
                    for (let k = 0; k < k1; k++) {
                        sum += this.data[b * m * k1 + i * k1 + k] *
                            other.data[b * k2 * n + k * n + j];
                    }
                    result[b * m * n + i * n + j] = sum;
                }
            }
        }
        return new Tensor(result, [b1, m, n]);
    }
    // =========================================================================
    // ACTIVATION FUNCTIONS
    // =========================================================================
    relu() {
        const result = new Float32Array(this.size);
        for (let i = 0; i < this.size; i++) {
            result[i] = Math.max(0, this.data[i]);
        }
        return new Tensor(result, [...this.shape]);
    }
    gelu() {
        const result = new Float32Array(this.size);
        const sqrt2overPi = Math.sqrt(2 / Math.PI);
        for (let i = 0; i < this.size; i++) {
            const x = this.data[i];
            // GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            const cdf = 0.5 * (1 + Math.tanh(sqrt2overPi * (x + 0.044715 * x * x * x)));
            result[i] = x * cdf;
        }
        return new Tensor(result, [...this.shape]);
    }
    silu() {
        // SiLU (Swish): x * sigmoid(x)
        const result = new Float32Array(this.size);
        for (let i = 0; i < this.size; i++) {
            const x = this.data[i];
            result[i] = x / (1 + Math.exp(-x));
        }
        return new Tensor(result, [...this.shape]);
    }
    sigmoid() {
        const result = new Float32Array(this.size);
        for (let i = 0; i < this.size; i++) {
            result[i] = 1 / (1 + Math.exp(-this.data[i]));
        }
        return new Tensor(result, [...this.shape]);
    }
    tanh() {
        const result = new Float32Array(this.size);
        for (let i = 0; i < this.size; i++) {
            result[i] = Math.tanh(this.data[i]);
        }
        return new Tensor(result, [...this.shape]);
    }
    // =========================================================================
    // NORMALIZATION
    // =========================================================================
    softmax(dim = -1) {
        const actualDim = dim < 0 ? this.shape.length + dim : dim;
        const result = new Float32Array(this.size);
        if (this.shape.length === 1 || (this.shape.length === 2 && actualDim === 1)) {
            // Fast path for 1D or 2D with last dimension
            const rows = this.shape.length === 1 ? 1 : this.shape[0];
            const cols = this.shape.length === 1 ? this.shape[0] : this.shape[1];
            for (let i = 0; i < rows; i++) {
                const offset = i * cols;
                // Find max for numerical stability
                let max = -Infinity;
                for (let j = 0; j < cols; j++) {
                    max = Math.max(max, this.data[offset + j]);
                }
                // Compute exp and sum
                let sum = 0;
                for (let j = 0; j < cols; j++) {
                    const exp = Math.exp(this.data[offset + j] - max);
                    result[offset + j] = exp;
                    sum += exp;
                }
                // Normalize
                for (let j = 0; j < cols; j++) {
                    result[offset + j] /= sum;
                }
            }
        }
        else {
            throw new Error("Softmax only implemented for 1D and 2D tensors");
        }
        return new Tensor(result, [...this.shape]);
    }
    layerNorm(gamma, beta, eps = 1e-5) {
        if (this.shape.length !== 2) {
            throw new Error("Layer norm currently only supports 2D tensors");
        }
        const [batchSize, hiddenSize] = this.shape;
        const result = new Float32Array(this.size);
        for (let b = 0; b < batchSize; b++) {
            const offset = b * hiddenSize;
            // Compute mean
            let mean = 0;
            for (let i = 0; i < hiddenSize; i++) {
                mean += this.data[offset + i];
            }
            mean /= hiddenSize;
            // Compute variance
            let variance = 0;
            for (let i = 0; i < hiddenSize; i++) {
                const diff = this.data[offset + i] - mean;
                variance += diff * diff;
            }
            variance /= hiddenSize;
            // Normalize
            const std = Math.sqrt(variance + eps);
            for (let i = 0; i < hiddenSize; i++) {
                let normalized = (this.data[offset + i] - mean) / std;
                // Apply affine transform if provided
                if (gamma && beta) {
                    normalized = normalized * gamma.data[i] + beta.data[i];
                }
                result[offset + i] = normalized;
            }
        }
        return new Tensor(result, [...this.shape]);
    }
    rmsNorm(gamma, eps = 1e-5) {
        const result = new Float32Array(this.size);
        if (this.shape.length === 2) {
            const [batchSize, hiddenSize] = this.shape;
            for (let b = 0; b < batchSize; b++) {
                const offset = b * hiddenSize;
                // Compute RMS
                let sumSquares = 0;
                for (let i = 0; i < hiddenSize; i++) {
                    sumSquares += this.data[offset + i] * this.data[offset + i];
                }
                const rms = Math.sqrt(sumSquares / hiddenSize + eps);
                // Normalize
                for (let i = 0; i < hiddenSize; i++) {
                    let normalized = this.data[offset + i] / rms;
                    if (gamma) {
                        normalized *= gamma.data[i % gamma.size];
                    }
                    result[offset + i] = normalized;
                }
            }
        }
        else if (this.shape.length === 3) {
            // 3D tensor: [batch, seq, hidden]
            const [batch, seq, hiddenSize] = this.shape;
            for (let b = 0; b < batch; b++) {
                for (let s = 0; s < seq; s++) {
                    const offset = b * seq * hiddenSize + s * hiddenSize;
                    // Compute RMS
                    let sumSquares = 0;
                    for (let i = 0; i < hiddenSize; i++) {
                        sumSquares += this.data[offset + i] * this.data[offset + i];
                    }
                    const rms = Math.sqrt(sumSquares / hiddenSize + eps);
                    // Normalize
                    for (let i = 0; i < hiddenSize; i++) {
                        let normalized = this.data[offset + i] / rms;
                        if (gamma) {
                            normalized *= gamma.data[i % gamma.size];
                        }
                        result[offset + i] = normalized;
                    }
                }
            }
        }
        else {
            throw new Error(`RMS norm supports 2D and 3D tensors, got ${this.shape.length}D`);
        }
        return new Tensor(result, [...this.shape]);
    }
    // =========================================================================
    // REDUCTION OPERATIONS
    // =========================================================================
    sum(dim) {
        if (dim === undefined) {
            let total = 0;
            for (let i = 0; i < this.size; i++) {
                total += this.data[i];
            }
            return total;
        }
        const actualDim = dim < 0 ? this.shape.length + dim : dim;
        const newShape = [...this.shape];
        newShape.splice(actualDim, 1);
        if (newShape.length === 0) {
            return this.sum();
        }
        const newSize = newShape.reduce((a, b) => a * b, 1);
        const result = new Float32Array(newSize);
        // Implementation for 2D case
        if (this.shape.length === 2) {
            const [rows, cols] = this.shape;
            if (actualDim === 0) {
                for (let j = 0; j < cols; j++) {
                    let sum = 0;
                    for (let i = 0; i < rows; i++) {
                        sum += this.data[i * cols + j];
                    }
                    result[j] = sum;
                }
            }
            else {
                for (let i = 0; i < rows; i++) {
                    let sum = 0;
                    for (let j = 0; j < cols; j++) {
                        sum += this.data[i * cols + j];
                    }
                    result[i] = sum;
                }
            }
        }
        return new Tensor(result, newShape);
    }
    mean(dim) {
        if (dim === undefined) {
            return this.sum() / this.size;
        }
        const actualDim = dim < 0 ? this.shape.length + dim : dim;
        const sumResult = this.sum(dim);
        return sumResult.div(this.shape[actualDim]);
    }
    max() {
        let max = -Infinity;
        for (let i = 0; i < this.size; i++) {
            max = Math.max(max, this.data[i]);
        }
        return max;
    }
    min() {
        let min = Infinity;
        for (let i = 0; i < this.size; i++) {
            min = Math.min(min, this.data[i]);
        }
        return min;
    }
    argmax(dim = -1) {
        if (this.shape.length === 1) {
            let maxIdx = 0;
            let maxVal = this.data[0];
            for (let i = 1; i < this.size; i++) {
                if (this.data[i] > maxVal) {
                    maxVal = this.data[i];
                    maxIdx = i;
                }
            }
            return maxIdx;
        }
        // 2D case: argmax along last dimension
        const [rows, cols] = this.shape;
        const result = new Float32Array(rows);
        for (let i = 0; i < rows; i++) {
            let maxIdx = 0;
            let maxVal = this.data[i * cols];
            for (let j = 1; j < cols; j++) {
                if (this.data[i * cols + j] > maxVal) {
                    maxVal = this.data[i * cols + j];
                    maxIdx = j;
                }
            }
            result[i] = maxIdx;
        }
        return new Tensor(result, [rows]);
    }
    // =========================================================================
    // UTILITY
    // =========================================================================
    toArray() {
        return Array.from(this.data);
    }
    toString() {
        return `Tensor(shape=[${this.shape.join(", ")}], data=[${this.data.slice(0, 5).join(", ")}${this.size > 5 ? ", ..." : ""}])`;
    }
    get(indices) {
        let idx = 0;
        for (let i = 0; i < indices.length; i++) {
            idx += indices[i] * this.strides[i];
        }
        return this.data[idx];
    }
    set(indices, value) {
        let idx = 0;
        for (let i = 0; i < indices.length; i++) {
            idx += indices[i] * this.strides[i];
        }
        this.data[idx] = value;
    }
    slice(start, end) {
        if (this.shape.length !== 1 && this.shape.length !== 2) {
            throw new Error("Slice only supports 1D and 2D tensors");
        }
        if (this.shape.length === 1) {
            const actualEnd = end ?? this.shape[0];
            const newData = this.data.slice(start, actualEnd);
            return new Tensor(newData, [actualEnd - start]);
        }
        else {
            const [rows, cols] = this.shape;
            const actualEnd = end ?? rows;
            const newData = this.data.slice(start * cols, actualEnd * cols);
            return new Tensor(newData, [actualEnd - start, cols]);
        }
    }
}
exports.Tensor = Tensor;
// =========================================================================
// UTILITY FUNCTIONS
// =========================================================================
function cat(tensors, dim = 0) {
    if (tensors.length === 0) {
        throw new Error("Cannot concatenate empty tensor list");
    }
    // For now, only support 1D concatenation
    if (tensors[0].shape.length === 1) {
        const totalSize = tensors.reduce((sum, t) => sum + t.size, 0);
        const result = new Float32Array(totalSize);
        let offset = 0;
        for (const t of tensors) {
            result.set(t.data, offset);
            offset += t.size;
        }
        return new Tensor(result, [totalSize]);
    }
    throw new Error("cat only implemented for 1D tensors");
}
function stack(tensors, dim = 0) {
    if (tensors.length === 0) {
        throw new Error("Cannot stack empty tensor list");
    }
    const shape = tensors[0].shape;
    const newShape = [...shape];
    newShape.splice(dim, 0, tensors.length);
    const totalSize = tensors.length * tensors[0].size;
    const result = new Float32Array(totalSize);
    for (let i = 0; i < tensors.length; i++) {
        result.set(tensors[i].data, i * tensors[0].size);
    }
    return new Tensor(result, newShape);
}
exports.default = Tensor;
//# sourceMappingURL=tensor.js.map