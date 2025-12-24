/**
 * Forge - Tensor Operations
 * ==========================
 *
 * Real tensor operations implemented in JavaScript.
 * This provides actual ML operations when Rust core is not available.
 */
export declare class Tensor {
    readonly data: Float32Array;
    readonly shape: number[];
    readonly size: number;
    readonly strides: number[];
    constructor(data: number[] | Float32Array, shape: number[]);
    static zeros(shape: number[]): Tensor;
    static ones(shape: number[]): Tensor;
    static randn(shape: number[], mean?: number, std?: number): Tensor;
    static fromArray(arr: number[], shape: number[]): Tensor;
    clone(): Tensor;
    reshape(newShape: number[]): Tensor;
    transpose(): Tensor;
    add(other: Tensor | number): Tensor;
    sub(other: Tensor | number): Tensor;
    mul(other: Tensor | number): Tensor;
    div(other: Tensor | number): Tensor;
    neg(): Tensor;
    matmul(other: Tensor): Tensor;
    bmm(other: Tensor): Tensor;
    relu(): Tensor;
    gelu(): Tensor;
    silu(): Tensor;
    sigmoid(): Tensor;
    tanh(): Tensor;
    softmax(dim?: number): Tensor;
    layerNorm(gamma?: Tensor, beta?: Tensor, eps?: number): Tensor;
    rmsNorm(gamma?: Tensor, eps?: number): Tensor;
    sum(dim?: number): Tensor | number;
    mean(dim?: number): Tensor | number;
    max(): number;
    min(): number;
    argmax(dim?: number): number | Tensor;
    toArray(): number[];
    toString(): string;
    get(indices: number[]): number;
    set(indices: number[], value: number): void;
    slice(start: number, end?: number): Tensor;
}
export declare function cat(tensors: Tensor[], dim?: number): Tensor;
export declare function stack(tensors: Tensor[], dim?: number): Tensor;
export default Tensor;
//# sourceMappingURL=tensor.d.ts.map