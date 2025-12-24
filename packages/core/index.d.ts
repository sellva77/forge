/**
 * Forge Core Types
 */

export class Tensor {
    constructor(data: number[] | Float32Array, shape: number[]);

    readonly data: Float32Array;
    readonly shape: number[];
    readonly size: number;

    static zeros(shape: number[]): Tensor;
    static ones(shape: number[]): Tensor;
    static randn(shape: number[], mean?: number, std?: number): Tensor;

    add(other: Tensor | number): Tensor;
    mul(other: Tensor | number): Tensor;
    matmul(other: Tensor): Tensor;
}

export class Tokenizer {
    constructor(path?: string);
    encode(text: string): Uint32Array;
    decode(tokens: Uint32Array | number[]): string;
}

export class Model {
    constructor(path?: string);
    forward(input: Uint32Array | number[]): Tensor;
    generate(prompt: Uint32Array | number[], maxTokens: number): Uint32Array;
}

export const isNative: boolean;
export const platform: string | null;
export const version: string;

export function checkNative(): boolean;
export function getInfo(): {
    version: string;
    native: boolean;
    platform: string | null;
    node: string;
};
