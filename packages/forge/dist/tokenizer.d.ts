/**
 * Forge - Tokenizer
 * ==================
 */
export declare class Tokenizer {
    private vocab;
    private reverseVocab;
    vocabSize: number;
    static PAD: number;
    static UNK: number;
    static BOS: number;
    static EOS: number;
    constructor(vocabSize?: number);
    private buildVocab;
    encode(text: string): number[];
    decode(tokens: number[]): string;
    get size(): number;
}
//# sourceMappingURL=tokenizer.d.ts.map