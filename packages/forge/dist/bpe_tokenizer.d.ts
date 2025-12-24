/**
 * Forge - BPE Tokenizer
 * ======================
 *
 * Byte-Pair Encoding tokenizer implementation.
 * Compatible with GPT-2/LLaMA style tokenization.
 */
export interface TokenizerConfig {
    vocabSize: number;
    padToken?: string;
    unkToken?: string;
    bosToken?: string;
    eosToken?: string;
}
/**
 * Special tokens
 */
export declare const SPECIAL_TOKENS: {
    PAD: string;
    UNK: string;
    BOS: string;
    EOS: string;
    MASK: string;
};
/**
 * BPE Tokenizer
 *
 * Provides byte-level BPE tokenization similar to GPT-2/LLaMA.
 */
export declare class BPETokenizer {
    private vocab;
    private reverseVocab;
    private merges;
    private bytesToUnicode;
    private unicodeToBytes;
    readonly vocabSize: number;
    readonly padId: number;
    readonly unkId: number;
    readonly bosId: number;
    readonly eosId: number;
    constructor(config?: TokenizerConfig);
    private _initBytesToUnicode;
    private _initVocab;
    private _generateCommonTokens;
    /**
     * Encode text to token IDs
     */
    encode(text: string, addBos?: boolean, addEos?: boolean): number[];
    /**
     * Decode token IDs to text
     */
    decode(tokens: number[], skipSpecial?: boolean): string;
    private _textToByteTokens;
    private _byteTokensToText;
    private _applyBPE;
    private _isSpecialToken;
    /**
     * Get vocabulary size
     */
    getVocabSize(): number;
    /**
     * Get token ID for a string
     */
    tokenToId(token: string): number;
    /**
     * Get string for a token ID
     */
    idToToken(id: number): string;
    /**
     * Batch encode multiple texts
     */
    encodeBatch(texts: string[], addBos?: boolean, addEos?: boolean): number[][];
    /**
     * Batch decode multiple token sequences
     */
    decodeBatch(tokenBatch: number[][], skipSpecial?: boolean): string[];
    /**
     * Pad sequences to same length
     */
    pad(sequences: number[][], maxLen?: number, padRight?: boolean): number[][];
    /**
     * Create attention mask (1 for real tokens, 0 for padding)
     */
    createAttentionMask(sequences: number[][]): number[][];
}
/**
 * Simple character-level tokenizer (fallback)
 */
export declare class CharTokenizer {
    private vocab;
    private reverseVocab;
    readonly vocabSize: number;
    readonly padId = 0;
    readonly unkId = 1;
    readonly bosId = 2;
    readonly eosId = 3;
    constructor(vocabSize?: number);
    private _addToken;
    encode(text: string, addBos?: boolean, addEos?: boolean): number[];
    decode(tokens: number[], skipSpecial?: boolean): string;
    getVocabSize(): number;
}
export default BPETokenizer;
//# sourceMappingURL=bpe_tokenizer.d.ts.map