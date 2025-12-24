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
export const SPECIAL_TOKENS = {
    PAD: "<pad>",
    UNK: "<unk>",
    BOS: "<s>",
    EOS: "</s>",
    MASK: "<mask>",
};

/**
 * BPE Tokenizer
 * 
 * Provides byte-level BPE tokenization similar to GPT-2/LLaMA.
 */
export class BPETokenizer {
    private vocab: Map<string, number>;
    private reverseVocab: Map<number, string>;
    private merges: Map<string, number>;
    private bytesToUnicode: Map<number, string>;
    private unicodeToBytes: Map<string, number>;

    readonly vocabSize: number;
    readonly padId: number;
    readonly unkId: number;
    readonly bosId: number;
    readonly eosId: number;

    constructor(config: TokenizerConfig = { vocabSize: 32000 }) {
        this.vocabSize = config.vocabSize;
        this.vocab = new Map();
        this.reverseVocab = new Map();
        this.merges = new Map();

        // Build byte-to-unicode mapping (for handling all bytes)
        this.bytesToUnicode = new Map();
        this.unicodeToBytes = new Map();
        this._initBytesToUnicode();

        // Initialize vocabulary
        this._initVocab(config);

        // Store special token IDs
        this.padId = this.vocab.get(config.padToken || SPECIAL_TOKENS.PAD) || 0;
        this.unkId = this.vocab.get(config.unkToken || SPECIAL_TOKENS.UNK) || 1;
        this.bosId = this.vocab.get(config.bosToken || SPECIAL_TOKENS.BOS) || 2;
        this.eosId = this.vocab.get(config.eosToken || SPECIAL_TOKENS.EOS) || 3;
    }

    private _initBytesToUnicode(): void {
        // GPT-2 style byte-to-unicode mapping
        // Maps all 256 bytes to unique unicode characters
        const bs: number[] = [];
        const cs: number[] = [];

        // Printable ASCII
        for (let i = 33; i <= 126; i++) {
            bs.push(i);
            cs.push(i);
        }

        // Extended characters
        for (let i = 161; i <= 172; i++) {
            bs.push(i);
            cs.push(i);
        }
        for (let i = 174; i <= 255; i++) {
            bs.push(i);
            cs.push(i);
        }

        // Map remaining bytes to extended unicode
        let n = 0;
        for (let b = 0; b < 256; b++) {
            if (!bs.includes(b)) {
                bs.push(b);
                cs.push(256 + n);
                n++;
            }
        }

        for (let i = 0; i < bs.length; i++) {
            this.bytesToUnicode.set(bs[i], String.fromCharCode(cs[i]));
            this.unicodeToBytes.set(String.fromCharCode(cs[i]), bs[i]);
        }
    }

    private _initVocab(config: TokenizerConfig): void {
        let id = 0;

        // Special tokens
        const specialTokens = [
            config.padToken || SPECIAL_TOKENS.PAD,
            config.unkToken || SPECIAL_TOKENS.UNK,
            config.bosToken || SPECIAL_TOKENS.BOS,
            config.eosToken || SPECIAL_TOKENS.EOS,
        ];

        for (const token of specialTokens) {
            this.vocab.set(token, id);
            this.reverseVocab.set(id, token);
            id++;
        }

        // Single byte tokens
        for (let b = 0; b < 256; b++) {
            const char = this.bytesToUnicode.get(b) || "";
            if (!this.vocab.has(char)) {
                this.vocab.set(char, id);
                this.reverseVocab.set(id, char);
                id++;
            }
        }

        // Common subword tokens (simplified - in production, load from file)
        const commonTokens = this._generateCommonTokens();
        for (const token of commonTokens) {
            if (id >= config.vocabSize) break;
            if (!this.vocab.has(token)) {
                this.vocab.set(token, id);
                this.reverseVocab.set(id, token);
                id++;
            }
        }

        // Fill remaining with placeholder tokens
        while (id < config.vocabSize) {
            const placeholder = `<unused${id}>`;
            this.vocab.set(placeholder, id);
            this.reverseVocab.set(id, placeholder);
            id++;
        }
    }

    private _generateCommonTokens(): string[] {
        // Common English subwords and tokens
        // In production, this would be learned from data
        const tokens: string[] = [];

        // Common word endings
        const endings = ["ing", "ed", "er", "est", "ly", "tion", "ness", "ment", "able", "ible"];

        // Common words
        const words = [
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
            "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
            "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
            "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
            "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
            "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
            "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
            "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
            "even", "new", "want", "because", "any", "these", "give", "day", "most", "us",
        ];

        // Common programming tokens
        const progTokens = [
            "function", "return", "const", "let", "var", "class", "import", "export",
            "async", "await", "if", "else", "for", "while", "try", "catch",
            "true", "false", "null", "undefined", "console", "log", "error",
        ];

        // Add with space prefix (common BPE pattern)
        for (const word of [...words, ...progTokens]) {
            tokens.push(` ${word}`);
            tokens.push(word);
        }

        // Add endings
        for (const ending of endings) {
            tokens.push(ending);
        }

        // Common bigrams
        const bigrams = ["th", "he", "in", "er", "an", "re", "on", "at", "en", "nd"];
        for (const bi of bigrams) {
            tokens.push(bi);
        }

        // Numbers
        for (let i = 0; i <= 100; i++) {
            tokens.push(` ${i}`);
            tokens.push(`${i}`);
        }

        // Punctuation combinations
        tokens.push("...", "->", "=>", "==", "!=", "<=", ">=", "++", "--", "&&", "||");

        return tokens;
    }

    /**
     * Encode text to token IDs
     */
    encode(text: string, addBos = true, addEos = false): number[] {
        const tokens: number[] = [];

        if (addBos) {
            tokens.push(this.bosId);
        }

        // Convert text to byte-level tokens
        const byteTokens = this._textToByteTokens(text);

        // Apply BPE merges
        const mergedTokens = this._applyBPE(byteTokens);

        // Convert to IDs
        for (const token of mergedTokens) {
            tokens.push(this.vocab.get(token) ?? this.unkId);
        }

        if (addEos) {
            tokens.push(this.eosId);
        }

        return tokens;
    }

    /**
     * Decode token IDs to text
     */
    decode(tokens: number[], skipSpecial = true): string {
        const textParts: string[] = [];

        for (const id of tokens) {
            const token = this.reverseVocab.get(id);

            if (!token) continue;

            // Skip special tokens if requested
            if (skipSpecial && this._isSpecialToken(token)) {
                continue;
            }

            textParts.push(token);
        }

        // Convert byte-level tokens back to text
        return this._byteTokensToText(textParts.join(""));
    }

    private _textToByteTokens(text: string): string[] {
        const tokens: string[] = [];
        const encoder = new TextEncoder();
        const bytes = encoder.encode(text);

        for (const byte of bytes) {
            const char = this.bytesToUnicode.get(byte);
            if (char) {
                tokens.push(char);
            }
        }

        return tokens;
    }

    private _byteTokensToText(tokenStr: string): string {
        const bytes: number[] = [];

        for (const char of tokenStr) {
            const byte = this.unicodeToBytes.get(char);
            if (byte !== undefined) {
                bytes.push(byte);
            }
        }

        const decoder = new TextDecoder();
        return decoder.decode(new Uint8Array(bytes));
    }

    private _applyBPE(tokens: string[]): string[] {
        // Simplified BPE - in production, use learned merges
        let result = [...tokens];

        // Look for common patterns to merge
        let changed = true;
        let iterations = 0;
        const maxIterations = 100;

        while (changed && iterations < maxIterations) {
            changed = false;
            iterations++;

            for (let i = 0; i < result.length - 1; i++) {
                const pair = result[i] + result[i + 1];

                // Check if merged token exists in vocab
                if (this.vocab.has(pair)) {
                    result = [...result.slice(0, i), pair, ...result.slice(i + 2)];
                    changed = true;
                    break;
                }
            }
        }

        return result;
    }

    private _isSpecialToken(token: string): boolean {
        return token.startsWith("<") && token.endsWith(">");
    }

    /**
     * Get vocabulary size
     */
    getVocabSize(): number {
        return this.vocabSize;
    }

    /**
     * Get token ID for a string
     */
    tokenToId(token: string): number {
        return this.vocab.get(token) ?? this.unkId;
    }

    /**
     * Get string for a token ID
     */
    idToToken(id: number): string {
        return this.reverseVocab.get(id) ?? SPECIAL_TOKENS.UNK;
    }

    /**
     * Batch encode multiple texts
     */
    encodeBatch(texts: string[], addBos = true, addEos = false): number[][] {
        return texts.map(text => this.encode(text, addBos, addEos));
    }

    /**
     * Batch decode multiple token sequences
     */
    decodeBatch(tokenBatch: number[][], skipSpecial = true): string[] {
        return tokenBatch.map(tokens => this.decode(tokens, skipSpecial));
    }

    /**
     * Pad sequences to same length
     */
    pad(sequences: number[][], maxLen?: number, padRight = true): number[][] {
        const targetLen = maxLen ?? Math.max(...sequences.map(s => s.length));

        return sequences.map(seq => {
            if (seq.length >= targetLen) {
                return seq.slice(0, targetLen);
            }

            const padding = new Array(targetLen - seq.length).fill(this.padId);
            return padRight ? [...seq, ...padding] : [...padding, ...seq];
        });
    }

    /**
     * Create attention mask (1 for real tokens, 0 for padding)
     */
    createAttentionMask(sequences: number[][]): number[][] {
        return sequences.map(seq => seq.map(id => id === this.padId ? 0 : 1));
    }
}

/**
 * Simple character-level tokenizer (fallback)
 */
export class CharTokenizer {
    private vocab: Map<string, number>;
    private reverseVocab: Map<number, string>;

    readonly vocabSize: number;
    readonly padId = 0;
    readonly unkId = 1;
    readonly bosId = 2;
    readonly eosId = 3;

    constructor(vocabSize = 256) {
        this.vocabSize = Math.max(vocabSize, 128);
        this.vocab = new Map();
        this.reverseVocab = new Map();

        // Special tokens
        this._addToken("<pad>", 0);
        this._addToken("<unk>", 1);
        this._addToken("<s>", 2);
        this._addToken("</s>", 3);

        // ASCII characters
        for (let i = 32; i < 127; i++) {
            this._addToken(String.fromCharCode(i), i - 32 + 4);
        }
    }

    private _addToken(token: string, id: number): void {
        this.vocab.set(token, id);
        this.reverseVocab.set(id, token);
    }

    encode(text: string, addBos = true, addEos = false): number[] {
        const tokens: number[] = [];

        if (addBos) tokens.push(this.bosId);

        for (const char of text) {
            tokens.push(this.vocab.get(char) ?? this.unkId);
        }

        if (addEos) tokens.push(this.eosId);

        return tokens;
    }

    decode(tokens: number[], skipSpecial = true): string {
        return tokens
            .filter(id => !skipSpecial || (id > 3))
            .map(id => this.reverseVocab.get(id) ?? "")
            .join("");
    }

    getVocabSize(): number {
        return this.vocabSize;
    }
}

// Export default
export default BPETokenizer;

