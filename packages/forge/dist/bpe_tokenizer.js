"use strict";
/**
 * Forge - BPE Tokenizer
 * ======================
 *
 * Byte-Pair Encoding tokenizer implementation.
 * Compatible with GPT-2/LLaMA style tokenization.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.CharTokenizer = exports.BPETokenizer = exports.SPECIAL_TOKENS = void 0;
/**
 * Special tokens
 */
exports.SPECIAL_TOKENS = {
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
class BPETokenizer {
    vocab;
    reverseVocab;
    merges;
    bytesToUnicode;
    unicodeToBytes;
    vocabSize;
    padId;
    unkId;
    bosId;
    eosId;
    constructor(config = { vocabSize: 32000 }) {
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
        this.padId = this.vocab.get(config.padToken || exports.SPECIAL_TOKENS.PAD) || 0;
        this.unkId = this.vocab.get(config.unkToken || exports.SPECIAL_TOKENS.UNK) || 1;
        this.bosId = this.vocab.get(config.bosToken || exports.SPECIAL_TOKENS.BOS) || 2;
        this.eosId = this.vocab.get(config.eosToken || exports.SPECIAL_TOKENS.EOS) || 3;
    }
    _initBytesToUnicode() {
        // GPT-2 style byte-to-unicode mapping
        // Maps all 256 bytes to unique unicode characters
        const bs = [];
        const cs = [];
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
    _initVocab(config) {
        let id = 0;
        // Special tokens
        const specialTokens = [
            config.padToken || exports.SPECIAL_TOKENS.PAD,
            config.unkToken || exports.SPECIAL_TOKENS.UNK,
            config.bosToken || exports.SPECIAL_TOKENS.BOS,
            config.eosToken || exports.SPECIAL_TOKENS.EOS,
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
            if (id >= config.vocabSize)
                break;
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
    _generateCommonTokens() {
        // Common English subwords and tokens
        // In production, this would be learned from data
        const tokens = [];
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
    encode(text, addBos = true, addEos = false) {
        const tokens = [];
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
    decode(tokens, skipSpecial = true) {
        const textParts = [];
        for (const id of tokens) {
            const token = this.reverseVocab.get(id);
            if (!token)
                continue;
            // Skip special tokens if requested
            if (skipSpecial && this._isSpecialToken(token)) {
                continue;
            }
            textParts.push(token);
        }
        // Convert byte-level tokens back to text
        return this._byteTokensToText(textParts.join(""));
    }
    _textToByteTokens(text) {
        const tokens = [];
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
    _byteTokensToText(tokenStr) {
        const bytes = [];
        for (const char of tokenStr) {
            const byte = this.unicodeToBytes.get(char);
            if (byte !== undefined) {
                bytes.push(byte);
            }
        }
        const decoder = new TextDecoder();
        return decoder.decode(new Uint8Array(bytes));
    }
    _applyBPE(tokens) {
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
    _isSpecialToken(token) {
        return token.startsWith("<") && token.endsWith(">");
    }
    /**
     * Get vocabulary size
     */
    getVocabSize() {
        return this.vocabSize;
    }
    /**
     * Get token ID for a string
     */
    tokenToId(token) {
        return this.vocab.get(token) ?? this.unkId;
    }
    /**
     * Get string for a token ID
     */
    idToToken(id) {
        return this.reverseVocab.get(id) ?? exports.SPECIAL_TOKENS.UNK;
    }
    /**
     * Batch encode multiple texts
     */
    encodeBatch(texts, addBos = true, addEos = false) {
        return texts.map(text => this.encode(text, addBos, addEos));
    }
    /**
     * Batch decode multiple token sequences
     */
    decodeBatch(tokenBatch, skipSpecial = true) {
        return tokenBatch.map(tokens => this.decode(tokens, skipSpecial));
    }
    /**
     * Pad sequences to same length
     */
    pad(sequences, maxLen, padRight = true) {
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
    createAttentionMask(sequences) {
        return sequences.map(seq => seq.map(id => id === this.padId ? 0 : 1));
    }
}
exports.BPETokenizer = BPETokenizer;
/**
 * Simple character-level tokenizer (fallback)
 */
class CharTokenizer {
    vocab;
    reverseVocab;
    vocabSize;
    padId = 0;
    unkId = 1;
    bosId = 2;
    eosId = 3;
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
    _addToken(token, id) {
        this.vocab.set(token, id);
        this.reverseVocab.set(id, token);
    }
    encode(text, addBos = true, addEos = false) {
        const tokens = [];
        if (addBos)
            tokens.push(this.bosId);
        for (const char of text) {
            tokens.push(this.vocab.get(char) ?? this.unkId);
        }
        if (addEos)
            tokens.push(this.eosId);
        return tokens;
    }
    decode(tokens, skipSpecial = true) {
        return tokens
            .filter(id => !skipSpecial || (id > 3))
            .map(id => this.reverseVocab.get(id) ?? "")
            .join("");
    }
    getVocabSize() {
        return this.vocabSize;
    }
}
exports.CharTokenizer = CharTokenizer;
// Export default
exports.default = BPETokenizer;
//# sourceMappingURL=bpe_tokenizer.js.map