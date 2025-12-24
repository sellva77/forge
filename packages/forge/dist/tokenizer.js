"use strict";
/**
 * Forge - Tokenizer
 * ==================
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.Tokenizer = void 0;
class Tokenizer {
    vocab = new Map();
    reverseVocab = new Map();
    vocabSize;
    // Special tokens
    static PAD = 0;
    static UNK = 1;
    static BOS = 2;
    static EOS = 3;
    constructor(vocabSize = 32000) {
        this.vocabSize = vocabSize;
        this.buildVocab();
    }
    buildVocab() {
        // Special tokens
        this.vocab.set("<pad>", Tokenizer.PAD);
        this.vocab.set("<unk>", Tokenizer.UNK);
        this.vocab.set("<bos>", Tokenizer.BOS);
        this.vocab.set("<eos>", Tokenizer.EOS);
        // ASCII printable characters
        for (let i = 32; i < 127; i++) {
            const char = String.fromCharCode(i);
            const id = i - 32 + 4;
            this.vocab.set(char, id);
        }
        // Common words
        const words = [
            "the", "is", "a", "an", "and", "or", "but", "in", "on", "at",
            "to", "for", "of", "with", "as", "by", "from", "that", "which",
            "this", "it", "be", "are", "was", "were", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "must", "can", "not", "no",
            "yes", "all", "some", "any", "each", "every", "both", "few",
            "more", "most", "other", "such", "only", "own", "same", "so",
            "than", "too", "very", "just", "also", "now", "here", "there",
        ];
        let id = 100;
        for (const word of words) {
            this.vocab.set(word, id++);
        }
        // Build reverse vocab
        for (const [token, tokenId] of this.vocab) {
            this.reverseVocab.set(tokenId, token);
        }
    }
    encode(text) {
        const tokens = [Tokenizer.BOS];
        // Simple word + character tokenization
        const words = text.toLowerCase().split(/\s+/);
        for (const word of words) {
            if (this.vocab.has(word)) {
                tokens.push(this.vocab.get(word));
            }
            else {
                // Character-level fallback
                for (const char of word) {
                    tokens.push(this.vocab.get(char) || Tokenizer.UNK);
                }
            }
            tokens.push(this.vocab.get(" ") || Tokenizer.UNK);
        }
        tokens.push(Tokenizer.EOS);
        return tokens;
    }
    decode(tokens) {
        return tokens
            .filter(t => t > 3) // Skip special tokens
            .map(t => this.reverseVocab.get(t) || "")
            .join("")
            .trim();
    }
    get size() {
        return this.vocab.size;
    }
}
exports.Tokenizer = Tokenizer;
//# sourceMappingURL=tokenizer.js.map