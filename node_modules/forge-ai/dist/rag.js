"use strict";
/**
 * Forge - RAG (Retrieval Augmented Generation)
 * =============================================
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.RAG = void 0;
const generator_1 = require("./generator");
class RAG {
    generator;
    documents = [];
    embeddingDim = 64;
    constructor(model) {
        this.generator = new generator_1.Generator(model);
    }
    /**
     * Add documents to the knowledge base
     */
    add(text, metadata) {
        const texts = Array.isArray(text) ? text : [text];
        for (const t of texts) {
            const doc = {
                id: `doc_${this.documents.length}`,
                text: t,
                embedding: this.embed(t),
                metadata,
            };
            this.documents.push(doc);
        }
        console.log(`📚 Added ${texts.length} document(s). Total: ${this.documents.length}`);
        return this;
    }
    /**
     * Query the RAG system
     */
    async query(question, config = {}) {
        const { topK = 3, minScore = 0.0 } = config;
        // Find relevant documents
        const questionEmb = this.embed(question);
        const scored = this.documents.map(doc => ({
            doc,
            score: this.cosineSimilarity(questionEmb, doc.embedding),
        }));
        scored.sort((a, b) => b.score - a.score);
        const relevant = scored
            .filter(s => s.score >= minScore)
            .slice(0, topK);
        // Build context
        const context = relevant.map(r => r.doc.text).join("\n\n");
        // Generate answer
        const prompt = `Context:\n${context}\n\nQuestion: ${question}\nAnswer:`;
        return this.generator.generate(prompt);
    }
    /**
     * Search for similar documents
     */
    search(query, topK = 5) {
        const queryEmb = this.embed(query);
        const scored = this.documents.map(doc => ({
            doc,
            score: this.cosineSimilarity(queryEmb, doc.embedding),
        }));
        scored.sort((a, b) => b.score - a.score);
        return scored.slice(0, topK);
    }
    /**
     * Get document count
     */
    get size() {
        return this.documents.length;
    }
    /**
     * Clear all documents
     */
    clear() {
        this.documents = [];
        return this;
    }
    // Simple embedding (bag of characters)
    embed(text) {
        const vec = new Array(this.embeddingDim).fill(0);
        const lower = text.toLowerCase();
        for (let i = 0; i < lower.length; i++) {
            const idx = lower.charCodeAt(i) % this.embeddingDim;
            vec[idx] += 1;
        }
        // Normalize
        const norm = Math.sqrt(vec.reduce((sum, v) => sum + v * v, 0)) || 1;
        return vec.map(v => v / norm);
    }
    cosineSimilarity(a, b) {
        let dot = 0;
        for (let i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
        }
        return dot;
    }
}
exports.RAG = RAG;
//# sourceMappingURL=rag.js.map