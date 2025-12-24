/**
 * Forge - RAG (Retrieval Augmented Generation)
 * =============================================
 */

import { Model } from "./model";
import { Generator } from "./generator";

export interface Document {
    id: string;
    text: string;
    embedding: number[];
    metadata?: Record<string, any>;
}

export interface RAGConfig {
    topK?: number;
    minScore?: number;
}

export class RAG {
    private generator: Generator;
    private documents: Document[] = [];
    private embeddingDim: number = 64;

    constructor(model: Model) {
        this.generator = new Generator(model);
    }

    /**
     * Add documents to the knowledge base
     */
    add(text: string | string[], metadata?: Record<string, any>): this {
        const texts = Array.isArray(text) ? text : [text];

        for (const t of texts) {
            const doc: Document = {
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
    async query(question: string, config: RAGConfig = {}): Promise<string> {
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
    search(query: string, topK: number = 5): Array<{ doc: Document; score: number }> {
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
    get size(): number {
        return this.documents.length;
    }

    /**
     * Clear all documents
     */
    clear(): this {
        this.documents = [];
        return this;
    }

    // Simple embedding (bag of characters)
    private embed(text: string): number[] {
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

    private cosineSimilarity(a: number[], b: number[]): number {
        let dot = 0;
        for (let i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
        }
        return dot;
    }
}
