/**
 * Forge - RAG (Retrieval Augmented Generation)
 * =============================================
 */
import { Model } from "./model";
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
export declare class RAG {
    private generator;
    private documents;
    private embeddingDim;
    constructor(model: Model);
    /**
     * Add documents to the knowledge base
     */
    add(text: string | string[], metadata?: Record<string, any>): this;
    /**
     * Query the RAG system
     */
    query(question: string, config?: RAGConfig): Promise<string>;
    /**
     * Search for similar documents
     */
    search(query: string, topK?: number): Array<{
        doc: Document;
        score: number;
    }>;
    /**
     * Get document count
     */
    get size(): number;
    /**
     * Clear all documents
     */
    clear(): this;
    private embed;
    private cosineSimilarity;
}
//# sourceMappingURL=rag.d.ts.map