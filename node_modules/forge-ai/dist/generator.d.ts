/**
 * Forge - Generator
 * ==================
 *
 * Real text generation using the transformer model.
 */
import { Model } from "./model";
export interface GenerateConfig {
    maxTokens?: number;
    temperature?: number;
    topK?: number;
    topP?: number;
    repetitionPenalty?: number;
    stop?: string[];
    stream?: boolean;
}
export declare class Generator {
    private model;
    private tokenizer;
    constructor(model: Model);
    /**
     * Generate text from a prompt
     */
    generate(prompt: string, config?: GenerateConfig): Promise<string>;
    /**
     * Stream text generation token by token
     */
    stream(prompt: string, config?: GenerateConfig): AsyncGenerator<string>;
    /**
     * Sample a token from logits with temperature, top-k, and top-p
     */
    private _sampleFromLogits;
    /**
     * Compute perplexity of text (useful for evaluation)
     */
    perplexity(text: string): Promise<number>;
}
//# sourceMappingURL=generator.d.ts.map