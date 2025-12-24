/**
 * Forge - Generator
 * ==================
 * 
 * Real text generation using the transformer model.
 */

import { Model } from "./model";
import { BPETokenizer } from "./bpe_tokenizer";

export interface GenerateConfig {
    maxTokens?: number;
    temperature?: number;
    topK?: number;
    topP?: number;
    repetitionPenalty?: number;
    stop?: string[];
    stream?: boolean;
}

export class Generator {
    private model: Model;
    private tokenizer: BPETokenizer;

    constructor(model: Model) {
        this.model = model;
        this.tokenizer = model.getTokenizer();
    }

    /**
     * Generate text from a prompt
     */
    async generate(prompt: string, config: GenerateConfig = {}): Promise<string> {
        const {
            maxTokens = 50,
            temperature = 0.7,
            topK = 50,
            topP = 0.9,
            repetitionPenalty = 1.1,
            stop = [],
        } = config;

        // Encode the prompt
        const inputTokens = this.tokenizer.encode(prompt, true, false);

        // Clear KV cache for fresh generation
        this.model.clearCache();

        // Generate using the model
        const outputTokens = this.model.generate(
            inputTokens,
            maxTokens,
            { temperature, topK, topP, repetitionPenalty }
        );

        // Decode the generated tokens (excluding input)
        const generatedTokens = outputTokens.slice(inputTokens.length);
        let output = this.tokenizer.decode(generatedTokens, true);

        // Apply stop sequences
        for (const stopSeq of stop) {
            const idx = output.indexOf(stopSeq);
            if (idx !== -1) {
                output = output.substring(0, idx);
                break;
            }
        }

        return output.trim();
    }

    /**
     * Stream text generation token by token
     */
    async *stream(prompt: string, config: GenerateConfig = {}): AsyncGenerator<string> {
        const {
            maxTokens = 50,
            temperature = 0.7,
            topK = 50,
            topP = 0.9,
            repetitionPenalty = 1.1,
            stop = [],
        } = config;

        // Encode prompt
        const tokens = this.tokenizer.encode(prompt, true, false);

        // Clear cache
        this.model.clearCache();

        // Forward pass on prompt first
        this.model.forward(tokens);

        let generated = "";
        const EOS_ID = 2; // End of sequence token

        for (let i = 0; i < maxTokens; i++) {
            // Get logits for next token
            const logits = this.model.forward(tokens);

            // Sample next token
            const nextToken = this._sampleFromLogits(
                logits,
                tokens,
                temperature,
                topK,
                topP,
                repetitionPenalty
            );

            // Check for EOS
            if (nextToken === EOS_ID) break;

            // Add to sequence
            tokens.push(nextToken);

            // Decode and yield the new token
            const tokenText = this.tokenizer.decode([nextToken], true);
            generated += tokenText;
            yield tokenText;

            // Check stop sequences
            if (stop.some(s => generated.includes(s))) break;
        }
    }

    /**
     * Sample a token from logits with temperature, top-k, and top-p
     */
    private _sampleFromLogits(
        logits: Float32Array,
        previousTokens: number[],
        temperature: number,
        topK: number,
        topP: number,
        repetitionPenalty: number
    ): number {
        const vocabSize = logits.length;

        // Apply repetition penalty
        if (repetitionPenalty !== 1.0) {
            for (const token of previousTokens) {
                if (token < vocabSize) {
                    logits[token] /= repetitionPenalty;
                }
            }
        }

        // Apply temperature
        if (temperature !== 1.0) {
            for (let i = 0; i < vocabSize; i++) {
                logits[i] /= temperature;
            }
        }

        // Create sorted indices for top-k and top-p
        const indices = Array.from({ length: vocabSize }, (_, i) => i);
        indices.sort((a, b) => logits[b] - logits[a]);

        // Apply top-k filtering
        const topKIndices = indices.slice(0, topK);

        // Compute softmax over top-k
        let maxLogit = logits[topKIndices[0]];
        const expLogits = topKIndices.map(i => Math.exp(logits[i] - maxLogit));
        const sumExp = expLogits.reduce((a, b) => a + b, 0);
        const probs = expLogits.map(e => e / sumExp);

        // Apply top-p (nucleus sampling)
        let cumSum = 0;
        let cutoffIdx = probs.length;
        for (let i = 0; i < probs.length; i++) {
            cumSum += probs[i];
            if (cumSum >= topP) {
                cutoffIdx = i + 1;
                break;
            }
        }

        // Renormalize after top-p cutoff
        const finalIndices = topKIndices.slice(0, cutoffIdx);
        const finalProbs = probs.slice(0, cutoffIdx);
        const finalSum = finalProbs.reduce((a, b) => a + b, 0);
        const normalizedProbs = finalProbs.map(p => p / finalSum);

        // Sample from the distribution
        const rand = Math.random();
        let cumulative = 0;
        for (let i = 0; i < normalizedProbs.length; i++) {
            cumulative += normalizedProbs[i];
            if (rand < cumulative) {
                return finalIndices[i];
            }
        }

        return finalIndices[0];
    }

    /**
     * Compute perplexity of text (useful for evaluation)
     */
    async perplexity(text: string): Promise<number> {
        const tokens = this.tokenizer.encode(text, true, true);
        if (tokens.length < 2) return Infinity;

        let totalLogProb = 0;
        let count = 0;

        this.model.clearCache();

        for (let i = 0; i < tokens.length - 1; i++) {
            const input = tokens.slice(0, i + 1);
            const logits = this.model.forward(input);

            // Convert logits to log-probabilities
            const maxLogit = Math.max(...logits);
            const expSum = logits.reduce((sum, l) => sum + Math.exp(l - maxLogit), 0);

            const targetToken = tokens[i + 1];
            const logProb = logits[targetToken] - maxLogit - Math.log(expSum);

            totalLogProb += logProb;
            count++;
        }

        return Math.exp(-totalLogProb / count);
    }
}
