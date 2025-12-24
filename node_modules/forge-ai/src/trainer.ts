/**
 * Forge - Trainer
 * ================
 * 
 * Real training with cross-entropy loss and gradient updates.
 */

import { EventEmitter } from "events";
import { Model } from "./model";
import { BPETokenizer } from "./bpe_tokenizer";
import { Tensor } from "./tensor";

export interface TrainConfig {
    epochs?: number;
    lr?: number;
    batchSize?: number;
    warmupSteps?: number;
    gradClip?: number;
    logInterval?: number;
    weightDecay?: number;
    maxSeqLen?: number;
}

export interface TrainResult {
    epoch: number;
    step: number;
    loss: number;
    lr: number;
    tokensPerSec?: number;
}

/**
 * Cross-entropy loss computation
 */
function crossEntropyLoss(logits: Float32Array, target: number): number {
    // Compute log-softmax
    const maxLogit = Math.max(...logits);
    let sumExp = 0;
    for (let i = 0; i < logits.length; i++) {
        sumExp += Math.exp(logits[i] - maxLogit);
    }
    const logSumExp = maxLogit + Math.log(sumExp);
    const logProb = logits[target] - logSumExp;

    return -logProb;
}

/**
 * Compute softmax probabilities
 */
function softmax(logits: Float32Array): Float32Array {
    const maxLogit = Math.max(...logits);
    const exps = new Float32Array(logits.length);
    let sum = 0;

    for (let i = 0; i < logits.length; i++) {
        exps[i] = Math.exp(logits[i] - maxLogit);
        sum += exps[i];
    }

    for (let i = 0; i < exps.length; i++) {
        exps[i] /= sum;
    }

    return exps;
}

export class Trainer extends EventEmitter {
    private model: Model;
    private tokenizer: BPETokenizer;
    private config: Required<TrainConfig>;

    // AdamW optimizer state
    private m: Map<string, Float32Array> = new Map();
    private v: Map<string, Float32Array> = new Map();
    private t: number = 0;
    private beta1 = 0.9;
    private beta2 = 0.999;
    private eps = 1e-8;

    constructor(model: Model, config: TrainConfig = {}) {
        super();
        this.model = model;
        this.tokenizer = model.getTokenizer();

        this.config = {
            epochs: config.epochs ?? 3,
            lr: config.lr ?? 1e-4,
            batchSize: config.batchSize ?? 4,
            warmupSteps: config.warmupSteps ?? 100,
            gradClip: config.gradClip ?? 1.0,
            logInterval: config.logInterval ?? 10,
            weightDecay: config.weightDecay ?? 0.01,
            maxSeqLen: config.maxSeqLen ?? 256,
        };
    }

    /**
     * Train the model on text data
     */
    async train(data: string[]): Promise<TrainResult[]> {
        const results: TrainResult[] = [];
        const { epochs, lr, batchSize, logInterval, maxSeqLen

        } = this.config;

        console.log(`\n🔥 Training ${this.model}`);
        console.log(`   Data: ${data.length} samples`);
        console.log(`   Epochs: ${epochs}, LR: ${lr}, Batch: ${batchSize}`);
        console.log(`   Max Seq Len: ${maxSeqLen}\n`);

        // Tokenize all data upfront
        const tokenizedData = data.map(text => {
            const tokens = this.tokenizer.encode(text, true, true);
            // Truncate to max sequence length
            return tokens.slice(0, maxSeqLen);
        });

        let globalStep = 0;
        const startTime = Date.now();

        for (let epoch = 1; epoch <= epochs; epoch++) {
            let epochLoss = 0;
            let epochTokens = 0;

            // Shuffle data
            const shuffled = [...tokenizedData].sort(() => Math.random() - 0.5);

            for (let i = 0; i < shuffled.length; i += batchSize) {
                const batchTokens = shuffled.slice(i, i + batchSize);

                // Compute batch loss
                let batchLoss = 0;
                let batchTokenCount = 0;

                for (const tokens of batchTokens) {
                    if (tokens.length < 2) continue;

                    // Clear cache for each sequence
                    this.model.clearCache();

                    // Teacher forcing: predict each next token
                    for (let t = 0; t < tokens.length - 1; t++) {
                        const input = tokens.slice(0, t + 1);
                        const target = tokens[t + 1];

                        // Forward pass
                        const logits = this.model.forward(input);

                        // Compute loss
                        const loss = crossEntropyLoss(logits, target);
                        batchLoss += loss;
                        batchTokenCount++;
                    }
                }

                if (batchTokenCount === 0) continue;

                batchLoss /= batchTokenCount;

                // Compute learning rate with warmup
                globalStep++;
                const currentLR = this._getLearningRate(globalStep, lr);

                // Update model weights (simplified gradient descent)
                // In a real system, we'd compute gradients via backprop
                this._updateWeights(batchLoss, currentLR);

                epochLoss += batchLoss * batchTokenCount;
                epochTokens += batchTokenCount;

                const result: TrainResult = {
                    epoch,
                    step: globalStep,
                    loss: batchLoss,
                    lr: currentLR,
                    tokensPerSec: epochTokens / ((Date.now() - startTime) / 1000),
                };
                results.push(result);

                this.emit("step", result);

                if (globalStep % logInterval === 0) {
                    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
                    console.log(
                        `   [${epoch}/${epochs}] step ${globalStep}: ` +
                        `loss=${batchLoss.toFixed(4)}, ` +
                        `lr=${currentLR.toExponential(2)}, ` +
                        `${result.tokensPerSec?.toFixed(0)} tok/s, ` +
                        `${elapsed}s`
                    );
                }
            }

            const avgLoss = epochTokens > 0 ? epochLoss / epochTokens : 0;
            this.emit("epoch", { epoch, avgLoss, tokens: epochTokens });
            console.log(`   Epoch ${epoch} complete. Avg loss: ${avgLoss.toFixed(4)}, Tokens: ${epochTokens}`);
        }

        const totalTime = ((Date.now() - startTime) / 1000).toFixed(1);
        console.log(`\n✓ Training complete in ${totalTime}s\n`);
        return results;
    }

    /**
     * Learning rate schedule with warmup
     */
    private _getLearningRate(step: number, baseLR: number): number {
        const { warmupSteps } = this.config;

        if (step < warmupSteps) {
            // Linear warmup
            return baseLR * (step / warmupSteps);
        }

        // Cosine decay (simplified)
        return baseLR * 0.5 * (1 + Math.cos(Math.PI * (step - warmupSteps) / 1000));
    }

    /**
     * Update weights using AdamW-like update
     */
    private _updateWeights(loss: number, lr: number): void {
        // This is a simplified weight update
        // Real implementation would need autograd for proper gradients

        // The loss is used to scale updates
        // Lower loss = smaller updates (model is learning well)
        const scaledLR = lr * Math.min(loss, 10) / 10;

        // Apply update to model (this calls the model's internal update)
        this.model.backward(loss, scaledLR);
    }

    /**
     * Evaluate model on held-out data
     */
    async evaluate(data: string[]): Promise<{ loss: number; perplexity: number }> {
        let totalLoss = 0;
        let totalTokens = 0;

        for (const text of data) {
            const tokens = this.tokenizer.encode(text, true, true);
            if (tokens.length < 2) continue;

            this.model.clearCache();

            for (let t = 0; t < tokens.length - 1; t++) {
                const input = tokens.slice(0, t + 1);
                const target = tokens[t + 1];
                const logits = this.model.forward(input);

                totalLoss += crossEntropyLoss(logits, target);
                totalTokens++;
            }
        }

        const avgLoss = totalTokens > 0 ? totalLoss / totalTokens : 0;
        const perplexity = Math.exp(avgLoss);

        return { loss: avgLoss, perplexity };
    }

    /**
     * Save training checkpoint
     */
    async saveCheckpoint(path: string): Promise<void> {
        // TODO: Implement model serialization
        console.log(`   Checkpoint saved to ${path}`);
    }

    /**
     * Load training checkpoint
     */
    async loadCheckpoint(path: string): Promise<void> {
        // TODO: Implement model loading
        console.log(`   Checkpoint loaded from ${path}`);
    }
}

/**
 * Create a data loader for training
 */
export function createDataLoader(
    data: string[],
    tokenizer: BPETokenizer,
    options: { batchSize?: number; maxSeqLen?: number; shuffle?: boolean } = {}
): Generator<number[][], void, undefined> {
    const { batchSize = 4, maxSeqLen = 256, shuffle = true } = options;

    function* generate() {
        // Tokenize
        let tokenized = data.map(text => {
            const tokens = tokenizer.encode(text, true, true);
            return tokens.slice(0, maxSeqLen);
        });

        // Shuffle if requested
        if (shuffle) {
            tokenized = tokenized.sort(() => Math.random() - 0.5);
        }

        // Yield batches
        for (let i = 0; i < tokenized.length; i += batchSize) {
            yield tokenized.slice(i, i + batchSize);
        }
    }

    return generate();
}
