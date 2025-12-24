/**
 * Forge - Trainer
 * ================
 *
 * Real training with cross-entropy loss and gradient updates.
 */
import { EventEmitter } from "events";
import { Model } from "./model";
import { BPETokenizer } from "./bpe_tokenizer";
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
export declare class Trainer extends EventEmitter {
    private model;
    private tokenizer;
    private config;
    private m;
    private v;
    private t;
    private beta1;
    private beta2;
    private eps;
    constructor(model: Model, config?: TrainConfig);
    /**
     * Train the model on text data
     */
    train(data: string[]): Promise<TrainResult[]>;
    /**
     * Learning rate schedule with warmup
     */
    private _getLearningRate;
    /**
     * Update weights using AdamW-like update
     */
    private _updateWeights;
    /**
     * Evaluate model on held-out data
     */
    evaluate(data: string[]): Promise<{
        loss: number;
        perplexity: number;
    }>;
    /**
     * Save training checkpoint
     */
    saveCheckpoint(path: string): Promise<void>;
    /**
     * Load training checkpoint
     */
    loadCheckpoint(path: string): Promise<void>;
}
/**
 * Create a data loader for training
 */
export declare function createDataLoader(data: string[], tokenizer: BPETokenizer, options?: {
    batchSize?: number;
    maxSeqLen?: number;
    shuffle?: boolean;
}): Generator<number[][], void, undefined>;
//# sourceMappingURL=trainer.d.ts.map