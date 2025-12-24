/**
 * Forge Model Hub
 * =================
 *
 * Download and manage pre-trained models.
 */
interface ModelInfo {
    url: string;
    size: string;
    params: string;
    description: string;
    config: {
        dim: number;
        layers: number;
        heads: number;
        vocabSize: number;
    };
}
interface DownloadOptions {
    force?: boolean;
}
export declare const MODELS: Record<string, ModelInfo>;
export declare const CACHE_DIR: string;
/**
 * List available models
 */
export declare function listModels(): void;
/**
 * Check if model is cached
 */
export declare function isModelCached(name: string): boolean;
/**
 * Get model path
 */
export declare function getModelPath(name: string): string;
/**
 * Download model
 */
export declare function downloadModel(name: string, options?: DownloadOptions): Promise<string>;
/**
 * Load model config
 */
export declare function loadModelConfig(name: string): Record<string, number> | null;
/**
 * Delete cached model
 */
export declare function deleteModel(name: string): void;
/**
 * Clear all cached models
 */
export declare function clearCache(): void;
/**
 * Get cache size
 */
export declare function getCacheSize(): number;
/**
 * Hub class for programmatic access
 */
export declare class Hub {
    static list: typeof listModels;
    static download: typeof downloadModel;
    static delete: typeof deleteModel;
    static clear: typeof clearCache;
    static getCacheSize: typeof getCacheSize;
    static isModelCached: typeof isModelCached;
    static getModelPath: typeof getModelPath;
    static loadModelConfig: typeof loadModelConfig;
    static MODELS: Record<string, ModelInfo>;
    static CACHE_DIR: string;
}
export default Hub;
//# sourceMappingURL=hub.d.ts.map