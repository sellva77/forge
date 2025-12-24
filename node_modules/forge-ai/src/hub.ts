/**
 * Forge Model Hub
 * =================
 * 
 * Download and manage pre-trained models.
 */

import * as https from "https";
import * as fs from "fs";
import * as path from "path";
import * as nodeCrypto from "crypto";

// Type definitions
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

// Model registry
export const MODELS: Record<string, ModelInfo> = {
    // Llama style models
    "llama-tiny": {
        url: "https://huggingface.co/forge-ai/llama-tiny/resolve/main/model.onnx",
        size: "50MB",
        params: "17M",
        description: "Tiny Llama for testing",
        config: { dim: 128, layers: 4, heads: 4, vocabSize: 32000 },
    },
    "llama-small": {
        url: "https://huggingface.co/forge-ai/llama-small/resolve/main/model.onnx",
        size: "200MB",
        params: "50M",
        description: "Small Llama for prototyping",
        config: { dim: 256, layers: 6, heads: 8, vocabSize: 32000 },
    },
    "llama-medium": {
        url: "https://huggingface.co/forge-ai/llama-medium/resolve/main/model.onnx",
        size: "600MB",
        params: "150M",
        description: "Medium Llama for production",
        config: { dim: 512, layers: 12, heads: 8, vocabSize: 32000 },
    },

    // Code models
    "codegen-small": {
        url: "https://huggingface.co/forge-ai/codegen-small/resolve/main/model.onnx",
        size: "300MB",
        params: "75M",
        description: "Small code generation model",
        config: { dim: 384, layers: 8, heads: 6, vocabSize: 50000 },
    },

    // Embedding models
    "embed-small": {
        url: "https://huggingface.co/forge-ai/embed-small/resolve/main/model.onnx",
        size: "100MB",
        params: "25M",
        description: "Small embedding model for RAG",
        config: { dim: 384, layers: 4, heads: 6, vocabSize: 32000 },
    },
};

// Model cache directory
export const CACHE_DIR = path.join(
    process.env.HOME || process.env.USERPROFILE || ".",
    ".forge",
    "models"
);

/**
 * List available models
 */
export function listModels(): void {
    console.log("\n🔥 Forge Model Hub\n");
    console.log("Available models:\n");

    for (const [name, info] of Object.entries(MODELS)) {
        const cached = isModelCached(name) ? "✓ cached" : "";
        console.log(`  ${name}`);
        console.log(`    ${info.description}`);
        console.log(`    Size: ${info.size} | Params: ${info.params} ${cached}`);
        console.log("");
    }
}

/**
 * Check if model is cached
 */
export function isModelCached(name: string): boolean {
    const modelPath = path.join(CACHE_DIR, name, "model.onnx");
    return fs.existsSync(modelPath);
}

/**
 * Get model path
 */
export function getModelPath(name: string): string {
    return path.join(CACHE_DIR, name, "model.onnx");
}

/**
 * Download model
 */
export async function downloadModel(name: string, options: DownloadOptions = {}): Promise<string> {
    const model = MODELS[name];
    if (!model) {
        throw new Error(`Unknown model: ${name}. Use listModels() to see available models.`);
    }

    const modelDir = path.join(CACHE_DIR, name);
    const modelPath = path.join(modelDir, "model.onnx");
    const configPath = path.join(modelDir, "config.json");

    // Check if already cached
    if (!options.force && fs.existsSync(modelPath)) {
        console.log(`✓ Model ${name} already cached at ${modelPath}`);
        return modelPath;
    }

    // Create directory
    fs.mkdirSync(modelDir, { recursive: true });

    console.log(`\n📥 Downloading ${name}...`);
    console.log(`   Size: ${model.size}`);
    console.log(`   From: ${model.url}\n`);

    // Download with progress
    await downloadWithProgress(model.url, modelPath);

    // Save config
    fs.writeFileSync(configPath, JSON.stringify(model.config, null, 2));

    console.log(`\n✓ Model saved to ${modelPath}\n`);
    return modelPath;
}

/**
 * Download with progress bar
 */
function downloadWithProgress(url: string, dest: string): Promise<void> {
    return new Promise((resolve, reject) => {
        const file = fs.createWriteStream(dest);

        https.get(url, (response) => {
            // Handle redirects
            if (response.statusCode === 301 || response.statusCode === 302) {
                file.close();
                fs.unlinkSync(dest);
                return downloadWithProgress(response.headers.location!, dest)
                    .then(resolve)
                    .catch(reject);
            }

            if (response.statusCode !== 200) {
                file.close();
                fs.unlinkSync(dest);
                return reject(new Error(`HTTP ${response.statusCode}`));
            }

            const total = parseInt(response.headers["content-length"] || "0", 10);
            let downloaded = 0;
            let lastPercent = 0;

            response.on("data", (chunk: Buffer) => {
                downloaded += chunk.length;
                const percent = Math.floor((downloaded / total) * 100);

                if (percent > lastPercent) {
                    const bar = "█".repeat(Math.floor(percent / 2)) + "░".repeat(50 - Math.floor(percent / 2));
                    process.stdout.write(`\r   ${bar} ${percent}%`);
                    lastPercent = percent;
                }
            });

            response.pipe(file);

            file.on("finish", () => {
                file.close();
                resolve();
            });
        }).on("error", (err) => {
            file.close();
            fs.unlinkSync(dest);
            reject(err);
        });
    });
}

/**
 * Load model config
 */
export function loadModelConfig(name: string): Record<string, number> | null {
    const configPath = path.join(CACHE_DIR, name, "config.json");

    if (fs.existsSync(configPath)) {
        return JSON.parse(fs.readFileSync(configPath, "utf-8"));
    }

    return MODELS[name]?.config || null;
}

/**
 * Delete cached model
 */
export function deleteModel(name: string): void {
    const modelDir = path.join(CACHE_DIR, name);

    if (fs.existsSync(modelDir)) {
        fs.rmSync(modelDir, { recursive: true });
        console.log(`✓ Deleted ${name}`);
    } else {
        console.log(`Model ${name} not cached`);
    }
}

/**
 * Clear all cached models
 */
export function clearCache(): void {
    if (fs.existsSync(CACHE_DIR)) {
        fs.rmSync(CACHE_DIR, { recursive: true });
        console.log("✓ Cache cleared");
    }
}

/**
 * Get cache size
 */
export function getCacheSize(): number {
    if (!fs.existsSync(CACHE_DIR)) {
        return 0;
    }

    let total = 0;

    function walkDir(dir: string): void {
        const entries = fs.readdirSync(dir, { withFileTypes: true });
        for (const entry of entries) {
            const fullPath = path.join(dir, entry.name);
            if (entry.isFile()) {
                total += fs.statSync(fullPath).size;
            } else if (entry.isDirectory()) {
                walkDir(fullPath);
            }
        }
    }

    walkDir(CACHE_DIR);
    return total;
}

/**
 * Hub class for programmatic access
 */
export class Hub {
    static list = listModels;
    static download = downloadModel;
    static delete = deleteModel;
    static clear = clearCache;
    static getCacheSize = getCacheSize;
    static isModelCached = isModelCached;
    static getModelPath = getModelPath;
    static loadModelConfig = loadModelConfig;
    static MODELS = MODELS;
    static CACHE_DIR = CACHE_DIR;
}

export default Hub;
