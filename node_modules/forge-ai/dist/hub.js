"use strict";
/**
 * Forge Model Hub
 * =================
 *
 * Download and manage pre-trained models.
 */
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.Hub = exports.CACHE_DIR = exports.MODELS = void 0;
exports.listModels = listModels;
exports.isModelCached = isModelCached;
exports.getModelPath = getModelPath;
exports.downloadModel = downloadModel;
exports.loadModelConfig = loadModelConfig;
exports.deleteModel = deleteModel;
exports.clearCache = clearCache;
exports.getCacheSize = getCacheSize;
const https = __importStar(require("https"));
const fs = __importStar(require("fs"));
const path = __importStar(require("path"));
// Model registry
exports.MODELS = {
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
exports.CACHE_DIR = path.join(process.env.HOME || process.env.USERPROFILE || ".", ".forge", "models");
/**
 * List available models
 */
function listModels() {
    console.log("\n🔥 Forge Model Hub\n");
    console.log("Available models:\n");
    for (const [name, info] of Object.entries(exports.MODELS)) {
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
function isModelCached(name) {
    const modelPath = path.join(exports.CACHE_DIR, name, "model.onnx");
    return fs.existsSync(modelPath);
}
/**
 * Get model path
 */
function getModelPath(name) {
    return path.join(exports.CACHE_DIR, name, "model.onnx");
}
/**
 * Download model
 */
async function downloadModel(name, options = {}) {
    const model = exports.MODELS[name];
    if (!model) {
        throw new Error(`Unknown model: ${name}. Use listModels() to see available models.`);
    }
    const modelDir = path.join(exports.CACHE_DIR, name);
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
function downloadWithProgress(url, dest) {
    return new Promise((resolve, reject) => {
        const file = fs.createWriteStream(dest);
        https.get(url, (response) => {
            // Handle redirects
            if (response.statusCode === 301 || response.statusCode === 302) {
                file.close();
                fs.unlinkSync(dest);
                return downloadWithProgress(response.headers.location, dest)
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
            response.on("data", (chunk) => {
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
function loadModelConfig(name) {
    const configPath = path.join(exports.CACHE_DIR, name, "config.json");
    if (fs.existsSync(configPath)) {
        return JSON.parse(fs.readFileSync(configPath, "utf-8"));
    }
    return exports.MODELS[name]?.config || null;
}
/**
 * Delete cached model
 */
function deleteModel(name) {
    const modelDir = path.join(exports.CACHE_DIR, name);
    if (fs.existsSync(modelDir)) {
        fs.rmSync(modelDir, { recursive: true });
        console.log(`✓ Deleted ${name}`);
    }
    else {
        console.log(`Model ${name} not cached`);
    }
}
/**
 * Clear all cached models
 */
function clearCache() {
    if (fs.existsSync(exports.CACHE_DIR)) {
        fs.rmSync(exports.CACHE_DIR, { recursive: true });
        console.log("✓ Cache cleared");
    }
}
/**
 * Get cache size
 */
function getCacheSize() {
    if (!fs.existsSync(exports.CACHE_DIR)) {
        return 0;
    }
    let total = 0;
    function walkDir(dir) {
        const entries = fs.readdirSync(dir, { withFileTypes: true });
        for (const entry of entries) {
            const fullPath = path.join(dir, entry.name);
            if (entry.isFile()) {
                total += fs.statSync(fullPath).size;
            }
            else if (entry.isDirectory()) {
                walkDir(fullPath);
            }
        }
    }
    walkDir(exports.CACHE_DIR);
    return total;
}
/**
 * Hub class for programmatic access
 */
class Hub {
    static list = listModels;
    static download = downloadModel;
    static delete = deleteModel;
    static clear = clearCache;
    static getCacheSize = getCacheSize;
    static isModelCached = isModelCached;
    static getModelPath = getModelPath;
    static loadModelConfig = loadModelConfig;
    static MODELS = exports.MODELS;
    static CACHE_DIR = exports.CACHE_DIR;
}
exports.Hub = Hub;
exports.default = Hub;
//# sourceMappingURL=hub.js.map