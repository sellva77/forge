"use strict";
/**
 * Forge - AI Framework for JavaScript
 * ====================================
 *
 * Express-style API for building AI.
 *
 * @example
 * const app = forge();
 *
 * app.model("7b");
 * app.use(logger);
 *
 * await app.train(data);
 *
 * app.post("/chat", async (req, res) => {
 *   const output = await app.generate(req.body.message);
 *   res.json({ output });
 * });
 *
 * app.listen(3000);
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
var __exportStar = (this && this.__exportStar) || function(m, exports) {
    for (var p in m) if (p !== "default" && !Object.prototype.hasOwnProperty.call(exports, p)) __createBinding(exports, m, p);
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.dateTool = exports.calculatorTool = exports.createTool = exports.compose = exports.rateLimit = exports.cache = exports.lowercase = exports.normalize = exports.timer = exports.logger = exports.Forge = void 0;
exports.forge = forge;
const events_1 = require("events");
const model_1 = require("./model");
const trainer_1 = require("./trainer");
const generator_1 = require("./generator");
const server_1 = require("./server");
const rag_1 = require("./rag");
const agent_1 = require("./agent");
// Re-export everything
__exportStar(require("./model"), exports);
__exportStar(require("./tokenizer"), exports);
__exportStar(require("./trainer"), exports);
__exportStar(require("./generator"), exports);
__exportStar(require("./middleware"), exports);
__exportStar(require("./server"), exports);
__exportStar(require("./rag"), exports);
__exportStar(require("./agent"), exports);
// New real ML modules
__exportStar(require("./tensor"), exports);
__exportStar(require("./transformer"), exports);
__exportStar(require("./bpe_tokenizer"), exports);
// Backend abstraction (auto-selects best available)
__exportStar(require("./backend"), exports);
class Forge extends events_1.EventEmitter {
    _model;
    _generator;
    _trainer;
    _server;
    _middlewares = [];
    constructor(config = {}) {
        super();
        this._model = new model_1.Model(config.model || "small");
        this._generator = new generator_1.Generator(this._model);
        this._trainer = new trainer_1.Trainer(this._model);
        this._server = new server_1.Server();
    }
    // ---------------------------------------------------------------------------
    // MODEL
    // ---------------------------------------------------------------------------
    /**
     * Set or get the model
     */
    model(name) {
        if (name) {
            this._model = new model_1.Model(name);
            this._generator = new generator_1.Generator(this._model);
            this._trainer = new trainer_1.Trainer(this._model);
            return this;
        }
        return this._model;
    }
    // ---------------------------------------------------------------------------
    // MIDDLEWARE
    // ---------------------------------------------------------------------------
    /**
     * Add middleware
     */
    use(middleware) {
        this._middlewares.push(middleware);
        return this;
    }
    // ---------------------------------------------------------------------------
    // TRAINING
    // ---------------------------------------------------------------------------
    /**
     * Train the model
     */
    async train(data, config) {
        this._trainer = new trainer_1.Trainer(this._model, config);
        this._trainer.on("step", (result) => this.emit("step", result));
        this._trainer.on("epoch", (result) => this.emit("epoch", result));
        await this._trainer.train(data);
        return this;
    }
    // ---------------------------------------------------------------------------
    // GENERATION
    // ---------------------------------------------------------------------------
    /**
     * Generate text
     */
    async generate(prompt, config) {
        const ctx = { input: prompt };
        await this.runMiddleware(ctx, async () => {
            ctx.output = await this._generator.generate(prompt, config);
        });
        return ctx.output || "";
    }
    /**
     * Stream generation
     */
    async *stream(prompt, config) {
        for await (const token of this._generator.stream(prompt, config)) {
            yield token;
        }
    }
    // ---------------------------------------------------------------------------
    // RAG
    // ---------------------------------------------------------------------------
    /**
     * Create a RAG instance
     */
    rag() {
        return new rag_1.RAG(this._model);
    }
    // ---------------------------------------------------------------------------
    // AGENT
    // ---------------------------------------------------------------------------
    /**
     * Create an Agent instance
     */
    agent(config) {
        return new agent_1.Agent(this._model, config);
    }
    // ---------------------------------------------------------------------------
    // SERVER
    // ---------------------------------------------------------------------------
    get(path, handler) {
        this._server.get(path, handler);
        return this;
    }
    post(path, handler) {
        this._server.post(path, handler);
        return this;
    }
    put(path, handler) {
        this._server.put(path, handler);
        return this;
    }
    delete(path, handler) {
        this._server.delete(path, handler);
        return this;
    }
    listen(port, callback) {
        // Add default routes
        this._server.get("/health", (req, res) => {
            res.json({ status: "ok", model: this._model.name });
        });
        this._server.post("/generate", async (req, res) => {
            try {
                const output = await this.generate(req.body.prompt, req.body);
                res.json({ output });
            }
            catch (error) {
                res.status(500).json({ error: error.message });
            }
        });
        this._server.listen(port, callback);
        return this;
    }
    // ---------------------------------------------------------------------------
    // PRIVATE
    // ---------------------------------------------------------------------------
    async runMiddleware(ctx, fn) {
        let index = 0;
        const next = async () => {
            if (index < this._middlewares.length) {
                await this._middlewares[index++](ctx, next);
            }
            else {
                await fn();
            }
        };
        await next();
    }
}
exports.Forge = Forge;
// =============================================================================
// FACTORY FUNCTION
// =============================================================================
/**
 * Create a Forge app
 *
 * @example
 * const app = forge();
 * const app = forge({ model: "7b" });
 */
function forge(config) {
    if (typeof config === "string") {
        return new Forge({ model: config });
    }
    return new Forge(config);
}
// =============================================================================
// MIDDLEWARE EXPORTS
// =============================================================================
var middleware_1 = require("./middleware");
Object.defineProperty(exports, "logger", { enumerable: true, get: function () { return middleware_1.logger; } });
Object.defineProperty(exports, "timer", { enumerable: true, get: function () { return middleware_1.timer; } });
Object.defineProperty(exports, "normalize", { enumerable: true, get: function () { return middleware_1.normalize; } });
Object.defineProperty(exports, "lowercase", { enumerable: true, get: function () { return middleware_1.lowercase; } });
Object.defineProperty(exports, "cache", { enumerable: true, get: function () { return middleware_1.cache; } });
Object.defineProperty(exports, "rateLimit", { enumerable: true, get: function () { return middleware_1.rateLimit; } });
Object.defineProperty(exports, "compose", { enumerable: true, get: function () { return middleware_1.compose; } });
// =============================================================================
// TOOL EXPORTS
// =============================================================================
var agent_2 = require("./agent");
Object.defineProperty(exports, "createTool", { enumerable: true, get: function () { return agent_2.createTool; } });
Object.defineProperty(exports, "calculatorTool", { enumerable: true, get: function () { return agent_2.calculatorTool; } });
Object.defineProperty(exports, "dateTool", { enumerable: true, get: function () { return agent_2.dateTool; } });
// =============================================================================
// DEFAULT EXPORT
// =============================================================================
exports.default = forge;
//# sourceMappingURL=index.js.map