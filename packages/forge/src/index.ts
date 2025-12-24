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

import { EventEmitter } from "events";
import { Model, PRESETS, ModelConfig } from "./model";
import { Tokenizer } from "./tokenizer";
import { Trainer, TrainConfig } from "./trainer";
import { Generator, GenerateConfig } from "./generator";
import { Context, Middleware, NextFn } from "./middleware";
import { Server, Request, Response, RouteHandler } from "./server";
import { RAG, RAGConfig } from "./rag";
import { Agent, Tool, AgentConfig, createTool } from "./agent";

// Re-export everything
export * from "./model";
export * from "./tokenizer";
export * from "./trainer";
export * from "./generator";
export * from "./middleware";
export * from "./server";
export * from "./rag";
export * from "./agent";

// New real ML modules
export * from "./tensor";
export * from "./transformer";
export * from "./bpe_tokenizer";

// Backend abstraction (auto-selects best available)
export * from "./backend";

// =============================================================================
// FORGE APP
// =============================================================================

export interface ForgeConfig {
    model?: string;
    modelConfig?: Partial<ModelConfig>;
}

export class Forge extends EventEmitter {
    private _model: Model;
    private _generator: Generator;
    private _trainer: Trainer;
    private _server: Server;
    private _middlewares: Middleware[] = [];

    constructor(config: ForgeConfig = {}) {
        super();
        this._model = new Model(config.model || "small");
        this._generator = new Generator(this._model);
        this._trainer = new Trainer(this._model);
        this._server = new Server();
    }

    // ---------------------------------------------------------------------------
    // MODEL
    // ---------------------------------------------------------------------------

    /**
     * Set or get the model
     */
    model(name?: string): this | Model {
        if (name) {
            this._model = new Model(name);
            this._generator = new Generator(this._model);
            this._trainer = new Trainer(this._model);
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
    use(middleware: Middleware): this {
        this._middlewares.push(middleware);
        return this;
    }

    // ---------------------------------------------------------------------------
    // TRAINING
    // ---------------------------------------------------------------------------

    /**
     * Train the model
     */
    async train(data: string[], config?: TrainConfig): Promise<this> {
        this._trainer = new Trainer(this._model, config);

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
    async generate(prompt: string, config?: GenerateConfig): Promise<string> {
        const ctx: Context = { input: prompt };

        await this.runMiddleware(ctx, async () => {
            ctx.output = await this._generator.generate(prompt, config);
        });

        return ctx.output || "";
    }

    /**
     * Stream generation
     */
    async *stream(prompt: string, config?: GenerateConfig): AsyncGenerator<string> {
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
    rag(): RAG {
        return new RAG(this._model);
    }

    // ---------------------------------------------------------------------------
    // AGENT
    // ---------------------------------------------------------------------------

    /**
     * Create an Agent instance
     */
    agent(config?: AgentConfig): Agent {
        return new Agent(this._model, config);
    }

    // ---------------------------------------------------------------------------
    // SERVER
    // ---------------------------------------------------------------------------

    get(path: string, handler: RouteHandler): this {
        this._server.get(path, handler);
        return this;
    }

    post(path: string, handler: RouteHandler): this {
        this._server.post(path, handler);
        return this;
    }

    put(path: string, handler: RouteHandler): this {
        this._server.put(path, handler);
        return this;
    }

    delete(path: string, handler: RouteHandler): this {
        this._server.delete(path, handler);
        return this;
    }

    listen(port: number, callback?: () => void): this {
        // Add default routes
        this._server.get("/health", (req, res) => {
            res.json({ status: "ok", model: this._model.name });
        });

        this._server.post("/generate", async (req, res) => {
            try {
                const output = await this.generate(req.body.prompt, req.body);
                res.json({ output });
            } catch (error: any) {
                res.status(500).json({ error: error.message });
            }
        });

        this._server.listen(port, callback);
        return this;
    }

    // ---------------------------------------------------------------------------
    // PRIVATE
    // ---------------------------------------------------------------------------

    private async runMiddleware(ctx: Context, fn: () => Promise<void>): Promise<void> {
        let index = 0;

        const next: NextFn = async () => {
            if (index < this._middlewares.length) {
                await this._middlewares[index++](ctx, next);
            } else {
                await fn();
            }
        };

        await next();
    }
}

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
export function forge(config?: ForgeConfig | string): Forge {
    if (typeof config === "string") {
        return new Forge({ model: config });
    }
    return new Forge(config);
}

// =============================================================================
// MIDDLEWARE EXPORTS
// =============================================================================

export {
    logger,
    timer,
    normalize,
    lowercase,
    cache,
    rateLimit,
    compose,
} from "./middleware";

// =============================================================================
// TOOL EXPORTS
// =============================================================================

export {
    createTool,
    calculatorTool,
    dateTool,
} from "./agent";

// =============================================================================
// DEFAULT EXPORT
// =============================================================================

export default forge;
