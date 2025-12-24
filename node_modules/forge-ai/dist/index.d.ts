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
import { Model, ModelConfig } from "./model";
import { TrainConfig } from "./trainer";
import { GenerateConfig } from "./generator";
import { Middleware } from "./middleware";
import { RouteHandler } from "./server";
import { RAG } from "./rag";
import { Agent, AgentConfig } from "./agent";
export * from "./model";
export * from "./tokenizer";
export * from "./trainer";
export * from "./generator";
export * from "./middleware";
export * from "./server";
export * from "./rag";
export * from "./agent";
export * from "./tensor";
export * from "./transformer";
export * from "./bpe_tokenizer";
export * from "./backend";
export interface ForgeConfig {
    model?: string;
    modelConfig?: Partial<ModelConfig>;
}
export declare class Forge extends EventEmitter {
    private _model;
    private _generator;
    private _trainer;
    private _server;
    private _middlewares;
    constructor(config?: ForgeConfig);
    /**
     * Set or get the model
     */
    model(name?: string): this | Model;
    /**
     * Add middleware
     */
    use(middleware: Middleware): this;
    /**
     * Train the model
     */
    train(data: string[], config?: TrainConfig): Promise<this>;
    /**
     * Generate text
     */
    generate(prompt: string, config?: GenerateConfig): Promise<string>;
    /**
     * Stream generation
     */
    stream(prompt: string, config?: GenerateConfig): AsyncGenerator<string>;
    /**
     * Create a RAG instance
     */
    rag(): RAG;
    /**
     * Create an Agent instance
     */
    agent(config?: AgentConfig): Agent;
    get(path: string, handler: RouteHandler): this;
    post(path: string, handler: RouteHandler): this;
    put(path: string, handler: RouteHandler): this;
    delete(path: string, handler: RouteHandler): this;
    listen(port: number, callback?: () => void): this;
    private runMiddleware;
}
/**
 * Create a Forge app
 *
 * @example
 * const app = forge();
 * const app = forge({ model: "7b" });
 */
export declare function forge(config?: ForgeConfig | string): Forge;
export { logger, timer, normalize, lowercase, cache, rateLimit, compose, } from "./middleware";
export { createTool, calculatorTool, dateTool, } from "./agent";
export default forge;
//# sourceMappingURL=index.d.ts.map