/**
 * Forge - Agent System
 * =====================
 */
import { Model } from "./model";
export interface Tool {
    name: string;
    description: string;
    parameters?: Record<string, any>;
    fn: (...args: any[]) => any | Promise<any>;
}
export interface AgentConfig {
    maxIterations?: number;
    verbose?: boolean;
}
export interface AgentStep {
    thought?: string;
    action?: string;
    actionInput?: string;
    observation?: string;
}
export declare class Agent {
    private generator;
    private tools;
    private memory;
    private config;
    constructor(model: Model, config?: AgentConfig);
    /**
     * Add a tool to the agent
     */
    tool(tool: Tool): this;
    /**
     * Run the agent with a task
     */
    run(task: string): Promise<string>;
    /**
     * Get conversation memory
     */
    getMemory(): AgentStep[];
    /**
     * Clear memory
     */
    clearMemory(): this;
    /**
     * List available tools
     */
    listTools(): string[];
}
/**
 * Create a tool easily
 */
export declare function createTool(name: string, description: string, fn: (...args: any[]) => any): Tool;
export declare const calculatorTool: Tool;
export declare const dateTool: Tool;
//# sourceMappingURL=agent.d.ts.map