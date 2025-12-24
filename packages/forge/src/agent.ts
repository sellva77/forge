/**
 * Forge - Agent System
 * =====================
 */

import { Model } from "./model";
import { Generator } from "./generator";

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

export class Agent {
    private generator: Generator;
    private tools: Map<string, Tool> = new Map();
    private memory: AgentStep[] = [];
    private config: Required<AgentConfig>;

    constructor(model: Model, config: AgentConfig = {}) {
        this.generator = new Generator(model);
        this.config = {
            maxIterations: config.maxIterations || 5,
            verbose: config.verbose ?? true,
        };
    }

    /**
     * Add a tool to the agent
     */
    tool(tool: Tool): this {
        this.tools.set(tool.name, tool);
        return this;
    }

    /**
     * Run the agent with a task
     */
    async run(task: string): Promise<string> {
        if (this.config.verbose) {
            console.log(`\n🤖 Agent: "${task}"\n`);
        }

        // Check if task matches a tool directly
        for (const [name, tool] of this.tools) {
            if (task.toLowerCase().includes(name.toLowerCase())) {
                try {
                    const result = await tool.fn(task);

                    if (this.config.verbose) {
                        console.log(`   Tool: ${name}`);
                        console.log(`   Result: ${result}\n`);
                    }

                    this.memory.push({
                        action: name,
                        actionInput: task,
                        observation: String(result),
                    });

                    return String(result);
                } catch (error: any) {
                    return `Error: ${error.message}`;
                }
            }
        }

        // No tool matched, use model
        const response = await this.generator.generate(task);

        if (this.config.verbose) {
            console.log(`   Response: ${response}\n`);
        }

        this.memory.push({
            thought: task,
            observation: response,
        });

        return response;
    }

    /**
     * Get conversation memory
     */
    getMemory(): AgentStep[] {
        return [...this.memory];
    }

    /**
     * Clear memory
     */
    clearMemory(): this {
        this.memory = [];
        return this;
    }

    /**
     * List available tools
     */
    listTools(): string[] {
        return Array.from(this.tools.keys());
    }
}

/**
 * Create a tool easily
 */
export function createTool(
    name: string,
    description: string,
    fn: (...args: any[]) => any
): Tool {
    return { name, description, fn };
}

// Built-in tools
export const calculatorTool: Tool = {
    name: "calculator",
    description: "Calculate mathematical expressions",
    fn: (expr: string) => {
        const match = expr.match(/[\d+\-*/().]+/);
        if (match) {
            try {
                return eval(match[0]);
            } catch {
                return "Invalid expression";
            }
        }
        return "No expression found";
    },
};

export const dateTool: Tool = {
    name: "date",
    description: "Get current date and time",
    fn: () => new Date().toISOString(),
};
