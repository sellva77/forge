"use strict";
/**
 * Forge - Agent System
 * =====================
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.dateTool = exports.calculatorTool = exports.Agent = void 0;
exports.createTool = createTool;
const generator_1 = require("./generator");
class Agent {
    generator;
    tools = new Map();
    memory = [];
    config;
    constructor(model, config = {}) {
        this.generator = new generator_1.Generator(model);
        this.config = {
            maxIterations: config.maxIterations || 5,
            verbose: config.verbose ?? true,
        };
    }
    /**
     * Add a tool to the agent
     */
    tool(tool) {
        this.tools.set(tool.name, tool);
        return this;
    }
    /**
     * Run the agent with a task
     */
    async run(task) {
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
                }
                catch (error) {
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
    getMemory() {
        return [...this.memory];
    }
    /**
     * Clear memory
     */
    clearMemory() {
        this.memory = [];
        return this;
    }
    /**
     * List available tools
     */
    listTools() {
        return Array.from(this.tools.keys());
    }
}
exports.Agent = Agent;
/**
 * Create a tool easily
 */
function createTool(name, description, fn) {
    return { name, description, fn };
}
// Built-in tools
exports.calculatorTool = {
    name: "calculator",
    description: "Calculate mathematical expressions",
    fn: (expr) => {
        const match = expr.match(/[\d+\-*/().]+/);
        if (match) {
            try {
                return eval(match[0]);
            }
            catch {
                return "Invalid expression";
            }
        }
        return "No expression found";
    },
};
exports.dateTool = {
    name: "date",
    description: "Get current date and time",
    fn: () => new Date().toISOString(),
};
//# sourceMappingURL=agent.js.map