#!/usr/bin/env node
/**
 * 🔥 Forge CLI
 * ============
 * Command-line interface for the Forge AI framework.
 * 
 * Commands:
 *   forge new <name>        Create a new Forge project
 *   forge generate <prompt> Generate text from a prompt
 *   forge train <file>      Train a model from data
 *   forge serve             Start an API server
 *   forge info              Show system & backend info
 */

const { program } = require("commander");
const fs = require("fs");
const path = require("path");

// Package version
const pkg = require("../package.json");

// Colors for terminal output
const colors = {
    reset: "\x1b[0m",
    bright: "\x1b[1m",
    dim: "\x1b[2m",
    red: "\x1b[31m",
    green: "\x1b[32m",
    yellow: "\x1b[33m",
    blue: "\x1b[34m",
    magenta: "\x1b[35m",
    cyan: "\x1b[36m",
};

const log = {
    info: (msg) => console.log(`${colors.blue}ℹ${colors.reset} ${msg}`),
    success: (msg) => console.log(`${colors.green}✓${colors.reset} ${msg}`),
    warn: (msg) => console.log(`${colors.yellow}⚠${colors.reset} ${msg}`),
    error: (msg) => console.error(`${colors.red}✗${colors.reset} ${msg}`),
    fire: (msg) => console.log(`${colors.bright}🔥${colors.reset} ${msg}`),
};

// ASCII Banner
const banner = `
${colors.bright}${colors.magenta}
  ╔═══════════════════════════════════════════╗
  ║   🔥  F O R G E   A I                     ║
  ║   ─────────────────────────               ║
  ║   AI Framework for JavaScript             ║
  ╚═══════════════════════════════════════════╝
${colors.reset}`;

program
    .name("forge")
    .description("🔥 AI Framework for JavaScript — Express-style API for AI")
    .version(pkg.version, "-v, --version", "Show version number")
    .addHelpText("before", banner);

// ═══════════════════════════════════════════════════════════════════════════
// NEW COMMAND - Create a new project
// ═══════════════════════════════════════════════════════════════════════════
program
    .command("new <name>")
    .alias("create")
    .alias("init")
    .description("Create a new Forge project")
    .option("-m, --model <model>", "Model preset (tiny|small|medium|large|7b)", "small")
    .option("-t, --template <template>", "Project template (basic|server|rag|agent)", "basic")
    .option("--typescript", "Use TypeScript", false)
    .option("--no-git", "Skip git initialization")
    .action((name, options) => {
        console.log(banner);
        log.fire(`Creating new Forge project: ${colors.cyan}${name}${colors.reset}`);

        const dir = path.join(process.cwd(), name);

        if (fs.existsSync(dir)) {
            log.error(`Directory "${name}" already exists`);
            process.exit(1);
        }

        fs.mkdirSync(dir, { recursive: true });

        // Determine file extension
        const ext = options.typescript ? "ts" : "js";
        const mainFile = options.typescript ? "src/index.ts" : "index.js";

        // package.json
        const pkgJson = {
            name,
            version: "1.0.0",
            description: `${name} - Built with Forge AI`,
            main: mainFile,
            type: "module",
            scripts: {
                start: options.typescript ? "tsx src/index.ts" : "node index.js",
                dev: options.typescript ? "tsx watch src/index.ts" : "node --watch index.js",
                train: options.typescript ? "tsx src/train.ts" : "node train.js",
                build: options.typescript ? "tsc" : undefined,
            },
            keywords: ["ai", "forge", "llm"],
            author: "",
            license: "MIT",
            dependencies: {
                "forge-ai": `^${pkg.version}`,
            },
            devDependencies: options.typescript ? {
                "typescript": "^5.6.0",
                "tsx": "^4.19.0",
                "@types/node": "^22.0.0",
            } : undefined,
        };

        // Remove undefined values
        Object.keys(pkgJson.scripts).forEach(key => {
            if (pkgJson.scripts[key] === undefined) delete pkgJson.scripts[key];
        });
        if (!pkgJson.devDependencies) delete pkgJson.devDependencies;

        fs.writeFileSync(
            path.join(dir, "package.json"),
            JSON.stringify(pkgJson, null, 2)
        );
        log.success("Created package.json");

        // Create src directory for TypeScript
        if (options.typescript) {
            fs.mkdirSync(path.join(dir, "src"), { recursive: true });
        }

        // Generate main file based on template
        let mainContent;
        switch (options.template) {
            case "server":
                mainContent = generateServerTemplate(options.model, options.typescript);
                break;
            case "rag":
                mainContent = generateRAGTemplate(options.model, options.typescript);
                break;
            case "agent":
                mainContent = generateAgentTemplate(options.model, options.typescript);
                break;
            default:
                mainContent = generateBasicTemplate(options.model, options.typescript);
        }

        fs.writeFileSync(path.join(dir, mainFile), mainContent);
        log.success(`Created ${mainFile}`);

        // train.js / train.ts
        const trainContent = generateTrainTemplate(options.model, options.typescript);
        const trainFile = options.typescript ? "src/train.ts" : "train.js";
        fs.writeFileSync(path.join(dir, trainFile), trainContent);
        log.success(`Created ${trainFile}`);

        // data.json - sample training data
        fs.writeFileSync(
            path.join(dir, "data.json"),
            JSON.stringify([
                "Forge is an AI framework for JavaScript",
                "It uses an Express-style API for building AI applications",
                "You can train models, generate text, and build chatbots",
                "Forge supports RAG, agents, and custom middleware",
            ], null, 2)
        );
        log.success("Created data.json");

        // .gitignore
        fs.writeFileSync(
            path.join(dir, ".gitignore"),
            `node_modules/
dist/
.env
*.log
.DS_Store
`
        );
        log.success("Created .gitignore");

        // TypeScript config
        if (options.typescript) {
            fs.writeFileSync(
                path.join(dir, "tsconfig.json"),
                JSON.stringify({
                    compilerOptions: {
                        target: "ES2022",
                        module: "NodeNext",
                        moduleResolution: "NodeNext",
                        strict: true,
                        esModuleInterop: true,
                        outDir: "./dist",
                        rootDir: "./src",
                        declaration: true,
                    },
                    include: ["src/**/*"],
                    exclude: ["node_modules", "dist"],
                }, null, 2)
            );
            log.success("Created tsconfig.json");
        }

        // README.md
        fs.writeFileSync(
            path.join(dir, "README.md"),
            `# ${name}

Built with 🔥 **Forge AI**

## Quick Start

\`\`\`bash
npm install
npm start
\`\`\`

## Commands

- \`npm start\` - Run the application
- \`npm run dev\` - Run with watch mode
- \`npm run train\` - Train the model

## Learn More

- [Forge Documentation](https://github.com/forge-ai/forge)
`
        );
        log.success("Created README.md");

        // Initialize git
        if (options.git !== false) {
            try {
                require("child_process").execSync("git init", { cwd: dir, stdio: "ignore" });
                log.success("Initialized git repository");
            } catch {
                log.warn("Could not initialize git repository");
            }
        }

        console.log(`
${colors.green}${colors.bright}✨ Project created successfully!${colors.reset}

  ${colors.dim}cd ${name}${colors.reset}
  ${colors.dim}npm install${colors.reset}
  ${colors.dim}npm start${colors.reset}

${colors.bright}Happy building! 🔥${colors.reset}
`);
    });

// ═══════════════════════════════════════════════════════════════════════════
// GENERATE COMMAND - Generate text
// ═══════════════════════════════════════════════════════════════════════════
program
    .command("generate <prompt>")
    .alias("gen")
    .alias("g")
    .description("Generate text from a prompt")
    .option("-m, --model <model>", "Model preset", "small")
    .option("-t, --tokens <n>", "Max tokens to generate", "50")
    .option("-T, --temperature <n>", "Sampling temperature", "0.8")
    .option("-s, --stream", "Stream output token by token", false)
    .action(async (prompt, options) => {
        try {
            const { forge } = require("../dist/index.js");
            const app = forge({ model: options.model });

            log.info(`Using model: ${options.model}`);
            log.info(`Generating from: "${prompt}"`);
            console.log();

            if (options.stream) {
                process.stdout.write(`${colors.cyan}`);
                for await (const token of app.stream(prompt, {
                    maxTokens: parseInt(options.tokens),
                    temperature: parseFloat(options.temperature),
                })) {
                    process.stdout.write(token);
                }
                process.stdout.write(`${colors.reset}\n`);
            } else {
                const output = await app.generate(prompt, {
                    maxTokens: parseInt(options.tokens),
                    temperature: parseFloat(options.temperature),
                });
                console.log(`${colors.cyan}${output}${colors.reset}`);
            }

            console.log();
        } catch (error) {
            log.error(error.message);
            process.exit(1);
        }
    });

// ═══════════════════════════════════════════════════════════════════════════
// TRAIN COMMAND - Train a model
// ═══════════════════════════════════════════════════════════════════════════
program
    .command("train <file>")
    .description("Train a model from a JSON data file")
    .option("-m, --model <model>", "Model preset", "small")
    .option("-e, --epochs <n>", "Number of training epochs", "10")
    .option("-l, --learning-rate <n>", "Learning rate", "0.001")
    .option("-o, --output <path>", "Output path for trained model", "./model.json")
    .option("-v, --verbose", "Show detailed training progress", false)
    .action(async (file, options) => {
        console.log(banner);
        log.fire("Starting training...");

        try {
            if (!fs.existsSync(file)) {
                throw new Error(`File not found: ${file}`);
            }

            const data = JSON.parse(fs.readFileSync(file, "utf-8"));
            log.info(`Loaded ${data.length} training samples`);

            const { forge } = require("../dist/index.js");
            const app = forge({ model: options.model });

            log.info(`Model: ${options.model}`);
            log.info(`Epochs: ${options.epochs}`);
            log.info(`Learning rate: ${options.learningRate}`);
            console.log();

            await app.train(data, {
                epochs: parseInt(options.epochs),
                learningRate: parseFloat(options.learningRate),
                verbose: options.verbose,
            });

            log.success(`Training complete!`);
            log.info(`Model saved to: ${options.output}`);
        } catch (error) {
            log.error(error.message);
            process.exit(1);
        }
    });

// ═══════════════════════════════════════════════════════════════════════════
// SERVE COMMAND - Start API server
// ═══════════════════════════════════════════════════════════════════════════
program
    .command("serve")
    .alias("server")
    .alias("start")
    .description("Start a Forge API server")
    .option("-m, --model <model>", "Model preset", "small")
    .option("-p, --port <port>", "Port number", "3000")
    .option("-h, --host <host>", "Host address", "localhost")
    .option("--cors", "Enable CORS", false)
    .action((options) => {
        console.log(banner);
        try {
            const { forge, logger, timer } = require("../dist/index.js");
            const app = forge({ model: options.model });

            app.use(logger);
            app.use(timer);

            // Health check
            app.get("/health", async (req, res) => {
                res.json({ status: "ok", model: options.model });
            });

            // Chat endpoint
            app.post("/chat", async (req, res) => {
                const output = await app.generate(req.body.message || req.body.prompt);
                res.json({ output });
            });

            // Generate endpoint
            app.post("/generate", async (req, res) => {
                const output = await app.generate(req.body.prompt, {
                    maxTokens: req.body.maxTokens || 50,
                    temperature: req.body.temperature || 0.8,
                });
                res.json({ output });
            });

            app.listen(parseInt(options.port), () => {
                log.fire(`Forge server running!`);
                console.log();
                log.info(`Model: ${options.model}`);
                log.info(`Endpoint: http://${options.host}:${options.port}`);
                console.log();
                log.info("API Routes:");
                console.log(`  ${colors.dim}GET  /health${colors.reset}   - Health check`);
                console.log(`  ${colors.dim}POST /chat${colors.reset}     - Chat completion`);
                console.log(`  ${colors.dim}POST /generate${colors.reset} - Text generation`);
                console.log();
            });
        } catch (error) {
            log.error(error.message);
            process.exit(1);
        }
    });

// ═══════════════════════════════════════════════════════════════════════════
// INFO COMMAND - Show system info
// ═══════════════════════════════════════════════════════════════════════════
program
    .command("info")
    .description("Show Forge system and backend information")
    .action(() => {
        console.log(banner);

        try {
            const { printBackendInfo, getBackendInfo } = require("../dist/index.js");

            console.log(`${colors.bright}System Information${colors.reset}`);
            console.log(`${"─".repeat(40)}`);
            console.log(`  Node.js:    ${process.version}`);
            console.log(`  Platform:   ${process.platform}`);
            console.log(`  Arch:       ${process.arch}`);
            console.log(`  Forge:      v${pkg.version}`);
            console.log();

            printBackendInfo();

            // Try to show native core info
            try {
                const info = getBackendInfo();
                console.log();
                console.log(`${colors.bright}Backend Details${colors.reset}`);
                console.log(`${"─".repeat(40)}`);
                console.log(`  Native:     ${info.native ? colors.green + "Yes" : colors.yellow + "No (fallback)"}${colors.reset}`);
                console.log(`  CUDA:       ${info.cuda ? colors.green + "Available" : colors.dim + "Not available"}${colors.reset}`);
            } catch {
                // Ignore
            }
        } catch (error) {
            log.error(error.message);
            process.exit(1);
        }
    });

// ═══════════════════════════════════════════════════════════════════════════
// BENCHMARK COMMAND - Run benchmarks
// ═══════════════════════════════════════════════════════════════════════════
program
    .command("benchmark")
    .alias("bench")
    .description("Run performance benchmarks")
    .option("-m, --model <model>", "Model preset to benchmark", "small")
    .option("-i, --iterations <n>", "Number of iterations", "10")
    .action(async (options) => {
        console.log(banner);
        log.fire("Running benchmarks...");
        console.log();

        try {
            const { forge, Tensor } = require("../dist/index.js");

            // Tensor benchmarks
            console.log(`${colors.bright}Tensor Operations${colors.reset}`);
            console.log(`${"─".repeat(40)}`);

            const sizes = [100, 500, 1000];
            for (const size of sizes) {
                const t1 = Tensor.randn([size, size]);
                const t2 = Tensor.randn([size, size]);

                const start = performance.now();
                for (let i = 0; i < 5; i++) {
                    t1.matmul(t2);
                }
                const elapsed = (performance.now() - start) / 5;

                console.log(`  ${size}x${size} matmul: ${elapsed.toFixed(2)}ms`);
            }

            console.log();

            // Generation benchmarks
            console.log(`${colors.bright}Generation (${options.model})${colors.reset}`);
            console.log(`${"─".repeat(40)}`);

            const app = forge({ model: options.model });
            const iterations = parseInt(options.iterations);

            const genStart = performance.now();
            for (let i = 0; i < iterations; i++) {
                await app.generate("Hello", { maxTokens: 10 });
            }
            const genElapsed = (performance.now() - genStart) / iterations;

            console.log(`  Avg generation time: ${genElapsed.toFixed(2)}ms`);
            console.log(`  Throughput: ${(1000 / genElapsed).toFixed(2)} gen/sec`);

            console.log();
            log.success("Benchmark complete!");
        } catch (error) {
            log.error(error.message);
            process.exit(1);
        }
    });

// ═══════════════════════════════════════════════════════════════════════════
// TEMPLATE GENERATORS
// ═══════════════════════════════════════════════════════════════════════════

function generateBasicTemplate(model, typescript) {
    const importStyle = typescript
        ? `import { forge, logger } from "forge-ai";`
        : `const { forge, logger } = require("forge-ai");`;

    return `${importStyle}

/**
 * 🔥 Forge AI Application
 */

async function main() {
    console.log("🔥 Starting Forge AI...");
    
    // Create a Forge app with ${model} model
    const app = forge({ model: "${model}" });
    
    // Add logging middleware
    app.use(logger);
    
    // Generate text
    const output = await app.generate("Hello, world!", {
        maxTokens: 50,
        temperature: 0.8,
    });
    
    console.log("Generated:", output);
    
    // Stream tokens
    console.log("\\nStreaming:");
    for await (const token of app.stream("AI is", { maxTokens: 20 })) {
        process.stdout.write(token);
    }
    console.log();
}

main().catch(console.error);
`;
}

function generateServerTemplate(model, typescript) {
    const importStyle = typescript
        ? `import { forge, logger, timer } from "forge-ai";`
        : `const { forge, logger, timer } = require("forge-ai");`;

    return `${importStyle}

/**
 * 🔥 Forge AI Server
 */

const app = forge({ model: "${model}" });

// Middleware
app.use(logger);
app.use(timer);

// Health check
app.get("/health", async (req, res) => {
    res.json({ status: "ok", model: "${model}" });
});

// Chat endpoint
app.post("/chat", async (req, res) => {
    const { message, maxTokens = 100 } = req.body;
    const output = await app.generate(message, { maxTokens });
    res.json({ output });
});

// Start server
app.listen(3000, () => {
    console.log("🔥 Forge server running on http://localhost:3000");
});
`;
}

function generateRAGTemplate(model, typescript) {
    const importStyle = typescript
        ? `import { forge, logger } from "forge-ai";`
        : `const { forge, logger } = require("forge-ai");`;

    return `${importStyle}

/**
 * 🔥 Forge AI with RAG
 */

async function main() {
    const app = forge({ model: "${model}" });
    app.use(logger);
    
    // Create RAG instance
    const rag = app.rag();
    
    // Add knowledge
    rag.add("Forge was created in 2024");
    rag.add("Forge is an AI framework for JavaScript");
    rag.add("It uses Express-style APIs");
    
    // Query
    const answer = await rag.query("When was Forge created?");
    console.log("Answer:", answer);
}

main().catch(console.error);
`;
}

function generateAgentTemplate(model, typescript) {
    const importStyle = typescript
        ? `import { forge, createTool, calculatorTool } from "forge-ai";`
        : `const { forge, createTool, calculatorTool } = require("forge-ai");`;

    return `${importStyle}

/**
 * 🔥 Forge AI Agent with Tools
 */

async function main() {
    const app = forge({ model: "${model}" });
    
    // Create agent
    const agent = app.agent();
    
    // Add built-in calculator
    agent.tool(calculatorTool);
    
    // Add custom tool
    agent.tool(createTool("weather", "Get the current weather", async () => {
        return "Sunny, 25°C";
    }));
    
    // Run agent
    console.log("Agent: What is 10 * 5?");
    await agent.run("Calculate 10 * 5");
    
    console.log("\\nAgent: What's the weather?");
    await agent.run("What's the weather like?");
}

main().catch(console.error);
`;
}

function generateTrainTemplate(model, typescript) {
    const importStyle = typescript
        ? `import { forge } from "forge-ai";
import fs from "fs";`
        : `const { forge } = require("forge-ai");
const fs = require("fs");`;

    return `${importStyle}

/**
 * 🔥 Train a Forge Model
 */

async function main() {
    console.log("🔥 Starting training...");
    
    // Load training data
    const data = JSON.parse(fs.readFileSync("./data.json", "utf-8"));
    console.log(\`Loaded \${data.length} training samples\`);
    
    // Create app
    const app = forge({ model: "${model}" });
    
    // Train
    await app.train(data, {
        epochs: 10,
        learningRate: 0.001,
        verbose: true,
    });
    
    // Test generation
    console.log("\\nTesting trained model:");
    const output = await app.generate("Forge is", { maxTokens: 30 });
    console.log("Output:", output);
    
    console.log("\\n✓ Training complete!");
}

main().catch(console.error);
`;
}

program.parse();
