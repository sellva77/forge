"use strict";
/**
 * Forge - Server
 * ===============
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
exports.Server = void 0;
const http = __importStar(require("http"));
class Server {
    routes = new Map();
    server = null;
    get(path, handler) {
        this.routes.set(`GET:${path}`, handler);
        return this;
    }
    post(path, handler) {
        this.routes.set(`POST:${path}`, handler);
        return this;
    }
    put(path, handler) {
        this.routes.set(`PUT:${path}`, handler);
        return this;
    }
    delete(path, handler) {
        this.routes.set(`DELETE:${path}`, handler);
        return this;
    }
    listen(port, callback) {
        this.server = http.createServer(async (req, res) => {
            // CORS
            res.setHeader("Access-Control-Allow-Origin", "*");
            res.setHeader("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
            res.setHeader("Access-Control-Allow-Headers", "Content-Type");
            if (req.method === "OPTIONS") {
                res.writeHead(200);
                res.end();
                return;
            }
            // Parse URL
            const url = new URL(req.url || "/", `http://localhost:${port}`);
            const path = url.pathname;
            const query = {};
            url.searchParams.forEach((value, key) => {
                query[key] = value;
            });
            // Parse body
            let body = {};
            if (req.method === "POST" || req.method === "PUT") {
                body = await this.parseBody(req);
            }
            // Build request/response objects
            const request = {
                method: req.method || "GET",
                url: path,
                headers: req.headers,
                body,
                params: {},
                query,
            };
            const response = {
                statusCode: 200,
                json: (data) => {
                    res.setHeader("Content-Type", "application/json");
                    res.writeHead(response.statusCode);
                    res.end(JSON.stringify(data));
                },
                send: (data) => {
                    res.writeHead(response.statusCode);
                    res.end(data);
                },
                status: (code) => {
                    response.statusCode = code;
                    return response;
                },
            };
            // Find handler
            const key = `${req.method}:${path}`;
            const handler = this.routes.get(key);
            if (handler) {
                try {
                    await handler(request, response);
                }
                catch (error) {
                    response.status(500).json({ error: error.message });
                }
            }
            else {
                response.status(404).json({ error: "Not Found" });
            }
        });
        this.server.listen(port, () => {
            console.log(`\n🚀 Forge server running on http://localhost:${port}`);
            console.log(`   Routes:`);
            for (const [key] of this.routes) {
                const [method, path] = key.split(":");
                console.log(`   - ${method} ${path}`);
            }
            console.log("");
            if (callback)
                callback();
        });
        return this;
    }
    close() {
        if (this.server) {
            this.server.close();
        }
    }
    parseBody(req) {
        return new Promise((resolve) => {
            let data = "";
            req.on("data", chunk => data += chunk);
            req.on("end", () => {
                try {
                    resolve(JSON.parse(data));
                }
                catch {
                    resolve(data);
                }
            });
        });
    }
}
exports.Server = Server;
//# sourceMappingURL=server.js.map