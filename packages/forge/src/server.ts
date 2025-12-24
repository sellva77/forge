/**
 * Forge - Server
 * ===============
 */

import * as http from "http";

export interface Request {
    method: string;
    url: string;
    headers: Record<string, string>;
    body: any;
    params: Record<string, string>;
    query: Record<string, string>;
}

export interface Response {
    statusCode: number;
    json: (data: any) => void;
    send: (data: string) => void;
    status: (code: number) => Response;
}

export type RouteHandler = (req: Request, res: Response) => void | Promise<void>;

export class Server {
    private routes: Map<string, RouteHandler> = new Map();
    private server: http.Server | null = null;

    get(path: string, handler: RouteHandler): this {
        this.routes.set(`GET:${path}`, handler);
        return this;
    }

    post(path: string, handler: RouteHandler): this {
        this.routes.set(`POST:${path}`, handler);
        return this;
    }

    put(path: string, handler: RouteHandler): this {
        this.routes.set(`PUT:${path}`, handler);
        return this;
    }

    delete(path: string, handler: RouteHandler): this {
        this.routes.set(`DELETE:${path}`, handler);
        return this;
    }

    listen(port: number, callback?: () => void): this {
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
            const query: Record<string, string> = {};
            url.searchParams.forEach((value, key) => {
                query[key] = value;
            });

            // Parse body
            let body: any = {};
            if (req.method === "POST" || req.method === "PUT") {
                body = await this.parseBody(req);
            }

            // Build request/response objects
            const request: Request = {
                method: req.method || "GET",
                url: path,
                headers: req.headers as Record<string, string>,
                body,
                params: {},
                query,
            };

            const response: Response = {
                statusCode: 200,
                json: (data: any) => {
                    res.setHeader("Content-Type", "application/json");
                    res.writeHead(response.statusCode);
                    res.end(JSON.stringify(data));
                },
                send: (data: string) => {
                    res.writeHead(response.statusCode);
                    res.end(data);
                },
                status: (code: number) => {
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
                } catch (error: any) {
                    response.status(500).json({ error: error.message });
                }
            } else {
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
            if (callback) callback();
        });

        return this;
    }

    close(): void {
        if (this.server) {
            this.server.close();
        }
    }

    private parseBody(req: http.IncomingMessage): Promise<any> {
        return new Promise((resolve) => {
            let data = "";
            req.on("data", chunk => data += chunk);
            req.on("end", () => {
                try {
                    resolve(JSON.parse(data));
                } catch {
                    resolve(data);
                }
            });
        });
    }
}
