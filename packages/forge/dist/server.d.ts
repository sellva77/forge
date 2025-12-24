/**
 * Forge - Server
 * ===============
 */
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
export declare class Server {
    private routes;
    private server;
    get(path: string, handler: RouteHandler): this;
    post(path: string, handler: RouteHandler): this;
    put(path: string, handler: RouteHandler): this;
    delete(path: string, handler: RouteHandler): this;
    listen(port: number, callback?: () => void): this;
    close(): void;
    private parseBody;
}
//# sourceMappingURL=server.d.ts.map