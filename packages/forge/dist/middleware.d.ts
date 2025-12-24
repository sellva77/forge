/**
 * Forge - Middleware System
 * ==========================
 */
export interface Context {
    input: string;
    output?: string;
    tokens?: number[];
    startTime?: number;
    endTime?: number;
    [key: string]: any;
}
export type NextFn = () => Promise<void>;
export type Middleware = (ctx: Context, next: NextFn) => Promise<void>;
/**
 * Logger middleware - logs input/output
 */
export declare const logger: Middleware;
/**
 * Timer middleware - measures execution time
 */
export declare const timer: Middleware;
/**
 * Normalize middleware - cleans input
 */
export declare const normalize: Middleware;
/**
 * LowerCase middleware
 */
export declare const lowercase: Middleware;
/**
 * Cache middleware
 */
export declare function cache(maxSize?: number): Middleware;
/**
 * RateLimit middleware
 */
export declare function rateLimit(maxRequests: number, windowMs: number): Middleware;
/**
 * Compose multiple middlewares
 */
export declare function compose(...middlewares: Middleware[]): Middleware;
//# sourceMappingURL=middleware.d.ts.map