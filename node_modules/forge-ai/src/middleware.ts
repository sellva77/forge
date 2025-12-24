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
export const logger: Middleware = async (ctx, next) => {
    console.log(`→ ${ctx.input}`);
    await next();
    console.log(`← ${ctx.output || "(no output)"}`);
};

/**
 * Timer middleware - measures execution time
 */
export const timer: Middleware = async (ctx, next) => {
    ctx.startTime = Date.now();
    await next();
    ctx.endTime = Date.now();
    ctx.duration = ctx.endTime - ctx.startTime;
};

/**
 * Normalize middleware - cleans input
 */
export const normalize: Middleware = async (ctx, next) => {
    ctx.input = ctx.input.trim();
    await next();
};

/**
 * LowerCase middleware
 */
export const lowercase: Middleware = async (ctx, next) => {
    ctx.input = ctx.input.toLowerCase();
    await next();
};

/**
 * Cache middleware
 */
export function cache(maxSize: number = 100): Middleware {
    const store = new Map<string, string>();

    return async (ctx, next) => {
        const cached = store.get(ctx.input);
        if (cached) {
            ctx.output = cached;
            ctx.cached = true;
            return;
        }

        await next();

        if (ctx.output) {
            if (store.size >= maxSize) {
                const firstKey = store.keys().next().value;
                if (firstKey !== undefined) {
                    store.delete(firstKey);
                }
            }
            store.set(ctx.input, ctx.output);
        }
    };
}

/**
 * RateLimit middleware
 */
export function rateLimit(maxRequests: number, windowMs: number): Middleware {
    const requests = new Map<string, number[]>();

    return async (ctx, next) => {
        const key = ctx.clientId || "default";
        const now = Date.now();
        const windowStart = now - windowMs;

        let timestamps = requests.get(key) || [];
        timestamps = timestamps.filter(t => t > windowStart);

        if (timestamps.length >= maxRequests) {
            ctx.error = "Rate limit exceeded";
            ctx.statusCode = 429;
            return;
        }

        timestamps.push(now);
        requests.set(key, timestamps);

        await next();
    };
}

/**
 * Compose multiple middlewares
 */
export function compose(...middlewares: Middleware[]): Middleware {
    return async (ctx, next) => {
        let index = 0;

        const dispatch = async (): Promise<void> => {
            if (index < middlewares.length) {
                await middlewares[index++](ctx, dispatch);
            } else {
                await next();
            }
        };

        await dispatch();
    };
}
