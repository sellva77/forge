"use strict";
/**
 * Forge - Middleware System
 * ==========================
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.lowercase = exports.normalize = exports.timer = exports.logger = void 0;
exports.cache = cache;
exports.rateLimit = rateLimit;
exports.compose = compose;
/**
 * Logger middleware - logs input/output
 */
const logger = async (ctx, next) => {
    console.log(`→ ${ctx.input}`);
    await next();
    console.log(`← ${ctx.output || "(no output)"}`);
};
exports.logger = logger;
/**
 * Timer middleware - measures execution time
 */
const timer = async (ctx, next) => {
    ctx.startTime = Date.now();
    await next();
    ctx.endTime = Date.now();
    ctx.duration = ctx.endTime - ctx.startTime;
};
exports.timer = timer;
/**
 * Normalize middleware - cleans input
 */
const normalize = async (ctx, next) => {
    ctx.input = ctx.input.trim();
    await next();
};
exports.normalize = normalize;
/**
 * LowerCase middleware
 */
const lowercase = async (ctx, next) => {
    ctx.input = ctx.input.toLowerCase();
    await next();
};
exports.lowercase = lowercase;
/**
 * Cache middleware
 */
function cache(maxSize = 100) {
    const store = new Map();
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
function rateLimit(maxRequests, windowMs) {
    const requests = new Map();
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
function compose(...middlewares) {
    return async (ctx, next) => {
        let index = 0;
        const dispatch = async () => {
            if (index < middlewares.length) {
                await middlewares[index++](ctx, dispatch);
            }
            else {
                await next();
            }
        };
        await dispatch();
    };
}
//# sourceMappingURL=middleware.js.map