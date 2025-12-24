# @forge-ai/core

🔥 **High-performance Rust core for Forge AI** with automatic JavaScript fallback.

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                    npm install forge-ai                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │   Detect your platform        │
              │   (Windows/Mac/Linux)         │
              └───────────────────────────────┘
                              │
           ┌──────────────────┴──────────────────┐
           │                                      │
           ▼                                      ▼
┌─────────────────────────┐         ┌─────────────────────────┐
│  Pre-built binary found │         │  No binary available    │
│  for your platform      │         │                         │
└─────────────────────────┘         └─────────────────────────┘
           │                                      │
           ▼                                      ▼
   ⚡ Native Rust Core              📦 JavaScript Fallback
   (5-10x faster)                   (Always works!)
```

**Users never need to compile anything. Pre-built binaries are downloaded automatically.**

## For Users

Just install and use:

```bash
npm install forge-ai
```

```javascript
const { forge } = require("forge-ai");

const app = forge({ model: "small" });
const output = await app.generate("Hello!");
```

The package automatically:
1. Detects your platform (Windows/Mac/Linux, x64/ARM64)
2. Downloads the pre-built native binary
3. Falls back to JavaScript if no binary is available

## Supported Platforms

| Platform | Architecture | Binary | Fallback |
|----------|--------------|--------|----------|
| Windows  | x64          | ✅     | ✅       |
| macOS    | x64 (Intel)  | ✅     | ✅       |
| macOS    | arm64 (M1+)  | ✅     | ✅       |
| Linux    | x64 (glibc)  | ✅     | ✅       |
| Linux    | arm64 (glibc)| ✅     | ✅       |
| Other    | Any          | ❌     | ✅       |

## Performance

| Operation | Native Rust | JS Fallback |
|-----------|-------------|-------------|
| 512×512 matmul | ~5ms | ~50ms |
| Softmax (256K) | ~0.5ms | ~5ms |
| Token generation | ~10ms/tok | ~100ms/tok |

Native is ~10x faster due to:
- Multi-threaded operations via Rayon
- SIMD vectorization (AVX2/NEON)
- Optimized memory layout

## Check Your Backend

```javascript
const { getInfo, printInfo } = require("@forge-ai/core");

// Get info object
console.log(getInfo());
// { native: true, platform: "win32-x64-msvc", threads: 8, ... }

// Pretty print
printInfo();
```

## For Framework Developers

If you want to build from source (not required for users):

### Prerequisites
- Rust toolchain (`rustup`)
- Visual Studio Build Tools (Windows only)
- Node.js 18+

### Build

```bash
cd packages/core
npm install
npm run build
```

### The binaries are built automatically by GitHub Actions:

1. Push code to GitHub
2. GitHub Actions builds for all platforms
3. Binaries are uploaded as npm packages
4. Users get pre-built binaries via `npm install`

## Package Structure

```
@forge-ai/core                    # Main package (loader + fallback)
├── @forge-ai/core-win32-x64-msvc  # Windows binary
├── @forge-ai/core-darwin-x64      # macOS Intel binary
├── @forge-ai/core-darwin-arm64    # macOS Apple Silicon binary
├── @forge-ai/core-linux-x64-gnu   # Linux x64 binary
└── @forge-ai/core-linux-arm64-gnu # Linux ARM64 binary
```

Users only download the binary for their platform (~2-5MB), not all of them.

## License

MIT © Forge AI Team
