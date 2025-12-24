---
description: Install VS Build Tools and compile Rust core for maximum performance
---

# Install VS Build Tools for Rust Core

Follow these steps when you need maximum performance with the Rust native core.

## Prerequisites
- Administrator access on Windows
- ~5GB free disk space
- Internet connection

## Steps

1. **Open PowerShell as Administrator**

2. **Install VS Build Tools using winget:**
   ```powershell
   winget install Microsoft.VisualStudio.2022.BuildTools --override "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
   ```
   // turbo

3. **Close and reopen your terminal** (to refresh environment variables)

4. **Verify installation:**
   ```powershell
   where.exe cl.exe
   where.exe link.exe
   ```

5. **Navigate to the Rust core:**
   ```powershell
   cd packages/core
   ```

6. **Build the Rust core:**
   ```powershell
   cargo build --release
   ```
   // turbo

7. **The compiled binary will be at:**
   `target/release/forge_core.dll`

8. **Install the npm package with native bindings:**
   ```powershell
   npm install
   ```

## Verification

After building, test that the Rust core works:

```javascript
const { Tensor, Model, Tokenizer } = require("@forge-ai/core");
const t = new Tensor([1, 2, 3], [3]);
console.log("Rust core working:", t.shape());
```

## Troubleshooting

### Error: linker `link.exe` not found
- Restart your terminal after installing VS Build Tools
- Make sure you installed with VCTools workload

### Error: cargo not found
- Install Rust: `winget install Rustlang.Rust.MSVC`

### Build errors with CUDA
- CUDA is optional, build without it: `cargo build --release --no-default-features`
