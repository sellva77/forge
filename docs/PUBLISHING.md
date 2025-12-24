# 📦 Publishing Guide

This guide explains how to publish the Forge AI packages to npm.

## Prerequisites

1. **npm account** - Create one at [npmjs.com](https://www.npmjs.com/signup)
2. **npm login** - Run `npm login` to authenticate
3. **Build the packages** - Ensure everything compiles

## Package Structure

The monorepo contains two packages:

| Package | npm Name | Description |
|---------|----------|-------------|
| `packages/forge` | `forge-ai` | Main framework (TypeScript) |
| `packages/core` | `@forge-ai/core` | Rust native bindings |

## Publishing Steps

### 1. Build Everything

```bash
# From the root directory
npm run build
```

### 2. Run Tests

```bash
npm test
```

### 3. Version Bump

```bash
# Patch version (1.0.0 -> 1.0.1)
npm run version:patch

# Minor version (1.0.0 -> 1.1.0)
npm run version:minor

# Major version (1.0.0 -> 2.0.0)
npm run version:major
```

### 4. Publish to npm

```bash
# Publish all packages
npm run publish:all

# Or publish individually
cd packages/forge
npm publish

cd packages/core
npm publish
```

## First-Time Setup

### Scoped Package (@forge-ai/core)

For the scoped package, you need to either:

1. **Create an organization** on npm called `forge-ai`
2. **Or use a different scope** (e.g., your username: `@yourusername/core`)

```bash
# Publish scoped package publicly
npm publish --access public
```

### Reserved Package Names

If `forge-ai` is taken, consider alternatives:
- `forge-ai-js`
- `forgeai`
- `forge-framework`
- `@yourorg/forge`

## Pre-publish Checklist

- [ ] All tests pass
- [ ] TypeScript compiles without errors
- [ ] README.md is up to date
- [ ] CHANGELOG.md is updated
- [ ] Version numbers are bumped
- [ ] License file is present
- [ ] .npmignore is configured correctly

## Verify Package Contents

Before publishing, verify what will be included:

```bash
cd packages/forge
npm pack --dry-run
```

This shows exactly which files will be in the published package.

## Post-Publish

After publishing:

1. **Test installation**:
   ```bash
   npm install forge-ai
   ```

2. **Verify on npm**:
   Visit https://www.npmjs.com/package/forge-ai

3. **Create a GitHub release** with the same version tag

## Troubleshooting

### "Package name already exists"

Choose a different name or request the name from the current owner.

### "You must be logged in"

Run `npm login` and enter your credentials.

### "Permission denied"

For scoped packages, ensure you have access to the organization.

### "402 Payment Required"

Scoped packages require `--access public` for free accounts:
```bash
npm publish --access public
```

## CI/CD Publishing

For automated publishing, set up GitHub Actions:

```yaml
# .github/workflows/publish.yml
name: Publish to npm

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          registry-url: 'https://registry.npmjs.org'
      - run: npm ci
      - run: npm run build
      - run: npm run publish:all
        env:
          NODE_AUTH_TOKEN: ${{ secrets.NPM_TOKEN }}
```

Add your npm token to GitHub Secrets as `NPM_TOKEN`.
