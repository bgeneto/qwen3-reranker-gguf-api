# Building with llama.cpp PR #14029 Patch

This project can be built with a patched version of llama.cpp that includes PR #14029, which provides a fallback for RANK pooling when bias vectors are missing.

## What the Patch Does

PR #14029 addresses the error:
```
RANK pooling requires either cls+cls_b or cls_out+cls_out_b
```

The patch adds a fallback mechanism that allows RANK pooling to work even when the model doesn't have the required bias vectors (`cls_b` or `cls_out_b`).

## Building the Patched Version

### Prerequisites
- Docker with NVIDIA container runtime
- At least 8GB free disk space
- NVIDIA GPU with CUDA support

### Build Process

1. **Using PowerShell (Windows):**
   ```powershell
   .\build-with-patch.ps1
   ```

2. **Using Bash (Linux/WSL):**
   ```bash
   chmod +x build-with-patch.sh
   ./build-with-patch.sh
   ```

3. **Manual build:**
   ```bash
   docker compose build --no-cache
   ```

### What Gets Built

The build process:
1. Builds base Python dependencies in a slim Python image
2. Switches to NVIDIA CUDA development image with full CUDA toolkit
3. Clones llama.cpp repository and checks out PR #14029 branch
4. Builds llama.cpp with CUDA support using the CUDA development environment
5. Builds llama-cpp-python from source using the patched llama.cpp
6. Creates the final Docker image with all components

### Configuration

After building, ensure your `.env` file has:
```env
RERANKING_MODE=modern
POOLING_TYPE=4
EMBEDDING_MODE=true
```

The patched version will automatically fall back to bias-free RANK pooling if your model doesn't have the required bias vectors.

### Troubleshooting

If the build fails:

1. **CUDA errors**: Make sure you have NVIDIA Docker runtime installed
2. **Memory issues**: The build requires significant memory. Close other applications
3. **Network issues**: The build downloads llama.cpp source code. Ensure internet connectivity

### Verification

After successful build and startup, check the logs:
```bash
docker compose logs -f
```

You should see messages like:
- "Model loaded successfully with configuration"
- "Using modern reranking (pooling-based)"

### Performance

The patched version should provide:
- Better compatibility with various reranking models
- Automatic fallback behavior for models without bias vectors
- Full CUDA acceleration support
- Same API compatibility as the original version
