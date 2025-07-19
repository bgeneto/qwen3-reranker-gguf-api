# Reranking Modes Guide

This guide explains the different reranking modes available with the updated llama-cpp-python version 0.3.14+ and how to use them.

## Overview

The newer versions of llama-cpp-python support native reranking through the `--reranking` runtime option and pooling types. This provides better performance and accuracy compared to the legacy logits-based approach.

## Available Modes

### 1. Auto Mode (Recommended)
```env
RERANKING_MODE=auto
```
- **Description**: Automatically detects the best reranking method
- **Behavior**:
  - First tries modern pooling-based reranking
  - Falls back to legacy logits-based reranking if modern fails
- **Use Case**: Best for most users, provides compatibility and performance

### 2. Modern Mode
```env
RERANKING_MODE=modern
POOLING_TYPE=4
EMBEDDING_MODE=true
```
- **Description**: Uses the new pooling-based reranking (LLAMA_POOLING_TYPE_RANK)
- **Requirements**:
  - llama-cpp-python >= 0.3.x
  - Model must support reranking pooling
- **Benefits**:
  - Better performance
  - More accurate scores
  - Native reranking support
- **Use Case**: When you have a compatible model and newer llama-cpp-python

### 3. Legacy Mode
```env
RERANKING_MODE=legacy
```
- **Description**: Uses the original logits-based reranking approach
- **Behavior**:
  - Generates "yes"/"no" tokens and extracts logits
  - Compatible with any chat model
- **Use Case**:
  - Older llama-cpp-python versions
  - Models that don't support pooling-based reranking
  - Debugging/comparison purposes

## Configuration Examples

### Basic Configuration (Auto Mode)
```env
# Model settings
MODEL_FILENAME=Qwen3-Reranker-4B-q4_k_m.gguf
MODEL_LINK=https://huggingface.co/Mungert/Qwen3-Reranker-4B-GGUF/resolve/main/Qwen3-Reranker-4B-q4_k_m.gguf

# Hardware settings
N_GPU_LAYERS=-1
N_CTX=8192
N_BATCH=512

# Reranking settings
RERANKING_MODE=auto
POOLING_TYPE=4
EMBEDDING_MODE=true
```

### High Performance Configuration (Modern Mode)
```env
# Model settings
MODEL_FILENAME=Qwen3-Reranker-4B-q4_k_m.gguf
MODEL_LINK=https://huggingface.co/Mungert/Qwen3-Reranker-4B-GGUF/resolve/main/Qwen3-Reranker-4B-q4_k_m.gguf

# Hardware settings - optimize for GPU
N_GPU_LAYERS=-1
N_CTX=32768
N_BATCH=1024
N_THREADS=8

# Reranking settings - force modern mode
RERANKING_MODE=modern
POOLING_TYPE=4
EMBEDDING_MODE=true
```

### Compatibility Configuration (Legacy Mode)
```env
# Model settings
MODEL_FILENAME=Qwen3-Reranker-4B-q4_k_m.gguf
MODEL_LINK=https://huggingface.co/Mungert/Qwen3-Reranker-4B-GGUF/resolve/main/Qwen3-Reranker-4B-q4_k_m.gguf

# Hardware settings - conservative for CPU
N_GPU_LAYERS=0
N_CTX=4096
N_BATCH=256
N_THREADS=4

# Reranking settings - use legacy approach
RERANKING_MODE=legacy
```

## Troubleshooting

### Model Loading Issues
If you're getting "Failed to load model" errors:

1. **Try different modes**:
   ```env
   RERANKING_MODE=legacy  # Most compatible
   ```

2. **Reduce hardware requirements**:
   ```env
   N_GPU_LAYERS=0         # Force CPU-only
   N_CTX=2048            # Reduce context size
   N_BATCH=128           # Reduce batch size
   ```

3. **Check model file integrity**:
   ```bash
   python debug_model.py  # Run the diagnostic script
   ```

### Performance Issues
If reranking is slow:

1. **Use modern mode**:
   ```env
   RERANKING_MODE=modern
   EMBEDDING_MODE=true
   ```

2. **Optimize hardware settings**:
   ```env
   N_GPU_LAYERS=-1       # Use all GPU layers
   N_BATCH=1024         # Increase batch size
   ```

### Accuracy Issues
If reranking scores seem incorrect:

1. **Try different modes**:
   ```env
   RERANKING_MODE=auto   # Let system choose best method
   ```

2. **Check model compatibility**:
   - Ensure you're using a proper reranking model
   - Verify the model supports the expected input format

## API Usage

The reranking API remains the same regardless of the mode:

```python
import requests

response = requests.post("http://localhost:8000/v1/rerank", {
    "model": "qwen3-reranker",
    "query": "What is machine learning?",
    "documents": [
        {"text": "Machine learning is a subset of AI..."},
        {"text": "Python is a programming language..."}
    ],
    "top_n": 5
}, headers={
    "Authorization": "Bearer your-api-token"
})

results = response.json()["results"]
for result in results:
    print(f"Document {result['index']}: {result['relevance_score']:.3f}")
```

## Migration Guide

### From Legacy Implementation
1. Update `llama-cpp-python` to version 0.3.14+
2. Add new configuration options to your `.env` file
3. Set `RERANKING_MODE=auto` for gradual migration
4. Test with your specific model and adjust as needed

### Configuration Migration
```diff
# Old configuration
- N_CTX=8192
- N_GPU_LAYERS=-1

# New configuration
+ N_CTX=8192
+ N_GPU_LAYERS=-1
+ RERANKING_MODE=auto
+ POOLING_TYPE=4
+ EMBEDDING_MODE=true
```

## Technical Details

### Pooling Types
- `POOLING_TYPE=4` corresponds to `LLAMA_POOLING_TYPE_RANK`
- This attaches a classification head to the model graph
- Specifically designed for reranking models

### Embedding Mode
- `EMBEDDING_MODE=true` enables embedding generation
- Required for pooling-based reranking
- Works with the pooling type to produce relevance scores

### Model Compatibility
The modern reranking mode works best with:
- Models specifically trained for reranking
- GGUF models with proper metadata
- Qwen3-Reranker, BGE-reranker, and similar architectures

For other models, the legacy mode provides better compatibility.
