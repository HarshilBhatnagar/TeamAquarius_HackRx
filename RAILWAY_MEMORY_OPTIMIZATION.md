# Railway Memory Optimization Guide

## 🚀 **Memory Optimization for Railway Deployment**

### **Current Memory Usage Analysis**

Based on our optimizations, the system should use approximately:
- **Base Memory**: ~200-300MB (FastAPI + dependencies)
- **Document Processing**: ~50-100MB per document (cached)
- **LLM Calls**: ~10-20MB per request
- **Vector Store**: ~20-50MB (Pinecone, cached)

**Total Estimated Usage**: 300-500MB under normal load

### **Memory Optimization Strategies**

#### **1. Document Cache Management**
```python
# In services/query_engine.py
# Limit cache size to prevent memory bloat
MAX_CACHE_SIZE = 10  # Maximum 10 documents in cache
if len(document_cache) > MAX_CACHE_SIZE:
    # Remove oldest entries
    oldest_key = next(iter(document_cache))
    del document_cache[oldest_key]
```

#### **2. Chunk Size Optimization**
```python
# In utils/chunking.py
# Reduce chunk sizes for lower memory usage
chunk_size = 600  # Reduced from 800
chunk_overlap = 150  # Reduced from 200
max_chunks = 30  # Reduced from 50
```

#### **3. LLM Response Optimization**
```python
# In utils/llm.py
# Limit context and response sizes
context_limit = 2000  # Reduced from 3000
max_tokens = 600  # Reduced from 800
```

#### **4. Concurrent Request Limiting**
```python
# In services/query_engine.py
# Limit concurrent processing
MAX_CONCURRENT_REQUESTS = 3
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

async def process_with_limit(payload):
    async with semaphore:
        return await process_query(payload)
```

### **Railway Configuration**

#### **1. Environment Variables**
```env
# Memory management
MAX_CACHE_SIZE=10
MAX_CONCURRENT_REQUESTS=3
CHUNK_SIZE=600
CONTEXT_LIMIT=2000

# Timeout settings
REQUEST_TIMEOUT=30
LLM_TIMEOUT=15
```

#### **2. Railway Service Configuration**
```yaml
# railway.toml or Railway dashboard settings
[build]
builder = "nixpacks"

[deploy]
startCommand = "python main.py"
healthcheckPath = "/"
healthcheckTimeout = 300

# Memory limits
memory = "512MB"  # Set memory limit
```

#### **3. Dockerfile Optimization**
```dockerfile
# Use slim base image
FROM python:3.9-slim

# Install only necessary packages
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Set memory limits
ENV PYTHONUNBUFFERED=1
ENV PYTHONOPTIMIZE=1

# Run with memory optimization
CMD ["python", "-O", "main.py"]
```

### **Memory Monitoring**

#### **1. Add Memory Monitoring Endpoint**
```python
# In main.py
import psutil
import os

@app.get("/health/memory")
async def memory_health():
    """Monitor memory usage."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        "memory_usage_mb": memory_info.rss / 1024 / 1024,
        "memory_percent": process.memory_percent(),
        "cache_size": len(document_cache),
        "available_memory_mb": psutil.virtual_memory().available / 1024 / 1024
    }
```

#### **2. Memory Usage Alerts**
```python
# In utils/logger.py
def check_memory_usage():
    """Check if memory usage is high."""
    process = psutil.Process(os.getpid())
    memory_percent = process.memory_percent()
    
    if memory_percent > 80:
        logger.warning(f"High memory usage: {memory_percent:.1f}%")
        # Clear cache if needed
        if len(document_cache) > 5:
            document_cache.clear()
            logger.info("Cache cleared due to high memory usage")
```

### **Performance vs Memory Trade-offs**

#### **Current Optimizations (Balanced)**
- ✅ **Response Time**: <30 seconds target
- ✅ **Memory Usage**: ~300-500MB
- ✅ **Accuracy**: 75%+ target maintained
- ✅ **Cache**: Smart document caching

#### **Memory-Heavy Mode (If needed)**
- ⚠️ **Response Time**: 30-45 seconds
- ✅ **Memory Usage**: ~200-300MB
- ✅ **Accuracy**: 75%+ target maintained
- ❌ **Cache**: Minimal caching

#### **Speed-Heavy Mode (If needed)**
- ✅ **Response Time**: 15-25 seconds
- ⚠️ **Memory Usage**: ~500-700MB
- ✅ **Accuracy**: 75%+ target maintained
- ✅ **Cache**: Aggressive caching

### **Railway Deployment Checklist**

#### **Before Deployment**
- [ ] Set memory limits in Railway dashboard
- [ ] Configure environment variables
- [ ] Test with memory monitoring
- [ ] Set up health checks

#### **After Deployment**
- [ ] Monitor memory usage via `/health/memory`
- [ ] Check response times
- [ ] Verify accuracy on test cases
- [ ] Monitor Railway logs for memory issues

#### **If Memory Issues Occur**
1. **Reduce cache size**: Set `MAX_CACHE_SIZE=5`
2. **Reduce chunk sizes**: Set `CHUNK_SIZE=500`
3. **Limit concurrent requests**: Set `MAX_CONCURRENT_REQUESTS=2`
4. **Clear cache periodically**: Add cache cleanup
5. **Upgrade Railway plan**: If needed for more memory

### **Testing Memory Usage**

#### **Local Testing**
```bash
# Monitor memory usage during testing
python -m memory_profiler test_deployed_endpoint.py
```

#### **Railway Testing**
```bash
# Test memory endpoint
curl https://teamaquariushackrx-production-1bbc.up.railway.app/health/memory
```

### **Expected Memory Usage**

#### **Normal Operation**
- **Idle**: ~200MB
- **Single Request**: ~250-300MB
- **Batch Request**: ~350-400MB
- **Peak Usage**: ~450-500MB

#### **Memory Alerts**
- **Warning**: >80% memory usage
- **Critical**: >90% memory usage
- **Action Required**: >95% memory usage

### **Optimization Commands**

#### **Quick Memory Test**
```bash
# Test current deployment
python test_deployed_endpoint.py

# Check memory usage
curl https://teamaquariushackrx-production-1bbc.up.railway.app/health/memory
```

#### **Memory Optimization**
```bash
# If memory issues occur, apply these settings:
export MAX_CACHE_SIZE=5
export CHUNK_SIZE=500
export MAX_CONCURRENT_REQUESTS=2
export CONTEXT_LIMIT=1500
```

This guide ensures your Railway deployment stays within memory limits while maintaining performance and accuracy targets! 🎯 