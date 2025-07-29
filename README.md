# HackRx RAG API - Hackathon Version

A high-performance Retrieval-Augmented Generation (RAG) API designed for insurance document processing, optimized for hackathon evaluation.

## 🚀 Key Features

### **Enhanced Accuracy with LLM Reranking**
- **LLM-Based Reranking**: Uses GPT-4o-mini to score and rank chunk relevance
- **Hybrid Search**: Combines BM25 keyword search with Pinecone semantic search
- **Intelligent Context Selection**: Automatically selects the most relevant chunks for answer generation

### **Answer Validation Layer**
- **AI-Powered Fact Checking**: Uses GPT-4o-mini to validate generated answers against context
- **Self-Correction Loop**: Automatically corrects answers that don't align with the provided context
- **Fallback Safety**: Returns safe responses when validation fails

### **Optimized for Insurance Documents**
- **PDF Processing**: Advanced PDF parsing with table-aware extraction
- **Policy Understanding**: Specialized for insurance policy document processing
- **Clause Retrieval**: Intelligent matching of policy clauses and conditions

## 📋 API Endpoint

### Main RAG Pipeline
- `POST /api/v1/hackrx/run` - Main endpoint for hackathon evaluation

## 🔧 Environment Configuration

Create a `.env` file with the following variables:

```env
# OpenAI API Configuration
OPENAI_API_KEY="your-openai-api-key-here"

# Authentication
API_AUTH_TOKEN="your-api-auth-token-here"

# Pinecone Vector Database
PINECONE_API_KEY="your-pinecone-api-key-here"
PINECONE_INDEX_NAME="hackrx-index"

# Optional: Logging Level
LOG_LEVEL="INFO"
```

## 🏗️ Architecture

### Processing Pipeline

1. **Document Ingestion**
   - PDF parsing with table-aware extraction using pdfplumber
   - Intelligent text chunking for optimal retrieval
   - Vector embedding and storage in Pinecone

2. **Enhanced Retrieval**
   - Ensemble retriever combining BM25 and semantic search
   - LLM-based reranking for precision filtering
   - Context optimization for maximum relevance

3. **Intelligent Generation**
   - GPT-4o for high-quality answer generation
   - Answer validation using GPT-4o-mini
   - Self-correction for improved accuracy

## 🚀 Performance Optimizations

### Accuracy Enhancements
- **LLM Reranking**: AI-powered chunk relevance scoring
- **Answer Validation**: AI-powered fact-checking
- **Hybrid Search**: Combines keyword and semantic search
- **Context Optimization**: Intelligent chunk selection

### Robustness Features
- **Error Handling**: Comprehensive error handling and fallbacks
- **Graceful Degradation**: System continues working even if optional services fail
- **Logging**: Detailed logging for debugging and monitoring
- **Security**: Bearer token authentication

## 📦 Installation & Deployment

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp env.template .env
# Edit .env with your API keys

# Run the application
uvicorn main:app --reload
```

### Docker Deployment
```bash
# Build and run with Docker
docker build -t hackrx-api .
docker run -p 8000:8000 hackrx-api
```

## 📊 Usage Examples

### Hackathon Submission Request
```bash
curl -X POST "https://your-app.railway.app/api/v1/hackrx/run" \
  -H "Authorization: Bearer YOUR_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
      "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
      "What is the waiting period for pre-existing diseases (PED) to be covered?"
    ]
  }'
```

### Expected Response
```json
{
  "answers": [
    "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.",
    "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered."
  ]
}
```

## 🎯 Performance Metrics

- **Response Time**: < 30 seconds for new documents
- **Accuracy**: Enhanced through LLM reranking and validation
- **Throughput**: Parallel processing of multiple questions
- **Reliability**: Graceful error handling and fallbacks

## 🔍 Monitoring & Debugging

- **Logging**: Comprehensive logging at all levels
- **Token Usage**: Tracked in response headers
- **Error Tracking**: Detailed error messages and stack traces

## 🧪 Testing

Run the test script to verify your API:
```bash
python test_enhanced_api.py
```

## 🏆 Hackathon Evaluation Features

Your API includes all required features:

✅ **Document Processing**: PDF, DOCX support  
✅ **Semantic Search**: Pinecone vector embeddings  
✅ **Clause Retrieval**: Intelligent chunk matching  
✅ **Explainable Decisions**: Answer validation  
✅ **Structured Output**: JSON responses  
✅ **Token Efficiency**: Optimized LLM usage  
✅ **Low Latency**: < 30 second responses  

## 📄 License

This project is developed for the Bajaj Hackathon and follows the specified requirements and constraints.
