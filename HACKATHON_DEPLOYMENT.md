# HackRx Hackathon Deployment Guide

## 🚀 Quick Deployment for Hackathon Submission

Your HackRx RAG API is ready for hackathon submission! This guide provides the fastest deployment options.

## 📋 Submission Requirements Checklist

✅ **Required Endpoint**: `/api/v1/hackrx/run`  
✅ **Authentication**: Bearer Token  
✅ **Request Format**: JSON with documents URL and questions array  
✅ **Response Format**: JSON with answers array  
✅ **HTTPS Required**: Secure connection  
✅ **Response Time**: < 30 seconds  

## 🎯 Your API Endpoint

**Main Endpoint**: `POST /api/v1/hackrx/run`  
**Authentication**: `Authorization: Bearer YOUR_API_TOKEN`

## 🚀 Deployment Options

### Option 1: Railway (Recommended - Fastest)

1. **Fork/Clone your repository**
2. **Connect to Railway**:
   - Go to [railway.app](https://railway.app)
   - Connect your GitHub repository
   - Railway will auto-detect FastAPI

3. **Set Environment Variables**:
   ```env
   OPENAI_API_KEY=your-openai-api-key
   API_AUTH_TOKEN=your-api-auth-token
   PINECONE_API_KEY=your-pinecone-api-key
   PINECONE_INDEX_NAME=hackrx-index
   ```

4. **Deploy**: Railway will automatically deploy your app
5. **Get URL**: Your API will be available at `https://your-app.railway.app`

### Option 2: Render

1. **Connect to Render**:
   - Go to [render.com](https://render.com)
   - Connect your GitHub repository
   - Choose "Web Service"

2. **Configure**:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Environment**: Python 3.11

3. **Set Environment Variables** (same as above)
4. **Deploy**: Render will build and deploy automatically

### Option 3: Heroku

1. **Install Heroku CLI**
2. **Deploy**:
   ```bash
   heroku create your-hackrx-app
   git push heroku main
   ```

3. **Set Environment Variables**:
   ```bash
   heroku config:set OPENAI_API_KEY=your-key
   heroku config:set API_AUTH_TOKEN=your-token
   heroku config:set PINECONE_API_KEY=your-key
   heroku config:set PINECONE_INDEX_NAME=hackrx-index
   ```

## 📝 Submission Format

### Your Webhook URL
```
https://your-app.railway.app/api/v1/hackrx/run
```

### Test Request
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

## 🔧 Environment Setup

### Required API Keys

1. **OpenAI API Key**:
   - Get from [platform.openai.com](https://platform.openai.com)
   - Required for LLM processing

2. **Pinecone API Key**:
   - Get from [pinecone.io](https://pinecone.io)
   - Required for vector search

3. **Authentication Token**:
   - Use your own secure token for authentication

## ✅ Pre-Submission Checklist

- [ ] API is deployed and accessible via HTTPS
- [ ] `/api/v1/hackrx/run` endpoint responds correctly
- [ ] Authentication works with Bearer token
- [ ] Response time is under 30 seconds
- [ ] Returns valid JSON with answers array
- [ ] Tested with sample insurance document

## 🎯 Hackathon Evaluation Features

Your API includes all required features:

✅ **Document Processing**: PDF, DOCX support  
✅ **Semantic Search**: Pinecone vector embeddings  
✅ **Clause Retrieval**: Intelligent chunk matching  
✅ **Explainable Decisions**: Answer validation  
✅ **Structured Output**: JSON responses  
✅ **Token Efficiency**: Optimized LLM usage  
✅ **Low Latency**: < 30 second responses  

## 🚨 Troubleshooting

### Common Issues:

1. **Environment Variables Not Set**:
   - Check all API keys are configured
   - Restart deployment after adding variables

2. **Import Errors**:
   - Ensure all dependencies in `requirements.txt`
   - Check Python version compatibility

3. **Timeout Issues**:
   - Optimize chunk sizes if needed
   - Check API key validity

### Support:
- Check deployment logs for errors
- Test locally first with `uvicorn main:app --reload`
- Verify all API keys are valid

## 🏆 Ready for Submission!

Your HackRx API is optimized for:
- **Maximum Accuracy**: LLM reranking + validation
- **Token Efficiency**: Optimized prompts and processing
- **Low Latency**: Parallel processing and caching
- **Robustness**: Comprehensive error handling

**Submit your webhook URL**: `https://your-app.railway.app/api/v1/hackrx/run` 
