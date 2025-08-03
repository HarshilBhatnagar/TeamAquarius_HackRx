# HackRx - LLM-Powered Insurance Document Q&A

A state-of-the-art RAG system for insurance document processing with **Master-Slave Architecture** for high accuracy.

## 🚀 **Key Features**

### **Master-Slave Architecture**
- **Master Agent**: Orchestrates between specialized agents
- **Text Agent**: RAG-based text processing with enhanced retrieval
- **Table Agent**: PDF table parsing with structured analysis
- **Chain-of-Thought Synthesis**: Combines responses intelligently

### **Enhanced Accuracy**
- **Smart Question Analysis**: Routes to appropriate agent
- **Multi-Strategy Retrieval**: BM25 + Pinecone + keyword expansion
- **Table-Aware Processing**: Extracts structured data from tables
- **Comprehensive Context**: Up to 80 chunks, 15K characters

### **Document Processing**
- **PDF Table Extraction**: Using pdfplumber
- **Layout-Aware Parsing**: Multi-column, complex structures
- **Dynamic Chunking**: 2000 chars with 400 overlap
- **Policy-Specific Optimization**: Insurance document focus

## 🏗️ **Architecture**

```
Master Agent
├── Question Analysis → Route to appropriate agent
├── Text Agent (RAG) → Process text content
├── Table Agent (pdfplumber) → Process tables
└── COT Synthesis → Combine responses
```

## 🚀 **Quick Start**

### **API Endpoint**
```
POST /api/v1/hackrx/run
Authorization: Bearer YOUR_API_TOKEN
```

### **Request Format**
```json
{
  "documents": "https://example.com/policy.pdf",
  "questions": [
    "What is the waiting period for pre-existing diseases?",
    "Are organ donor expenses covered?"
  ]
}
```

### **Response Format**
```json
{
  "answers": [
    "The waiting period is 36 months...",
    "Yes, organ donor expenses are covered..."
  ]
}
```

## 🛠️ **Deployment**

### **Environment Variables**
```env
OPENAI_API_KEY=your-openai-key
API_AUTH_TOKEN=your-auth-token
PINECONE_API_KEY=your-pinecone-key
PINECONE_INDEX_NAME=hackrx-index
```

### **Deploy to Railway**
1. Fork/clone repository
2. Connect to [Railway](https://railway.app)
3. Set environment variables
4. Deploy automatically

### **Local Development**
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

## 📊 **Performance**

- **Target Accuracy**: 80%+ on insurance questions
- **Response Time**: < 30 seconds
- **Document Types**: PDF insurance policies
- **Question Types**: Coverage, benefits, exclusions, calculations

## 🔧 **Technical Stack**

- **Framework**: FastAPI
- **LLM**: OpenAI GPT-4o-mini
- **Retrieval**: BM25 + Pinecone
- **PDF Processing**: pdfplumber
- **Deployment**: Railway/Render/Heroku

---

**Built for HackRx Hackathon** 🏆
