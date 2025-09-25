# 🤖 Agentic RAG POC

<div align="center">

[![CrewAI](https://img.shields.io/badge/CrewAI-🚢-blue?style=for-the-badge&logo=docker&logoColor=white)](https://crewai.com)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-🦙-green?style=for-the-badge&logo=python&logoColor=white)](https://llamaindex.ai)
[![Ollama](https://img.shields.io/badge/Ollama-🦙-orange?style=for-the-badge&logo=llama&logoColor=white)](https://ollama.ai)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-🐘-blue?style=for-the-badge&logo=postgresql&logoColor=white)](https://postgresql.org)
[![Docker](https://img.shields.io/badge/Docker-🐳-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)

**🎯 An intelligent agentic RAG system with multi-agent workflows, powered by CrewAI and local LLMs**

</div>

---

## 🌟 Features

<table>
<tr>
<td>

### 🧠 **AI-Powered Intelligence**
- 🤖 **Multi-Agent Workflow** - Specialized agents for research & synthesis
- 🦙 **Local LLM Integration** - Ollama with Gemma 3 (131K context)
- 🎯 **Contextual Responses** - AI-generated context for better understanding
- 📊 **Smart Document Processing** - PDF/DOCX parsing with DoclingReader

</td>
<td>

### ⚡ **Performance & Scale**
- 🚀 **Vector Search** - PostgreSQL + pgvector for lightning-fast retrieval
- 🔄 **Async Processing** - Concurrent document processing
- 📈 **Optimized Retrieval** - Hybrid search (semantic + keyword)
- 💾 **Efficient Storage** - Compressed embeddings and metadata

</td>
</tr>
<tr>
<td>

### 🌐 **User Experience**
- 💬 **OpenWebUI Integration** - Beautiful chat interface
- 🔗 **OpenAI-Compatible API** - Standard REST endpoints
- 📚 **Source Attribution** - Transparent document references
- 🎨 **Rich Formatting** - Professional response presentation

</td>
<td>

### 🔧 **DevOps Ready**
- 🐳 **Full Containerization** - Docker + Docker Compose
- 📊 **Observability** - Arize Phoenix tracing integration
- 🔍 **Comprehensive Logging** - Real-time monitoring
- 🛡️ **Production Ready** - CORS, error handling, graceful failures

</td>
</tr>
</table>

---

## 🏗️ Architecture

```mermaid
graph TB
    subgraph "📄 Data Ingestion Pipeline"
        RAW["📁 Raw Documents<br/>PDFs, Word, etc."]
        DOC["🔧 Docling Converter<br/>PDF → JSON/Markdown"]
        CTX["🧠 Contextual RAG<br/>LlamaIndex + Context Generation"]
        CHUNK["✂️ Smart Chunking<br/>Markdown Parser + Token Splitter"]
    end
    
    subgraph "🌐 User Interface"
        UI["🖥️ OpenWebUI"]
    end
    
    subgraph "🚢 API Layer"
        API["⚡ FastAPI Server<br/>OpenAI Compatible"]
    end
    
    subgraph "🤖 Agent Layer"
        CR["📋 CrewAI Orchestrator"]
        A1["🔍 Document Researcher"]
        A2["🧠 Insight Synthesizer"]
    end
    
    subgraph "🔍 Retrieval Layer"
        VDB["🐘 PostgreSQL + pgvector<br/>Vector Database<br/>768-dim Embeddings"]
        LLM["🦙 Ollama LLM<br/>Gemma 3 - 131K context"]
        EMB["🔤 Ollama Embeddings<br/>nomic-embed-text"]
    end
    
    subgraph "📊 Observability & Evaluation"
        PHX["🐦 Arize Phoenix<br/>Tracing & Monitoring"]
        RAGAS["📈 RAGAS Evaluation<br/>Faithfulness, Relevancy<br/>Context Precision/Recall"]
    end
    
    %% Data Flow
    RAW --> DOC
    DOC --> CTX
    CTX --> CHUNK
    CHUNK --> EMB
    EMB --> VDB
    
    %% Query Flow
    UI --> API
    API --> CR
    CR --> A1 & A2
    A1 --> VDB
    A2 --> LLM
    VDB --> LLM
    
    %% Evaluation Flow
    CR --> RAGAS
    VDB -.-> RAGAS
    
    %% Monitoring
    API -.-> PHX
    CR -.-> PHX
    CTX -.-> PHX
    
    style RAW fill:#ffebee
    style DOC fill:#f3e5f5
    style CTX fill:#e8f5e8
    style CHUNK fill:#fff3e0
    style UI fill:#e1f5fe
    style API fill:#f3e5f5
    style CR fill:#fff3e0
    style VDB fill:#e8f5e8
    style PHX fill:#fce4ec
    style RAGAS fill:#e3f2fd
    style EMB fill:#f1f8e9
```

### 🏗️ Architecture Components

| Component | Purpose | Technology Stack |
|-----------|---------|------------------|
| **📄 Data Ingestion Pipeline** | Document processing and vectorization | |
| ├── 📁 Raw Documents | Source files (PDFs, Word docs, etc.) | File system |
| ├── 🔧 Docling Converter | Document parsing and conversion | [Docling](https://github.com/DS4SD/docling) |
| ├── 🧠 Contextual RAG | Context-aware chunking with LLM | LlamaIndex + Ollama |
| └── ✂️ Smart Chunking | Intelligent text segmentation | Markdown Parser + Token Splitter |
| **🌐 User Interface** | Web-based chat interface | OpenWebUI |
| **🚢 API Layer** | OpenAI-compatible REST API | FastAPI |
| **🤖 Agent Layer** | Multi-agent orchestration | CrewAI Framework |
| ├── 🔍 Document Researcher | Retrieval and context gathering | Vector similarity search |
| └── 🧠 Insight Synthesizer | Response generation and formatting | LLM-powered synthesis |
| **🔍 Retrieval Layer** | Knowledge storage and retrieval | |
| ├── 🐘 PostgreSQL + pgvector | Vector database for embeddings | PostgreSQL 15+ with pgvector |
| ├── 🦙 Ollama LLM | Large language model (131K context) | Gemma 3:4b |
| └── 🔤 Ollama Embeddings | Text-to-vector conversion | nomic-embed-text (768-dim) |
| **📊 Observability & Evaluation** | System monitoring and quality assurance | |
| ├── 🐦 Arize Phoenix | Request tracing and performance monitoring | Phoenix Observability |
| └── 📈 RAGAS Evaluation | RAG system quality metrics | RAGAS Framework |

### 🔄 Data Flow Explained

1. **📥 Ingestion Phase**:
   - Raw documents are processed by Docling for parsing
   - Contextual RAG enhances chunks with semantic context
   - Smart chunking optimizes content for retrieval
   - Embeddings are generated and stored in pgvector

2. **🔍 Query Phase**:
   - User queries via OpenWebUI → FastAPI → CrewAI
   - Document Researcher retrieves relevant context
   - Insight Synthesizer generates comprehensive responses
   - Results formatted and returned to user

3. **📊 Evaluation Phase**:
   - RAGAS evaluates system performance across multiple metrics
   - Monitors faithfulness, relevancy, and context quality
   - Provides continuous feedback for system improvement

---

## 🚀 Complete Setup Guide

### 📋 Prerequisites

<table>
<tr>
<td>

**🔧 Required Components**
- 🐳 Docker & Docker Compose
- 🦙 Ollama (Local LLM)
- 🐍 Python 3.8+ (optional)

</td>
<td>

**💻 System Requirements**
- 8GB+ RAM (recommended)
- 50GB+ disk space
- macOS/Linux/Windows

</td>
</tr>
</table>

---

## ⚡ Quick Start (5 Steps)

### 1️⃣ **Setup Ollama**

```bash
# 🦙 Start Ollama with network access
OLLAMA_HOST=0.0.0.0 ollama serve
```

### 2️⃣ **Build & Deploy RAG API**

```bash
# 🏗️ Build the Docker image
docker build -t agentic-rag-api .

# 🚀 Deploy RAG API container
docker run --name rag-api -d \
  --network rag-network \
  -p 8000:8000 \
  --env-file .env.docker \
  agentic-rag-api
```

### 3️⃣ **Deploy OpenWebUI**

```bash
# 🌐 Launch OpenWebUI interface
docker run --name open-webui -d \
  --network rag-network \
  -p 3000:8080 \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -v open-webui:/app/backend/data \
  ghcr.io/open-webui/open-webui:main
```

### 4️⃣ **Setup Arize Phoenix Tracing**

```bash
# 📊 Deploy monitoring & tracing
docker run -d -p 6006:6006 --name phoenix-ui arizephoenix/phoenix:latest
```

### 5️⃣ **Access Your Applications**

| Service | URL | Description |
|---------|-----|-------------|
| 🌐 **OpenWebUI** | [`http://localhost:3000`](http://localhost:3000) | Main chat interface |
| ⚡ **RAG API** | [`http://localhost:8000`](http://localhost:8000) | API endpoints |
| 📊 **Phoenix UI** | [`http://localhost:6006`](http://localhost:6006) | Tracing & monitoring |

---

## 🔍 Monitoring & Debugging

### 📊 **Real-Time Logs**

#### 🚢 RAG API Container Logs
```bash
# 📋 View real-time logs from the RAG API container
docker logs -f rag-api

# 📜 View last 50 lines of logs
docker logs --tail 50 rag-api

# ⏰ View logs with timestamps
docker logs -t rag-api
```

#### 🌐 OpenWebUI Container Logs
```bash
# 🖥️ View OpenWebUI logs
docker logs -f open-webui

# 📜 View last 50 lines
docker logs --tail 50 open-webui
```

### 🔧 **Connection Testing**

#### 🦙 Test Ollama Connection from Container
```bash
# 🧪 Test if the container can reach Ollama
docker exec -it rag-api curl http://host.docker.internal:11434/api/tags

# 🔄 Alternative test
docker exec -it rag-api curl http://localhost:11434/api/tags
```

#### 🌐 Check Container Network Connectivity
```bash
# 🕸️ Inspect the network configuration
docker network inspect rag-network

# 🔍 Check if containers are on the right network
docker inspect rag-api | grep -A 10 "Networks"
docker inspect open-webui | grep -A 10 "Networks"
```

#### 🏠 Check Host Ollama Status
```bash
# ✅ Verify Ollama is running on your host
curl http://localhost:11434/api/tags

# 📋 Check if models are loaded
ollama list

# 🔍 Check Ollama process
ps aux | grep ollama
```

---

## 🛠️ Configuration

### � **Environment Variables**

Create your environment files:

#### 📁 `.env` (Local Development)
```bash
DATABASE_URL=postgresql://postgres:password@localhost:5432/rag_db
OLLAMA_BASE_URL=http://localhost:11434
```

#### 🐳 `.env.docker` (Container Environment)
```bash
DATABASE_URL=postgresql://postgres:password@host.docker.internal:5432/rag_db
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

### ⚙️ **Model Configuration**

Current setup uses **Gemma 3 (4B)** with maximum token configuration:
- 🎯 **Context Length**: 131,072 tokens
- 🚀 **Max Output**: 131,072 tokens
- 🔥 **Temperature**: 0.1 (precise responses)

---

## � Data Ingestion & Processing

### 🔧 **Document Processing Pipeline**

#### **Step 1: Document Conversion with Docling**
```bash
# 🔄 Convert PDFs and Word documents to structured format
cd /Users/kiwitech/Documents/agentic-rag-poc
python src/data_ingestion/ingestion_docling.py
```

**What it does:**
- ✨ Converts PDFs, Word docs to JSON and Markdown
- 📁 Processes files from `data/raw/` → `data/processed/`
- 🧹 De-duplicates and handles multiple file formats
- 📊 Preserves document structure and metadata

#### **Step 2: Contextual RAG Indexing**
```bash
# 🧠 Generate embeddings with contextual enhancement
python src/data_ingestion/ingest_contextual_rag.py
```

**Features:**
- 🔤 **Contextual Chunking**: LLM-enhanced context for each chunk
- 🗄️ **Vector Storage**: PostgreSQL + pgvector with 768-dim embeddings
- ⚡ **Smart Parsing**: Markdown-aware token splitting
- 📈 **Scalable**: Async processing for large document sets

### 📊 **Quality Evaluation with RAGAS**

#### **Run Comprehensive Evaluation**
```bash
# 📈 Evaluate RAG system performance
python src/evaluation/run_ragas_eval.py
```

**Metrics Tracked:**
- 🎯 **Faithfulness**: How factually accurate are responses?
- 🔍 **Answer Relevancy**: Does the answer address the question?
- 📚 **Context Recall**: Did retrieval find all relevant information?
- 🎪 **Context Precision**: How precise is the retrieved context?

**Sample Output:**
```yaml
Evaluation Results:
- Faithfulness: 0.92
- Answer Relevancy: 0.88
- Context Recall: 0.85
- Context Precision: 0.90
```

### 📁 **Data Directory Structure**
```
data/
├── raw/                          # Source documents
│   ├── Abu Dhabi Procurement Standards.PDF
│   ├── HR Bylaws.pdf
│   └── Procurement Manual.PDF
├── processed/
│   ├── json/                     # Structured JSON output
│   └── md/                       # Markdown conversions
└── evaluation/
    └── eval_dataset.jsonl        # Golden dataset for testing
```

---

## �📚 Usage Examples

### 💬 **Via OpenWebUI**
1. Open [`http://localhost:3000`](http://localhost:3000)
2. Select "crew-ai-rag" model
3. Ask questions about your documents

### 🔗 **Via API**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "crew-ai-rag",
    "messages": [
      {
        "role": "user", 
        "content": "What is the maximum salary deduction for disciplinary penalties?"
      }
    ]
  }'
```

---

## 🐛 Troubleshooting

<details>
<summary><strong>🔧 Common Issues & Solutions</strong></summary>

### 🚫 **Container Cannot Connect to Ollama**
- Ensure Ollama is running with `OLLAMA_HOST=0.0.0.0`
- Check if `host.docker.internal` resolves correctly
- Verify network configuration

### 📊 **Models Not Showing in OpenWebUI**
- Check API logs: `docker logs rag-api`
- Verify API is accessible: `curl http://localhost:8000/v1/models`
- Restart containers if needed

### 🐘 **Database Connection Issues**
- Ensure PostgreSQL container is running
- Check database credentials in environment files
- Verify network connectivity between containers

</details>

---

## 🏗️ Development

### 🔨 **Local Development Setup**

```bash
# 🐍 Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# 📦 Install dependencies
pip install -r requirements.txt

# 🚀 Run locally
python main.py "Your question here"
```

### 🧪 **Running Tests**

```bash
# 🧪 Run evaluation tests
python src/evaluation/run_ragas_eval.py
```

---

## 🤝 Contributing

1. 🍴 Fork the repository
2. 🌿 Create a feature branch (`git checkout -b feature/amazing-feature`)
3. 💾 Commit changes (`git commit -m 'Add amazing feature'`)
4. 📤 Push to branch (`git push origin feature/amazing-feature`)
5. 🔄 Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- 🚢 [CrewAI](https://crewai.com) - Multi-agent orchestration
- 🦙 [LlamaIndex](https://llamaindex.ai) - Document processing & retrieval
- 🦙 [Ollama](https://ollama.ai) - Local LLM inference
- 🐘 [PostgreSQL](https://postgresql.org) + [pgvector](https://github.com/pgvector/pgvector) - Vector database
- 🌐 [OpenWebUI](https://openwebui.com) - Beautiful chat interface
- 🐦 [Arize Phoenix](https://phoenix.arize.com) - Observability & tracing

---

<div align="center">

**⭐ Star this repo if it helped you! ⭐**

[🐛 Report Bug](https://github.com/syedasad-kiwi/agentic-rag-poc/issues) • [✨ Request Feature](https://github.com/syedasad-kiwi/agentic-rag-poc/issues) • [💬 Discussions](https://github.com/syedasad-kiwi/agentic-rag-poc/discussions)

</div>

#### 1. Clone the Repository

```bash
git clone https://github.com/syedasad-kiwi/agentic-rag-poc.git
cd agentic-rag-poc
```

#### 2. Install and Setup Ollama

```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull required models
ollama pull gemma2:9b          # Main LLM for responses
ollama pull nomic-embed-text   # Embedding model
```

#### 3. Setup PostgreSQL with Docker

```bash
# Start PostgreSQL with pgvector
docker-compose up -d

# Verify PostgreSQL is running
docker ps | grep postgres
```

#### 4. Configure Environment

Create environment files:

```bash
# Create .env file for local development
cat > .env << EOF
DATABASE_URL=postgresql://postgres:password@localhost:5432/rag_db
OLLAMA_BASE_URL=http://localhost:11434
EOF

# Create .env.docker for container environment  
cat > .env.docker << EOF
DATABASE_URL=postgresql://postgres:password@host.docker.internal:5432/rag_db
OLLAMA_BASE_URL=http://host.docker.internal:11434
EOF
```

#### 5. Document Ingestion

```bash
# Install Python dependencies locally for ingestion
pip install -r requirements.txt

# Place your documents in data/raw/ and run ingestion
python src/data_ingestion/ingest.py
```

#### 6. Build and Deploy RAG API

```bash
# Build the RAG API Docker image
docker build -t rag-api .

# Create Docker network for service communication
docker network create rag-network

# Run RAG API container
docker run --name rag-api -d \
  --network rag-network \
  -p 8000:8000 \
  --env-file .env.docker \
  rag-api
```

#### 7. Deploy OpenWebUI

```bash
# Run OpenWebUI container
docker run --name open-webui -d \
  --network rag-network \
  -p 3000:8080 \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -v open-webui:/app/backend/data \
  ghcr.io/open-webui/open-webui:main
```

#### 8. Configure OpenWebUI

1. Open your browser to `http://localhost:3000`
2. Complete the initial setup (create admin account)
3. Go to **Settings** → **Connections** → **OpenAI API**
4. Set the **API Base URL** to: `http://rag-api:8000/v1`
5. Leave the **API Key** field empty
6. Click **Save**
7. The RAG models should now appear in the model dropdown

### Option 2: Local Development Setup
For development without Docker:

#### 1. Python Environment Setup

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### 2. Set Up PostgreSQL with pgvector

##### Option A: Using Docker (Recommended)

```bash
docker-compose up -d
```

##### Option B: Local Installation

1. Install PostgreSQL 14+
2. Install pgvector extension
3. Create database and enable extension

#### 3. Set Up Ollama

```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull required models
ollama pull gemma2:9b
ollama pull nomic-embed-text
```

#### 4. Environment Configuration

Create a `.env` file in the root directory:

```env
DATABASE_URL=postgresql://postgres:password@localhost:5432/rag_db
OLLAMA_BASE_URL=http://localhost:11434
```

## 📊 Usage

### Using the Web Interface (Recommended)

1. **Access OpenWebUI**: Navigate to `http://localhost:3000`
2. **Select Model**: Choose `rag-model` or `crew-ai-rag` from the dropdown
3. **Ask Questions**: Type your questions about the ingested documents
4. **View Responses**: Get intelligent responses with source attributions

### Using the API Directly

```bash
# Test the models endpoint
curl http://localhost:8000/v1/models

# Send a chat completion request
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rag-model",
    "messages": [
      {"role": "user", "content": "What are the key procurement requirements?"}
    ]
  }'
```

### Command Line Usage

```bash
python main.py
```

### Programmatic Usage

```python
from src.rag_system.crew import create_rag_crew

# Create and run the crew
crew = create_rag_crew('What are the key procurement requirements?')
result = crew.kickoff()
print(result)
```

## 🔧 Document Ingestion

The system includes an enhanced ingestion pipeline with AI-generated context:

### 1. Place Documents

Place your documents (PDF/DOCX) in the `data/raw/` directory.

### 2. Run Ingestion

```bash
python src/data_ingestion/ingest.py
```

### Features:
- **DoclingReader**: Advanced document parsing
- **Sentence Splitter**: Intelligent text chunking  
- **AI Context Generation**: Each chunk gets 2-line context using Gemma LLM
- **Vector Storage**: Embeddings stored in PostgreSQL with pgvector

## � Docker Commands Reference

### Container Management

```bash
# View running containers
docker ps

# View container logs
docker logs rag-api
docker logs open-webui

# Restart containers
docker restart rag-api
docker restart open-webui

# Stop and remove containers
docker stop rag-api open-webui
docker rm rag-api open-webui

# Rebuild RAG API
docker build -t rag-api .
docker run --name rag-api -d --network rag-network -p 8000:8000 --env-file .env.docker rag-api
```

### Network Management

```bash
# Create network
docker network create rag-network

# View networks
docker network ls

# Inspect network
docker network inspect rag-network
```

### Troubleshooting

```bash
# Check if services are running on correct ports
lsof -i :8000  # RAG API
lsof -i :3000  # OpenWebUI
lsof -i :5432  # PostgreSQL

# Test API connectivity
curl http://localhost:8000/v1/models

# View detailed container logs
docker logs -f rag-api
```

## 🔍 Key Features Explained

### Enhanced Document Processing

The system now includes AI-generated context for each document chunk:

```python
# Each chunk gets contextual information
chunk_text = "Procurement standards require..."
ai_context = "This section outlines mandatory procurement compliance requirements for all departments."
```

### OpenAI-Compatible API

FastAPI server with OpenAI-compatible endpoints:

- `GET /v1/models` - List available models
- `POST /v1/chat/completions` - Chat completion endpoint

### Source Attribution

The system automatically extracts and includes source file information in responses:

```python
# Example response format
"Based on the Abu Dhabi Procurement Standards document..."

Sources:
- Abu Dhabi Procurement Standards.PDF
- Procurement Manual (Business Process).PDF
```

### CORS Support

Full CORS support for web interface integration:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 📁 Project Structure

```
agentic-rag-poc/
├── data/
│   ├── raw/                 # Input documents (PDF/DOCX)
│   └── processed/           # Processed data
├── src/
│   ├── config/             # Configuration settings
│   ├── data_ingestion/     # Document parsing and ingestion
│   │   └── ingest.py       # Enhanced ingestion with AI context
│   ├── evaluation/         # Evaluation scripts and datasets
│   └── rag_system/         # Core RAG implementation
│       ├── agents.py       # CrewAI agent definitions
│       ├── crew.py         # Workflow orchestration
│       └── tools.py        # Custom tools and utilities
├── notebooks/              # Jupyter notebooks for exploration
├── api.py                  # FastAPI server with OpenAI compatibility
├── main.py                 # Main application entry point
├── Dockerfile              # Docker configuration for RAG API
├── requirements.txt        # Python dependencies
├── requirements-docker.txt # Docker-specific dependencies
├── docker-compose.yml      # PostgreSQL setup
├── .env                    # Local environment variables
└── .env.docker            # Docker environment variables
```

## 🧪 Testing & Evaluation

### API Testing

```bash
# Test models endpoint
curl http://localhost:8000/v1/models

# Test chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rag-model",
    "messages": [{"role": "user", "content": "What are the procurement standards?"}]
  }'
```

### Evaluation Suite

```bash
python src/evaluation/run_ragas_eval.py
```

### Jupyter Notebooks

Explore the system using the provided notebooks:

- `notebooks/01_parsing_and_chunking.ipynb` - Document processing exploration
- `notebooks/02_agent_and_evaluation.ipynb` - Agent testing and evaluation

## ⚠️ Troubleshooting

### Common Issues

**1. Models not appearing in OpenWebUI dropdown:**
- Verify RAG API is running: `curl http://localhost:8000/v1/models`
- Check OpenWebUI connection settings: `http://rag-api:8000/v1`
- Restart OpenWebUI container: `docker restart open-webui`

**2. Docker connectivity issues:**
- Ensure all containers are on the same network: `docker network inspect rag-network`
- Check container logs: `docker logs rag-api` and `docker logs open-webui`
- Verify environment files are configured correctly

**3. Database connection errors:**
- Ensure PostgreSQL container is running: `docker ps | grep postgres`
- Check database URL in environment files
- Verify pgvector extension is installed

**4. Ollama model errors:**
- Ensure Ollama is running: `ollama list`
- Pull missing models: `ollama pull gemma2:9b` and `ollama pull nomic-embed-text`
- Check Ollama base URL in environment files

### Logs and Debugging

```bash
# View all container logs
docker logs rag-api -f
docker logs open-webui -f
docker logs postgres -f

# Check network connectivity
docker exec rag-api ping open-webui
docker exec open-webui ping rag-api

# Test database connection
docker exec postgres psql -U postgres -d rag_db -c "SELECT count(*) FROM data_document_embeddings;"
```

## 🔒 Security Considerations

For production deployment, consider:

- Remove `allow_origins=["*"]` and specify allowed origins
- Add API authentication and rate limiting
- Use environment-specific configuration files
- Enable PostgreSQL SSL/TLS
- Implement proper logging and monitoring

## 🚀 Production Deployment

### Docker Compose Production Setup

Create a `docker-compose.prod.yml`:

```yaml
version: '3.8'
services:
  postgres:
    image: pgvector/pgvector:pg14
    environment:
      POSTGRES_DB: rag_db
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - rag-network

  rag-api:
    build: .
    environment:
      DATABASE_URL: postgresql://postgres:${POSTGRES_PASSWORD}@postgres:5432/rag_db
      OLLAMA_BASE_URL: ${OLLAMA_BASE_URL}
    depends_on:
      - postgres
    networks:
      - rag-network

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - rag-api
    networks:
      - rag-network

volumes:
  postgres_data:

networks:
  rag-network:
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [CrewAI](https://github.com/joaomdmoura/crewAI) for the multi-agent framework
- [LlamaIndex](https://github.com/run-llama/llama_index) for RAG utilities and DoclingReader
- [Ollama](https://ollama.ai/) for local LLM inference
- [pgvector](https://github.com/pgvector/pgvector) for vector similarity search
- [OpenWebUI](https://github.com/open-webui/open-webui) for the web interface
- [FastAPI](https://github.com/tiangolo/fastapi) for the API framework

## 📞 Support

For questions and support, please open an issue on GitHub or contact the maintainers.

---

**Note**: This is a proof-of-concept implementation. For production use, consider additional security measures, error handling, and scalability optimizations.
