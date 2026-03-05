# MCP RAG Agent

A local RAG (Retrieval-Augmented Generation) agent that combines:
- **Ollama** - Local LLM inference
- **ChromaDB** - Vector database for document storage
- **LangGraph** - Agentic AI orchestration (ReAct-style)
- **MCP (Model Context Protocol)** - Tool integration via Python SDK
- **Safe Computer Control** - Human-approved system control

## Features

- 🔒 **Fully Local** - All processing happens on your machine
- 📚 **Document RAG** - Semantic search over your documents
- 🛠️ **MCP Tools** - Modular tool architecture via MCP protocol
- 🤖 **Agentic** - ReAct-style agent decides when to search/read docs
- 📝 **Citations** - Answers include source file references
- 🎮 **Safe Control** - AI-assisted computer control with human approval
- 🛑 **Emergency Stop** - Halt all actions instantly

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed and running
- ~8GB RAM (for llama3.2 model)

## Quick Start

### 1. Install Ollama

```bash
# macOS
brew install ollama

# Or download from https://ollama.ai
```

Start Ollama and pull the model:

```bash
ollama serve  # Start in a separate terminal
ollama pull llama3.2
```

### 2. Set Up Python Environment

```bash
cd mcp-rag-agent

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env if needed (defaults work for most setups)
```

### 4. Add Your Documents

```bash
mkdir -p data
# Add .txt or .md files to the data/ directory
echo "This is a sample document about AI and machine learning." > data/sample.txt
```

### 5. Ingest Documents

```bash
python ingest.py
```

This will:
- Read all `.txt` and `.md` files from `./data`
- Split into ~800 character chunks with overlap
- Create embeddings using sentence-transformers
- Store in ChromaDB at `./chroma_db`

### 6. Ask Questions

```bash
python app.py "What topics are covered in my documents?"
python app.py "Summarize the main points from the documentation"
```

### 7. Safe Computer Control (NEW)

Launch the desktop approval UI:
```bash
python ui.py
```

Or use the CLI controller:
```bash
python safe_controller.py
```

## Project Structure

```
mcp-rag-agent/
├── app.py              # Main agent application
├── ingest.py           # Document ingestion script
├── mcp_server.py       # MCP server with RAG + Safe Control tools
├── ui.py               # Tkinter desktop approval UI
├── safe_controller.py  # CLI-based safe controller
├── security.py         # Security hardening (audit, rate limits)
├── requirements.txt    # Python dependencies
├── .env.example        # Environment template
├── .env                # Your configuration
├── data/               # Your documents go here
└── chroma_db/          # Vector database (auto-created)
```

## MCP Tools

The MCP server exposes three tools:

| Tool | Description |
|------|-------------|
| `search_docs` | Semantic search over document chunks |
| `list_docs` | List available documents in `./data` |
| `read_doc` | Read full content of a specific file |

## Configuration

Edit `.env` to customize:

```bash
# LLM settings
OLLAMA_MODEL=llama3.2          # Change model
OLLAMA_BASE_URL=http://localhost:11434

# Embedding
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Retrieval
DEFAULT_TOP_K=5
CHUNK_SIZE=800
CHUNK_OVERLAP=200
```

## Troubleshooting

### Ollama Not Running

```
✗ Ollama is not running at http://localhost:11434
```

**Solution:** Start Ollama in a separate terminal:
```bash
ollama serve
```

### Model Not Found

```
Error: model 'llama3.2' not found
```

**Solution:** Pull the model:
```bash
ollama pull llama3.2
```

### ChromaDB Not Found

```
✗ ChromaDB not found at ./chroma_db
```

**Solution:** Run ingestion first:
```bash
python ingest.py
```

### No Documents Found

```
No .txt or .md files found in the data directory.
```

**Solution:** Add documents to `./data`:
```bash
echo "Your content here" > data/example.txt
python ingest.py
```

### Import Errors

```
ModuleNotFoundError: No module named 'langchain'
```

**Solution:** Ensure you're in the virtual environment:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Security Notes

⚠️ **Important security considerations:**

1. **Local Only** - Ollama binds to `localhost:11434` by default. Do not expose this port publicly.

2. **Path Restrictions** - The MCP server only allows access to files within `./data`. Directory traversal attacks are blocked.

3. **No External Requests** - All processing happens locally. No data is sent to external APIs.

4. **ChromaDB Access** - The database at `./chroma_db` is only accessible locally.

### Best Practices

- Keep Ollama running only when needed
- Don't add sensitive documents to `./data` unless necessary
- Review the MCP tools' path validation in `mcp_server.py`
- Run in isolated environments for sensitive use cases

---

## 🎮 Safe Computer Control

The Safe Computer Control feature allows AI-assisted system control with **mandatory human approval**.

### Safety Guarantees

| Guarantee | Description |
|-----------|-------------|
| ✅ **No Autonomous Execution** | AI NEVER executes actions without approval |
| ✅ **Command Allowlist** | Only safe commands can be executed |
| ✅ **Emergency Stop** | Halt all actions instantly |
| ✅ **Audit Logging** | All actions are logged |
| ✅ **No Shell Injection** | Commands run without shell=True |

### Command Allowlist

Only these commands can be executed:

```
ls, pwd, open, echo, date, whoami, cat, head, tail
```

### Blocked Patterns

These are **always blocked**:

```
rm, sudo, chmod, chown, mv, cp, kill, pkill
Shell operators: | ; & > >> $ ` eval exec
```

### Approval Workflow

```
User Request → AI Proposes Plan → Human Reviews → Approve/Reject → Execute (if approved)
```

1. **User** submits a request
2. **AI** creates a structured action plan
3. **Human** reviews each proposed action
4. **Approve** → Action executes
5. **Reject** → No execution, plan discarded
6. **STOP** → Emergency halt of all actions

### Using the Desktop UI

```bash
python ui.py
```

Opens a native desktop window showing:
- Text input for your request
- Proposed action plan with ✓/✗ buttons per step
- Approve All / Reject All buttons
- Live execution log
- Emergency STOP button (always visible)

### Using the CLI

```bash
python safe_controller.py
```

Commands:
- Type a goal → Creates action plan
- `approve` → Approve pending plan
- `reject` → Reject pending plan
- `execute` → Execute approved plan
- `stop` → Emergency stop
- `log` → Show execution log
- `quit` → Exit

---

## Advanced Usage

### Using a Different Model

```bash
# Pull a different model
ollama pull mistral

# Update .env
OLLAMA_MODEL=mistral
```

### Larger Context Window

For longer documents, consider models with larger context:
```bash
ollama pull llama3.2:70b  # Requires more RAM
```

### Custom Embeddings

Change the embedding model in `.env`:
```bash
EMBEDDING_MODEL=all-mpnet-base-v2  # Larger, more accurate
```

Then re-run ingestion:
```bash
python ingest.py
```

## License

MIT
