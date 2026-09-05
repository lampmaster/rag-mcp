# Setup and Run Commands

Run these commands in order to set up and use the Company Knowledge Base Assistant.

## Prerequisites

1. **Install Python 3.10+** (if not already installed)
2. **Install Ollama** and pull the model:
   ```bash
   # macOS
   brew install ollama
   
   # Or download from https://ollama.ai
   
   # Pull the model
   ollama pull llama3
   ```

## Setup Steps

### 1. Navigate to the src directory
```bash
cd src
```

### 2. Create a virtual environment (recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Create documents directory
```bash
mkdir -p docs
```

### 5. Add company documentation
Add your company documentation files (`.txt`, `.md`, `.pdf`, `.docx`) to the `docs/` directory:
```bash
# Example: Copy some sample documents
# cp /path/to/company/docs/* docs/
```

### 6. Update configuration (optional)
Edit `config.py` if needed:
- Set `DOCUMENTS_DIR` to your documents path (default: `./docs`)
- Change `OLLAMA_MODEL` if using a different model
- Adjust `CHUNK_SIZE`, `CHUNK_OVERLAP`, or `TOP_K` as needed

### 7. Build the FAISS index (Optional)

The index will be built automatically on first use. To manually build it:

```bash
python main.py build-index
```

Or directly:
```bash
python -m rag.build_index
```

This will:
- Load all documents from the `docs/` directory
- Chunk them into smaller pieces (each chunk gets a stable `chunk_id`)
- Generate embeddings
- Build the FAISS index
- Build the SQLite FTS5 full-text index from the same chunks
- Save `index.faiss`, `chunks.pkl` and `fts_index.db`

## Running the tests

```bash
python -m pytest tests -q
```

## Benchmark (vector-only vs hybrid retrieval)

```bash
# from the repository root
python benchmark/run_benchmark.py --model qwen3:1.7b --rebuild
```

## Usage

### Interactive CLI Mode

Run the assistant interactively:
```bash
python main.py
```

Then ask questions like:
- "What is our vacation policy?"
- "How do I request time off?"
- "What are the company values?"

Type `exit` or `quit` to stop.

## Updating the Knowledge Base

When you add new documents or update existing ones:

1. Add/update files in the `docs/` directory
2. Rebuild the index:
   ```bash
   python main.py build-index
   ```

## Troubleshooting

### "Index not found" error
- The index will be built automatically on first use
- Or manually run `python main.py build-index`

### "No documents found"
- Check that `docs/` directory exists and contains files
- Verify `DOCUMENTS_DIR` in `config.py` is correct
- Ensure files have supported extensions (`.txt`, `.md`, `.pdf`, `.docx`)

### Retrieval looks purely semantic / full-text results are missing
- Rebuild the index so that `fts_index.db` is created: `python main.py build-index`
- Set `RAG_LOG_LEVEL=INFO` to see expansion keywords, per-retriever latencies and the fused chunk ids

### Ollama connection errors
- Make sure Ollama is running: `ollama list`
- Verify the model is installed: `ollama pull llama3`
- Check `OLLAMA_URL` in `config.py` (default: `http://localhost:11434/api/generate`)

### MCP client errors
- MCP tools are optional - the assistant will work without them
- If MCP fails, RAG will still function

### Import errors
- Make sure you're in the `src/` directory
- Verify virtual environment is activated
- Check that all dependencies are installed: `pip install -r requirements.txt`

## Quick Start Summary

```bash
# 1. Setup
cd src
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Prepare documents
mkdir -p docs
# Add your company documentation files to docs/

# 3. Build index
python main.py build-index

# 4. Run
python main.py
```
