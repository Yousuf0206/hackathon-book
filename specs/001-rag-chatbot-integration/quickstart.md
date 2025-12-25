# Quickstart Guide: RAG Chatbot Integration

## Prerequisites
- Python 3.11+
- Node.js 18+ (for Docusaurus frontend)
- Cohere API key
- OpenRouter API key
- Qdrant Cloud account
- Neon Postgres account

## Setup Backend

1. **Install dependencies**:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Set environment variables**:
   ```bash
   export COHERE_API_KEY="your-cohere-key"
   export OPENROUTER_API_KEY="your-openrouter-key"
   export QDRANT_URL="your-qdrant-url"
   export QDRANT_API_KEY="your-qdrant-key"
   export DATABASE_URL="your-neon-postgres-url"
   ```

3. **Initialize the vector store**:
   ```bash
   python -m tools.content_loader --source-url https://your-book-url.com
   ```

4. **Run the backend**:
   ```bash
   uvicorn src.api.main:app --reload --port 8000
   ```

## Integrate with Frontend

1. **Install frontend dependencies**:
   ```bash
   cd frontend
   npm install
   ```

2. **Configure Docusaurus**:
   Update `docusaurus.config.js` with the backend API URL

3. **Add the chat widget**:
   Import and use the ChatWidget component in your Docusaurus pages

## API Usage

### General Question Answering
```javascript
fetch('http://localhost:8000/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query: 'What does the book say about RAG systems?'
  })
})
```

### Selected Text Question Answering
```javascript
fetch('http://localhost:8000/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query: 'Explain this concept further',
    selected_text: 'The text the user highlighted...'
  })
})
```

## Testing
- Run unit tests: `pytest tests/unit`
- Run integration tests: `pytest tests/integration`
- Validate API contracts: `pytest tests/contract`