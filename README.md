# RAG Chatbot Integration for Physical AI & Humanoid Robotics Book

This project implements a Retrieval-Augmented Generation (RAG) chatbot system for the Physical AI & Humanoid Robotics book, enabling users to ask questions about book content and receive accurate, contextually relevant answers with source citations.

## Features

- **Question Answering**: Ask questions about the book content and receive AI-generated responses
- **Selected Text Queries**: Ask questions specifically about selected portions of text
- **Source Citations**: All responses include citations to the original book content
- **Interactive Chat Interface**: User-friendly chat interface integrated with Docusaurus
- **Performance Monitoring**: Built-in metrics and monitoring for response times
- **Security**: Security headers, input validation, and rate limiting
- **Resilience**: Graceful degradation and fallback responses for service failures

## Architecture

The system consists of:

- **Frontend**: React-based chat interface with text selection capabilities
- **Backend**: FastAPI-based API with multiple service layers
- **Embedding Service**: Cohere integration for text embeddings
- **Retrieval Service**: Qdrant vector database for semantic search
- **Chat Service**: OpenAI integration for question answering
- **Content Service**: Web scraping and content processing

## Components

### Backend Services

- `EmbeddingService`: Generates vector embeddings using Cohere API
- `RetrievalService`: Handles vector search and storage in Qdrant
- `ChatService`: Processes queries with OpenAI and generates responses
- `ContentService`: Scrapes and processes book content

### API Endpoints

- `POST /api/chat`: Process user queries and return AI-generated responses
- `POST /api/query`: Retrieve relevant book content chunks
- `POST /api/embed`: Ingest book content into the vector database
- `GET /api/health`: Health check for all dependencies

### Frontend Components

- `ChatWidget`: Main chat interface
- `ChatMessage`: Individual message display
- `SourceCitation`: Citation display component
- `TextSelection`: Text selection functionality

## Setup and Installation

### Prerequisites

- Python 3.8+
- Node.js 16+
- Docker (optional, for containerized deployment)
- Cohere API key
- OpenAI API key
- Qdrant Cloud account (or local instance)

### Environment Configuration

Create a `.env` file with the following variables:

```bash
COHERE_API_KEY=your_cohere_api_key
OPENAI_API_KEY=your_openai_api_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key  # if using cloud version
```

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   uvicorn src.api.main:app --reload --port 8000
   ```

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Run the development server:
   ```bash
   npm start
   ```

## Usage

### Content Ingestion

To load book content into the system:

1. Use the `/api/embed` endpoint to process and store book content
2. The system will automatically chunk the content and generate embeddings
3. Content is stored in Qdrant with idempotent behavior (prevents duplicates)

### Asking Questions

1. Use the chat interface to ask questions about the book
2. For selected text queries, highlight text in the book and use the "Use Selected Text" button
3. Responses will include source citations to the original content

### API Usage

The API supports both direct API calls and the web interface:

```bash
# General question
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is ROS 2?", "selected_text": null}'

# Selected text question
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain this concept", "selected_text": "The selected text content..."}'
```

## Security Features

- Security headers (HSTS, CSP, X-Frame-Options, etc.)
- Input validation and sanitization
- Rate limiting to prevent abuse
- API key validation for all external services
- Secure handling of user data

## Performance and Monitoring

- Response time monitoring (target: <800ms for chat, <300ms for Qdrant)
- Structured logging with key metrics
- Health checks for all external dependencies
- Graceful degradation when services fail

## Deployment

### Docker

The project includes Docker configuration for easy deployment:

```bash
docker-compose up -d
```

### Production Considerations

- Use environment-specific configuration
- Set up proper monitoring and alerting
- Implement proper SSL/TLS configuration
- Set up backup and recovery procedures

## Testing

The system includes comprehensive tests:

- Unit tests for backend services
- Integration tests for API endpoints
- Frontend component tests
- End-to-end tests covering all user stories

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please open an issue in the GitHub repository or contact the development team.