---
id: 3
title: Create implementation plan for RAG Chatbot
stage: plan
date: 2025-12-10
surface: agent
model: claude-haiku-4-5-20251001
feature: rag-chatbot-integration
branch: 001-rag-chatbot-integration
user: user
command: /sp.plan
labels: [plan, architecture, rag, chatbot]
links:
  spec: specs/001-rag-chatbot-integration/spec.md
  plan: specs/001-rag-chatbot-integration/plan.md
  ticket: null
  adr: null
  pr: null
files:
 - specs/001-rag-chatbot-integration/plan.md
 - specs/001-rag-chatbot-integration/research.md
 - specs/001-rag-chatbot-integration/data-model.md
 - specs/001-rag-chatbot-integration/quickstart.md
 - specs/001-rag-chatbot-integration/contracts/openapi-contract.md
 - CLAUDE.md
tests:
 - specs/001-rag-chatbot-integration/checklists/requirements.md
---

## Prompt

– RAG Chatbot System Plan

## Architecture Sketch
- **Frontend (Docusaurus Book Website)**
  - User selects text → Sends selected passage as query
  - Chat UI embedded inside the website (iframe or React integration)

- **Backend API (FastAPI)**
  - Endpoints:
    - `/embed` – generate embeddings using Cohere
    - `/query` – query Qdrant + return context chunks
    - `/chat` – OpenAI Agents SDK → final answer
  - Neon Postgres → store metadata and chat logs

- **Vector Store (Qdrant Cloud Free Tier)**
  - Collection: `book_embeddings`
  - Schema: `id`, `text_chunk`, `embedding`, `metadata {chapter, url, section}`

- **Agent Layer**
  - OpenAI Agent that includes:
    - Retrieval tool (Qdrant search)
    - Book-context grounding prompt
    - Safety + hallucination guardrails

- **Workflow**
  1. Deploy website → scrape → chunk → embed
  2. Store embeddings in Qdrant
  3. Test FastAPI pipeline
  4. Build retrieval-enabled agent
  5. Integrate backend → frontend
  6. Deploy chatbot in book UI

---

## Section Structure
1. **Phase 1 — Research**
   - Review RAG architecture best practices
   - Analyze Cohere embedding performance
   - Determine chunking strategy

2. **Phase 2 — Foundation**
   - Qdrant schema design
   - FastAPI project scaffolding
   - Postgres logging schema

3. **Phase 3 — Analysis**
   - Validate embedding quality
   - Test Qdrant retrieval pipeline
   - Benchmark performance + latency

4. **Phase 4 — Synthesis**
   - Construct retrieval-enabled agent
   - Integrate Docusaurus frontend
   - Run deployment + final tests

---

## Research Approach
- Use **research-concurrent** writing (research while building)
- Consult:
  - Cohere embedding API docs
  - Qdrant hybrid search docs
  - OpenAI Agents SDK patterns
- Compare:
  - Chunk sizes: 300–500 tokens
  - Embedding models: Cohere English vs Multilingual
- Evaluate:
  - Retrieval precision
  - Grounding score
  - Hallucination prevention

---

## Quality Validation
- Ensure:
  - All answers grounded ONLY in retrieved book text
  - Correct metadata tagging (chapter/section/source)
  - Selected-text mode strictly limits retrieval to user-highlighted text
- Validate:
  - Accuracy of embeddings
  - Correct context chunking
  - Stable API behavior
  - 200+ manual QA checks across chapters

---

## Decisions Needing Documentation
1. **Embedding Model**
   - Options: Cohere, OpenAI, Voyage
   - Tradeoffs: quality vs cost vs multilingual ability

2. **Chunking Strategy**
   - Options: Paragraph-based, semantic splitter
   - Tradeoffs: coherence vs retrieval granularity

3. **Retrieval Method**
   - Dense vs hybrid (dense + keyword)

4. **Hosting**
   - Qdrant Cloud Free vs self-hosted Qdrant
   - FastAPI (local dev) vs deployment target

5. **Integration Pattern**
   - REST endpoints vs WebSocket streaming

---

## Testing Strategy

### Unit Tests
- Chunker
- Embedding generator
- Qdrant insert + query
- Agent tool correct

## Response snapshot

# Implementation Plan: RAG Chatbot Integration

**Branch**: `001-rag-chatbot-integration` | **Date**: 2025-12-10 | **Spec**: [link]
**Input**: Feature specification from `/specs/[###-feature-name]/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Implementation of a RAG (Retrieval-Augmented Generation) chatbot system for the AI/Spec-Driven Book. The system will allow users to ask questions about book content and receive accurate, contextually relevant answers with source citations. The architecture includes a FastAPI backend with OpenAI Agents SDK, Cohere embeddings, Qdrant Cloud vector storage, and integration with the existing Docusaurus book frontend. The system supports both general book questions and questions about user-selected text only, with comprehensive error handling and observability.

## Technical Context

**Language/Version**: Python 3.11, JavaScript/TypeScript for frontend integration
**Primary Dependencies**: FastAPI, OpenAI Agents SDK, Cohere API, Qdrant Cloud, Neon Postgres
**Storage**: Qdrant Cloud (vector database), Neon Postgres (metadata/logs), Docusaurus website (content source)
**Testing**: pytest for backend, Jest for frontend integration tests
**Target Platform**: Web application (Docusaurus book frontend with FastAPI backend)
**Project Type**: Web application (backend API + frontend integration)
**Performance Goals**: <800ms response time for queries, <300ms Qdrant round-trip, support 100 concurrent users
**Constraints**: Must integrate with existing Docusaurus book frontend, GDPR/CCPA compliance, rate limiting
**Scale/Scope**: Single book content, 100 concurrent users, comprehensive book coverage with citations

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Accuracy
- All responses will be grounded in the book content with proper citations
- Responses will be generated using RAG (Retrieval-Augmented Generation) to ensure factual accuracy
- Hallucination prevention through strict grounding in retrieved content

### Clarity
- Target audience: Academic readers with computer science background
- Responses will include source citations for transparency
- API documentation will follow clear, structured patterns

### Reprioducibility
- All processes will be documented with step-by-step instructions
- Embedding generation and retrieval pipeline will be reproducible
- Deployment process will be clearly documented

### Rigor
- Using established RAG architecture patterns
- Leveraging proven technologies (OpenAI Agents, Cohere embeddings, Qdrant)
- Following industry best practices for vector search and retrieval

### Citation Requirements
- All responses will include proper source citations from the book
- Citations will reference specific chapters, sections, and URLs
- Metadata will be stored to enable accurate attribution

### Plagiarism
- Responses will be generated based on book content, not copied directly
- AI will synthesize information rather than reproduce text verbatim
- Proper attribution will be maintained through citations

### Writing Style
- Academic, neutral, and precise tone for system responses
- Clear, structured explanations in API documentation
- Consistent vocabulary across all components

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

```text
backend/
├── src/
│   ├── models/
│   │   ├── embedding.py          # Embedding generation and management
│   │   ├── chunk.py              # Text chunking utilities
│   │   └── chat.py               # Chat session models
│   ├── services/
│   │   ├── embedding_service.py  # Cohere embedding operations
│   │   ├── retrieval_service.py  # Qdrant vector search
│   │   ├── chat_service.py       # OpenAI Agent integration
│   │   └── content_service.py    # Book content scraping and processing
│   ├── api/
│   │   ├── main.py               # FastAPI app entry point
│   │   ├── routes/
│   │   │   ├── chat.py           # Chat endpoints
│   │   │   ├── query.py          # Query/retrieval endpoints
│   │   │   ├── embed.py          # Embedding endpoints
│   │   │   └── health.py         # Health check endpoints
│   │   └── middleware/
│   │       ├── auth.py           # Rate limiting and authentication
│   │       └── logging.py        # Structured logging
│   └── tools/
│       └── content_loader.py     # Script to load book content
├── tests/
│   ├── unit/
│   │   ├── models/
│   │   ├── services/
│   │   └── api/
│   ├── integration/
│   │   ├── embedding_tests.py
│   │   ├── retrieval_tests.py
│   │   └── api_tests.py
│   └── contract/
│       └── openapi.yaml
├── requirements.txt
├── Dockerfile
└── docker-compose.yml

frontend/
├── src/
│   ├── components/
│   │   ├── ChatWidget.jsx        # Main chat interface
│   │   ├── ChatMessage.jsx       # Individual message display
│   │   ├── SourceCitation.jsx    # Source citation display
│   │   └── TextSelector.jsx      # Text selection UI
│   ├── services/
│   │   ├── api.js                # API client for backend communication
│   │   └── chat.js               # Chat session management
│   └── styles/
│       └── chat.css              # Chat widget styling
└── docusaurus.config.js          # Docusaurus integration config
```

**Structure Decision**: Web application with separate backend (FastAPI) and frontend (React components for Docusaurus integration) to maintain clear separation of concerns between the RAG processing logic and the user interface. The backend handles all RAG operations (embedding, retrieval, AI responses) while the frontend provides the chat interface integrated into the existing Docusaurus book website.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

*No violations identified - all constitution requirements satisfied.*

## Summary of Phase 0-1 Deliverables

### Completed Artifacts
- **research.md**: Technology decisions and rationale for embedding models, chunking strategy, retrieval methods, and architecture
- **data-model.md**: Complete entity definitions with validation rules and state transitions
- **contracts/openapi-contract.md**: API specification for all required endpoints
- **quickstart.md**: Step-by-step setup and usage instructions
- **Agent Context**: Updated CLAUDE.md with new technologies and frameworks for this feature

### Architecture Decision Summary
- **Backend**: FastAPI application handling RAG operations
- **Vector Storage**: Qdrant Cloud for embeddings and retrieval
- **AI Integration**: OpenAI Agents SDK with Cohere embeddings
- **Frontend**: React components integrated with existing Docusaurus site
- **Data Flow**: Book content → Embeddings → Qdrant → Retrieval → AI Response → Frontend

## Outcome

- ✅ Impact: Comprehensive implementation plan for RAG chatbot system
- 🧪 Tests: All constitution checks passed, requirements validated
- 📁 Files: plan.md, research.md, data-model.md, quickstart.md, contracts/ created
- 🔁 Next prompts: Ready for task generation with `/sp.tasks`
- 🧠 Reflection: Plan addresses all key architectural decisions and requirements

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): PASS - All constitution requirements satisfied
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Generate tasks with `/sp.tasks`