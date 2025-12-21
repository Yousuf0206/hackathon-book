# RAG Chatbot Frontend

This is the frontend for the RAG (Retrieval-Augmented Generation) Chatbot Integration for the Physical AI & Humanoid Robotics book.

## Features

- Interactive chat interface for asking questions about book content
- Text selection functionality to ask questions about specific parts of the text
- Source citations for all AI-generated responses
- Responsive design for different screen sizes
- Accessibility features for improved usability

## Components

- `ChatWidget`: Main chat interface component
- `ChatMessage`: Individual message display component
- `SourceCitation`: Citation display component
- `api.js`: API service for backend communication
- `chat.js`: Chat session management service

## Getting Started

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm start
```

3. Build for production:
```bash
npm run build
```

## Environment Variables

- `REACT_APP_API_BASE_URL`: Base URL for the backend API (optional, defaults to `/api`)

## API Integration

The frontend communicates with the backend through the following endpoints:

- `POST /api/chat`: Process user queries and return AI-generated responses with citations
- `POST /api/query`: Retrieve relevant book content chunks based on user query
- `GET /api/health`: Health check for backend services

## Text Selection Feature

The chat interface supports text selection from the book content:

1. Select text in the book content area
2. Click the "Use Selected Text" button in the chat interface
3. The selected text will be used as context for your question

## Accessibility Features

- Proper ARIA labels and roles
- Keyboard navigation support
- Screen reader compatibility
- Sufficient color contrast
- Responsive design for different screen sizes

## Testing

Run the test suite:
```bash
npm test
```

## Integration with Docusaurus

The chat widget is designed to be integrated with the Docusaurus book site. The `BookChat` component in the Docusaurus src directory can be used directly in Docusaurus pages.