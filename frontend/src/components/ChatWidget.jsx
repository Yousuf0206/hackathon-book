import React, { useState, useEffect, useRef } from 'react';
import ChatMessage from './ChatMessage';
import SourceCitation from './SourceCitation';
import './chat.css';

const ChatWidget = () => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedText, setSelectedText] = useState('');
  const [sessionId, setSessionId] = useState(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    // Add user message to chat
    const userMessage = {
      id: Date.now(),
      text: inputValue,
      sender: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      // Call the backend API
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: inputValue,
          selected_text: selectedText,
          session_id: sessionId
        })
      });

      const data = await response.json();

      if (response.ok) {
        // Add AI response to chat
        const aiMessage = {
          id: Date.now() + 1,
          text: data.response,
          sender: 'ai',
          citations: data.citations,
          timestamp: new Date()
        };

        setMessages(prev => [...prev, aiMessage]);
        setSessionId(data.session_id);
      } else {
        // Add error message
        const errorMessage = {
          id: Date.now() + 1,
          text: `Error: ${data.detail || 'An error occurred'}`,
          sender: 'system',
          timestamp: new Date()
        };
        setMessages(prev => [...prev, errorMessage]);
      }
    } catch (error) {
      const errorMessage = {
        id: Date.now() + 1,
        text: `Error: ${error.message}`,
        sender: 'system',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleTextSelection = () => {
    const text = window.getSelection ? window.getSelection().toString() : '';
    if (text) {
      setSelectedText(text);
      setInputValue(`Question about selected text: ${text.substring(0, 50)}...`);
    }
  };

  return (
    <div className="chat-widget" role="main" aria-label="Book Assistant Chat Interface">
      <div className="chat-header" role="banner">
        <h3>Book Assistant</h3>
        {selectedText && (
          <div className="selected-text-preview" aria-label="Selected text preview">
            <small>Selected: {selectedText.substring(0, 60)}...</small>
            <button
              onClick={() => setSelectedText('')}
              className="clear-selection"
              aria-label="Clear selected text"
            >
              Clear
            </button>
          </div>
        )}
      </div>

      <div
        className="chat-messages"
        role="log"
        aria-live="polite"
        aria-label="Chat messages"
      >
        {messages.length === 0 ? (
          <div className="welcome-message" role="status" aria-live="polite">
            <p>Hello! I'm your book assistant. Ask me anything about the book content.</p>
            <p>You can also select text in the book and click "Use Selected Text" to ask questions about that specific part.</p>
          </div>
        ) : (
          messages.map((message) => (
            <ChatMessage key={message.id} message={message} />
          ))
        )}
        {isLoading && (
          <div className="loading-message" role="status" aria-live="polite">
            <div className="typing-indicator" aria-label="Assistant is typing">
              <span aria-hidden="true"></span>
              <span aria-hidden="true"></span>
              <span aria-hidden="true"></span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} aria-hidden="true" />
      </div>

      <div className="chat-input-area" role="form" aria-label="Chat input area">
        <div className="input-controls">
          <button
            onClick={handleTextSelection}
            className={`text-selection-btn ${selectedText ? 'active' : ''}`}
            title="Use selected text from the book"
            aria-label="Use selected text from the book"
          >
            Use Selected Text
          </button>
        </div>

        <div className="input-container">
          <textarea
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about the book..."
            disabled={isLoading}
            rows="1"
            aria-label="Type your question here"
            role="textbox"
            aria-multiline="true"
          />
          <button
            onClick={handleSendMessage}
            disabled={!inputValue.trim() || isLoading}
            className="send-button"
            aria-label="Send message"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatWidget;