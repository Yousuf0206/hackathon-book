import React, { useState, useEffect, useRef } from 'react';
import './chat-widget.css'; // We'll create this CSS file

const BookChat = () => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedText, setSelectedText] = useState('');
  const [sessionId, setSessionId] = useState(null);
  const messagesEndRef = useRef(null);

  // Function to scroll to bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Function to handle sending messages
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
      // Determine the API base URL based on environment
      const apiBaseUrl = process.env.NODE_ENV === 'production'
        ? '/api'  // In production, relative path
        : 'http://localhost:8000/api'; // For development

      // Call the backend API
      const response = await fetch(`${apiBaseUrl}/chat`, {
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

  // Handle Enter key press
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // Function to handle text selection
  const handleTextSelection = () => {
    const text = window.getSelection ? window.getSelection().toString() : '';
    if (text) {
      setSelectedText(text);
      setInputValue(`Question about selected text: ${text.substring(0, 50)}...`);
    }
  };

  // Format text to handle newlines
  const formatText = (text) => {
    return text.split('\n').map((line, i) => (
      <React.Fragment key={i}>
        {line}
        <br />
      </React.Fragment>
    ));
  };

  return (
    <div className="book-chat-widget">
      <div className="chat-header">
        <h3>Book Assistant</h3>
        {selectedText && (
          <div className="selected-text-preview">
            <small>Selected: {selectedText.substring(0, 60)}...</small>
            <button
              onClick={() => setSelectedText('')}
              className="clear-selection-btn"
            >
              Clear
            </button>
          </div>
        )}
      </div>

      <div className="chat-messages">
        {messages.length === 0 ? (
          <div className="welcome-message">
            <p>Hello! I'm your book assistant. Ask me anything about the book content.</p>
            <p>You can also select text in the book and click "Use Selected Text" to ask questions about that specific part.</p>
          </div>
        ) : (
          messages.map((message) => (
            <div
              key={message.id}
              className={`chat-message ${message.sender}-message`}
            >
              <div className="message-content">
                <div className="message-text">
                  {formatText(message.text)}
                </div>
                {message.citations && message.citations.length > 0 && (
                  <div className="message-citations">
                    <h4>Sources:</h4>
                    <ul>
                      {message.citations.map((citation, index) => (
                        <li key={index}>
                          <div className="citation-item">
                            {citation.chapter && (
                              <div className="citation-chapter">Chapter: {citation.chapter}</div>
                            )}
                            {citation.section && (
                              <div className="citation-section">Section: {citation.section}</div>
                            )}
                            {citation.source_url && (
                              <div className="citation-source">
                                <a
                                  href={citation.source_url}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                >
                                  Source Link
                                </a>
                              </div>
                            )}
                            {citation.text_snippet && (
                              <div className="citation-text">
                                <small>"{citation.text_snippet}"</small>
                              </div>
                            )}
                          </div>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          ))
        )}
        {isLoading && (
          <div className="loading-message">
            <div className="typing-indicator">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="chat-input-area">
        <div className="input-controls">
          <button
            onClick={handleTextSelection}
            className={`text-selection-btn ${selectedText ? 'active' : ''}`}
            title="Use selected text from the book"
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
          />
          <button
            onClick={handleSendMessage}
            disabled={!inputValue.trim() || isLoading}
            className="send-button"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
};

export default BookChat;