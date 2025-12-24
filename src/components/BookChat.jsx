import React, { useState, useEffect, useRef } from 'react';

const BookChat = () => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedText, setSelectedText] = useState('');
  const [sessionId, setSessionId] = useState(null);
  const [targetLanguage, setTargetLanguage] = useState('en'); // Track active language
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

    // Add user message to chat using the new schema
    const userMessage = {
      id: Date.now(),
      content: inputValue,
      sender: 'user',
      timestamp: new Date(),
      language: targetLanguage
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      // Get user ID from localStorage if available
      const userStr = localStorage.getItem('user');
      let userId = null;
      if (userStr) {
        try {
          const user = JSON.parse(userStr);
          userId = user.id || null;
        } catch (e) {
          console.warn('Could not parse user data from localStorage');
        }
      }

      // Determine the API base URL based on environment variable or fallback
      // In production, this should be set to your actual backend API URL via REACT_APP_API_BASE_URL environment variable
      // Example for Vercel: Set REACT_APP_API_BASE_URL in your Vercel project settings
      const apiBaseUrl = process.env.REACT_APP_API_BASE_URL ||
                        (process.env.NODE_ENV === 'production'
                          ? `${window.location.protocol}//${window.location.hostname}:8000/api` // Production - fallback to same host
                          : 'http://localhost:8000/api'); // Development

      // Call the backend API
      const response = await fetch(`${apiBaseUrl}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: inputValue,
          selected_text: selectedText,
          session_id: sessionId,
          user_id: userId,  // Pass user ID for personalization
          target_language: targetLanguage  // Pass target language for multilingual support
        })
      });

      // Check if response is ok before parsing JSON
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // Check if response body is empty
      const responseText = await response.text();
      if (!responseText) {
        throw new Error('Empty response from server');
      }

      // Try to parse JSON
      let data;
      try {
        data = JSON.parse(responseText);
      } catch (e) {
        console.error('Failed to parse JSON response:', responseText);
        throw new Error('Invalid JSON response from server');
      }

      // Add AI response to chat using the new schema
      const aiMessage = {
        id: Date.now() + 1,
        content: data.response,
        sender: 'ai',
        citations: data.citations,
        timestamp: new Date(),
        language: targetLanguage,
        confidence: data.confidence || 'high' // Default to high if not provided
      };

      setMessages(prev => [...prev, aiMessage]);
      setSessionId(data.session_id);
    } catch (error) {
      const errorMessage = {
        id: Date.now() + 1,
        content: `Error: ${error.message}`,
        sender: 'system',
        timestamp: new Date(),
        language: targetLanguage
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
    <div className={`book-chat-widget ${targetLanguage === 'ur' ? 'rtl-language' : ''}`} role="main" aria-label="Book Assistant Chat Interface">
      {/* Header with language selector */}
      <div className="chat-header">
        <div className="chat-header-content">
          <h3 className="chat-title">📚 Book Assistant</h3>
          <div className="language-selector-container">
            <div className="language-selector-wrapper">
              <label htmlFor="language-select" className="language-label">Language:</label>
              <select
                id="language-select"
                value={targetLanguage}
                onChange={(e) => setTargetLanguage(e.target.value)}
                className="language-select"
              >
                <option value="en">English</option>
                <option value="ur">Urdu</option>
              </select>
            </div>
          </div>
        </div>

        {selectedText && (
          <div className="selected-text-display">
            <span className="selected-text-preview">Selected: {selectedText.substring(0, 60)}...</span>
            <button
              onClick={() => setSelectedText('')}
              className="clear-selection-btn"
              aria-label="Clear selected text"
            >
              Clear
            </button>
          </div>
        )}
      </div>

      <div className="chat-messages" role="log" aria-live="polite" aria-label="Chat messages">
        {messages.length === 0 ? (
          <div className="welcome-message" role="status" aria-live="polite">
            <div className="welcome-icon">🤖</div>
            <h3 className="welcome-title">Welcome to Book Assistant!</h3>
            <p className="welcome-text">Ask me anything about the book content.</p>
            <p className="welcome-subtext">Select text in the book and click "Use Selected Text" to ask questions about specific parts.</p>
          </div>
        ) : (
          messages.map((message) => (
            <div
              key={message.id}
              className={`message-container ${message.sender === 'user' ? 'user-message' : message.sender === 'system' ? 'system-message' : 'ai-message'} ${targetLanguage === 'ur' ? 'rtl-message' : ''}`}
            >
              <div className={`message-bubble ${message.sender === 'user' ? 'user-bubble' : message.sender === 'system' ? 'system-bubble' : 'ai-bubble'}`}>
                <div className="message-text">
                  {formatText(message.content || message.text)}
                </div>
                {message.citations && message.citations.length > 0 && (
                  <div className="message-citations">
                    <h4 className="citations-title">Sources:</h4>
                    <ul className="citations-list">
                      {message.citations.map((citation, index) => (
                        <li key={index} className="citation-item">
                          <div className="citation-content">
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
                                  className="source-link"
                                >
                                  Source Link
                                </a>
                              </div>
                            )}
                            {citation.text_snippet && (
                              <div className="citation-text">
                                "{citation.text_snippet}"
                              </div>
                            )}
                          </div>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                {message.confidence && (
                  <div className="message-confidence" title={`Confidence: ${message.confidence}`}>
                    <span className={`confidence-indicator confidence-${message.confidence}`}></span>
                  </div>
                )}
              </div>
            </div>
          ))
        )}
        {isLoading && (
          <div className="loading-container" role="status" aria-live="polite">
            <div className="loading-bubble">
              <div className="loading-content">
                <span className="loading-text">AI is thinking</span>
                <div className="loading-dots">
                  <div className="loading-dot"></div>
                  <div className="loading-dot" style={{ animationDelay: '0.2s' }}></div>
                  <div className="loading-dot" style={{ animationDelay: '0.4s' }}></div>
                </div>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="chat-input-area" role="form" aria-label="Chat input area">
        <div className="input-container">
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

          <div className="input-elements">
            <textarea
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question about the book..."
              disabled={isLoading}
              rows="1"
              className="chat-textarea"
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
    </div>
  );
};

export default BookChat;