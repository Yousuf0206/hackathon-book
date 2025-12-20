import React, { useState, useEffect, useRef } from 'react';
import ChatMessage from './ChatMessage';
import SourceCitation from './SourceCitation';
import apiService from '../services/api';

const ChatWidget = () => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedText, setSelectedText] = useState('');
  const [sessionId, setSessionId] = useState(null);
  const [targetLanguage, setTargetLanguage] = useState('en'); // Track active language
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
      // Use the API service to call the backend
      const data = await apiService.chat(inputValue, selectedText, sessionId, targetLanguage);

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
    <div className="flex flex-col h-[500px] border border-gray-300 dark:border-gray-700 rounded-xl overflow-hidden bg-white dark:bg-gray-800 shadow-xl transition-colors duration-300" role="main" aria-label="Book Assistant Chat Interface">
      {/* Header with language selector */}
      <div className="bg-gradient-to-r from-blue-600 to-indigo-700 p-4 text-white">
        <div className="flex justify-between items-center">
          <h3 className="text-lg font-semibold">📚 Book Assistant</h3>
          <div className="flex items-center space-x-2">
            <div className="flex items-center space-x-2">
              <label htmlFor="language-select" className="text-sm">Language:</label>
              <select
                id="language-select"
                value={targetLanguage}
                onChange={(e) => setTargetLanguage(e.target.value)}
                className="bg-white/20 border border-white/30 rounded px-2 py-1 text-sm focus:outline-none focus:ring-1 focus:ring-white"
              >
                <option value="en">English</option>
                <option value="ur">Urdu</option>
              </select>
            </div>
          </div>
        </div>

        {selectedText && (
          <div className="mt-2 p-2 bg-white/20 rounded-lg text-sm flex justify-between items-center">
            <span className="truncate max-w-[70%]">Selected: {selectedText.substring(0, 60)}...</span>
            <button
              onClick={() => setSelectedText('')}
              className="ml-2 px-2 py-1 bg-white/30 hover:bg-white/40 rounded text-xs transition-colors"
              aria-label="Clear selected text"
            >
              Clear
            </button>
          </div>
        )}
      </div>

      {/* Messages container */}
      <div
        className="flex-1 overflow-y-auto p-4 bg-gray-50 dark:bg-gray-900/50 flex flex-col gap-4 max-h-[320px]"
        role="log"
        aria-live="polite"
        aria-label="Chat messages"
      >
        {messages.length === 0 ? (
          <div className="flex-1 flex flex-col items-center justify-center text-center p-8 text-gray-500 dark:text-gray-400" role="status" aria-live="polite">
            <div className="mb-4 text-4xl">🤖</div>
            <h3 className="text-xl font-medium mb-2 text-gray-800 dark:text-gray-200">Welcome to Book Assistant!</h3>
            <p className="mb-1">Ask me anything about the book content.</p>
            <p className="text-sm">Select text in the book and click "Use Selected Text" to ask questions about specific parts.</p>
          </div>
        ) : (
          messages.map((message) => (
            <ChatMessage
              key={message.id}
              message={message}
              targetLanguage={targetLanguage} // Pass language to message component if needed
            />
          ))
        )}
        {isLoading && (
          <div className="flex justify-start" role="status" aria-live="polite">
            <div className="bg-white dark:bg-gray-700 border border-gray-200 dark:border-gray-600 text-gray-800 dark:text-gray-200 rounded-2xl px-4 py-3 rounded-bl-sm">
              <div className="flex items-center">
                <span className="mr-2">AI is thinking</span>
                <div className="flex space-x-1">
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></div>
                </div>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} aria-hidden="true" />
      </div>

      {/* Input area */}
      <div className="p-4 bg-gray-50 dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700" role="form" aria-label="Chat input area">
        <div className="flex flex-col space-y-2">
          <div className="flex space-x-2">
            <button
              onClick={handleTextSelection}
              className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                selectedText
                  ? 'bg-green-600 text-white hover:bg-green-700'
                  : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600'
              }`}
              title="Use selected text from the book"
              aria-label="Use selected text from the book"
            >
              Use Selected Text
            </button>
          </div>

          <div className="flex space-x-2">
            <textarea
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question about the book..."
              disabled={isLoading}
              rows="1"
              className="flex-1 p-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none min-h-[44px] max-h-[120px]"
              aria-label="Type your question here"
              role="textbox"
              aria-multiline="true"
            />
            <button
              onClick={handleSendMessage}
              disabled={!inputValue.trim() || isLoading}
              className={`px-4 py-3 rounded-lg font-medium transition-colors ${
                !inputValue.trim() || isLoading
                  ? 'bg-gray-300 dark:bg-gray-600 text-gray-500 dark:text-gray-400 cursor-not-allowed'
                  : 'bg-blue-600 hover:bg-blue-700 text-white'
              }`}
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

export default ChatWidget;