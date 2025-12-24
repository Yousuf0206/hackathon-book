import React, { useState } from 'react';
import Layout from '@theme-original/Layout';
import BookChat from '@site/src/components/BookChat';

export default function LayoutWrapper(props) {
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [isChatMinimized, setIsChatMinimized] = useState(false);

  const toggleMinimize = () => {
    setIsChatMinimized(!isChatMinimized);
  };

  return (
    <>
      <Layout {...props}>
        {props.children}
        {!isChatOpen && (
          <button
            className="chat-toggle-button"
            onClick={() => setIsChatOpen(true)}
            aria-label="Open book assistant chat"
          >
            <div className="chat-icon">🤖</div>
            <div className="chat-text">Ask me anything</div>
          </button>
        )}
        {isChatOpen && (
          <div className={`chat-modal-container ${isChatMinimized ? 'minimized' : ''}`}>
            <div className="chat-modal-header">
              <h3>Book Assistant</h3>
              <div className="chat-controls">
                <button
                  className="minimize-chat-button"
                  onClick={toggleMinimize}
                  aria-label={isChatMinimized ? "Maximize chat" : "Minimize chat"}
                >
                  {isChatMinimized ? '☐' : '−'}
                </button>
                <button
                  className="close-chat-button"
                  onClick={() => {
                    setIsChatOpen(false);
                    setIsChatMinimized(false);
                  }}
                  aria-label="Close chat"
                >
                  ×
                </button>
              </div>
            </div>
            {!isChatMinimized && <BookChat />}
          </div>
        )}
      </Layout>
    </>
  );
}