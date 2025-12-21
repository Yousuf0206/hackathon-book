import React, { useState } from 'react';
import Layout from '@theme-original/Layout';
import BookChat from '@site/src/components/BookChat';

export default function LayoutWrapper(props) {
  const [isChatOpen, setIsChatOpen] = useState(false);

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
            <div className="chat-text">Ask me anything regarding the book</div>
          </button>
        )}
        {isChatOpen && (
          <div className="chat-modal-container">
            <div className="chat-modal-header">
              <h3>📚 Book Assistant</h3>
              <button
                className="close-chat-button"
                onClick={() => setIsChatOpen(false)}
                aria-label="Close chat"
              >
                ×
              </button>
            </div>
            <BookChat />
          </div>
        )}
      </Layout>
    </>
  );
}