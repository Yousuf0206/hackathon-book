import React, { useState } from 'react';
import './component-styles.css';

const AIAssistantPreview = () => {
  const [messages, setMessages] = useState([
    { type: 'bot', text: 'Hello! I\'m your Physical AI & Robotics assistant. How can I help you understand humanoid robotics today?' },
    { type: 'user', text: 'What are the key challenges in humanoid locomotion?' },
    { type: 'bot', text: 'The main challenges in humanoid locomotion include: 1) Balance control on two legs, 2) Dynamic walking on uneven terrain, 3) Real-time adaptation to disturbances, and 4) Energy-efficient movement patterns.' }
  ]);

  const [inputValue, setInputValue] = useState('');

  const handleSendMessage = (e) => {
    e.preventDefault();
    if (inputValue.trim()) {
      setMessages([...messages, { type: 'user', text: inputValue }]);
      setInputValue('');

      // Simulate bot response after a delay
      setTimeout(() => {
        setMessages(prev => [...prev, {
          type: 'bot',
          text: 'That\'s an excellent question! In our curriculum, we cover this topic in depth in the "Humanoid Locomotion" module. Would you like to know more about the specific techniques we teach?'
        }]);
      }, 1000);
    }
  };

  return (
    <section className="ai-assistant-preview-section">
      <div className="ai-assistant-preview-container">
        <div className="ai-assistant-preview-header">
          <h2 className="ai-assistant-preview-title">
            Interactive <span style={{background: 'linear-gradient(135deg, var(--futuristic-cyan), var(--neon-purple))', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent'}}>AI</span> Assistant
          </h2>
          <p className="ai-assistant-preview-subtitle">
            Get instant answers to your questions about physical AI and humanoid robotics
          </p>
        </div>

        <div className="ai-assistant-chat-container">
          <div className="ai-assistant-chat-header">
            <div className="ai-assistant-header-content">
              <div className="ai-assistant-status-indicator ai-assistant-status-green"></div>
              <span className="ai-assistant-status-text">AI Assistant - Ready to help</span>
            </div>
          </div>

          <div className="ai-assistant-messages-container">
            <div className="ai-assistant-messages-list">
              {messages.map((message, index) => (
                <div
                  key={index}
                  className={`ai-assistant-message-container ${message.type === 'user' ? 'ai-assistant-message-user' : 'ai-assistant-message-bot'}`}
                >
                  <div className="ai-assistant-message-bubble">
                    {message.text}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="ai-assistant-input-container">
            <form onSubmit={handleSendMessage} className="ai-assistant-form">
              <input
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                placeholder="Ask about physical AI, robotics, or humanoid systems..."
                className="ai-assistant-input"
              />
              <button
                type="submit"
                className="ai-assistant-send-button"
              >
                Send
              </button>
            </form>
          </div>
        </div>

        <div className="ai-assistant-features-grid">
          <div className="ai-assistant-feature-card">
            <div className="ai-assistant-feature-icon">🧠</div>
            <h3 className="ai-assistant-feature-title">Concept Explanations</h3>
            <p className="ai-assistant-feature-description">Get detailed explanations of complex robotics concepts</p>
          </div>
          <div className="ai-assistant-feature-card">
            <div className="ai-assistant-feature-icon">📚</div>
            <h3 className="ai-assistant-feature-title">Curriculum Guidance</h3>
            <p className="ai-assistant-feature-description">Navigate through learning paths and modules</p>
          </div>
          <div className="ai-assistant-feature-card">
            <div className="ai-assistant-feature-icon">🔧</div>
            <h3 className="ai-assistant-feature-title">Technical Support</h3>
            <p className="ai-assistant-feature-description">Get help with implementation and troubleshooting</p>
          </div>
        </div>
      </div>
    </section>
  );
};

export default AIAssistantPreview;