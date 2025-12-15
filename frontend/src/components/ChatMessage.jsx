import React from 'react';
import SourceCitation from './SourceCitation';

const ChatMessage = ({ message }) => {
  const { sender, text, citations } = message;

  const formatText = (text) => {
    // Simple formatting to handle newlines and basic markdown
    return text.split('\n').map((line, i) => (
      <React.Fragment key={i}>
        {line}
        <br />
      </React.Fragment>
    ));
  };

  return (
    <div
      className={`chat-message ${sender}-message`}
      role="listitem"
      aria-label={`${sender} message`}
    >
      <div className="message-content">
        <div className="message-text">
          {formatText(text)}
        </div>
        {citations && citations.length > 0 && (
          <div className="message-citations" role="region" aria-label="Citations for this response">
            <h4>Sources:</h4>
            <ul role="list" aria-label="List of sources">
              {citations.map((citation, index) => (
                <li key={index} role="listitem">
                  <SourceCitation citation={citation} />
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatMessage;