import React from 'react';
import SourceCitation from './SourceCitation';

const ChatMessage = ({ message, targetLanguage = 'en' }) => {
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

  const isUrdu = targetLanguage === 'ur';

  return (
    <div
      className={`flex ${sender === 'user' ? 'justify-end' : sender === 'system' ? 'justify-center' : 'justify-start'}`}
      role="listitem"
      aria-label={`${sender} message`}
    >
      <div className={`max-w-[85%] rounded-2xl px-4 py-3 ${sender === 'user'
        ? 'bg-blue-600 text-white rounded-br-sm'
        : sender === 'system'
        ? 'bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-300 text-sm italic'
        : 'bg-white dark:bg-gray-700 text-gray-800 dark:text-gray-200 border border-gray-200 dark:border-gray-600 rounded-bl-sm'}`}
      >
        <div className={`message-text ${isUrdu ? 'text-right' : ''}`}>
          {formatText(text)}
        </div>
        {citations && citations.length > 0 && (
          <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-600" role="region" aria-label="Citations for this response">
            <h4 className="text-xs font-semibold text-gray-600 dark:text-gray-300 mb-2">Sources:</h4>
            <ul role="list" aria-label="List of sources" className="space-y-2">
              {citations.map((citation, index) => (
                <li key={index} role="listitem" className="text-xs text-gray-600 dark:text-gray-400">
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