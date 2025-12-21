/*
 * Placeholder component for text selection functionality.
 * Full implementation will be completed in Phase 5: User Story 3 - Interactive Chat Interface.
 * This component will allow users to select text in the book and ask questions about the selected text.
 */

import React, { useState } from 'react';

const TextSelection = ({ onTextSelected }) => {
  const [selectedText, setSelectedText] = useState('');

  // This is a placeholder implementation
  // The actual implementation will be done in Phase 5
  const handleTextSelection = () => {
    // Get the selected text from the page
    const text = window.getSelection ? window.getSelection().toString() : '';
    if (text) {
      setSelectedText(text);
      if (onTextSelected) {
        onTextSelected(text);
      }
    }
  };

  return (
    <div className="text-selection-component">
      <p>Text selection functionality will be implemented here.</p>
      <p>Currently selected: {selectedText.substring(0, 50)}{selectedText.length > 50 ? '...' : ''}</p>
      <button onClick={handleTextSelection}>Use Selected Text</button>
    </div>
  );
};

export default TextSelection;