import React, { useState } from 'react';
import './translation.css';

const UrduTranslationControls = ({ content, onContentChange }) => {
  const [isTranslated, setIsTranslated] = useState(false);
  const [isTranslating, setIsTranslating] = useState(false);
  const [userProfile, setUserProfile] = useState(null);

  // Check if user is authenticated
  React.useEffect(() => {
    const token = localStorage.getItem('access_token');
    const user = localStorage.getItem('user');

    if (token && user) {
      try {
        const userData = JSON.parse(user);
        setUserProfile(userData);
      } catch (error) {
        console.error('Error parsing user data:', error);
      }
    }
  }, []);

  const handleTranslateToUrdu = async () => {
    if (!userProfile) {
      alert('Please sign in to use translation features');
      return;
    }

    setIsTranslating(true);

    try {
      // In a real implementation, this would call an API to translate the content to Urdu
      const urduContent = await translateToUrdu(content);
      onContentChange(urduContent);
      setIsTranslated(true);
    } catch (error) {
      console.error('Error translating content:', error);
      alert('Error translating content. Please try again.');
    } finally {
      setIsTranslating(false);
    }
  };

  const handleReset = () => {
    // Reset to original content by calling parent's handler with original content
    // This will be handled by the parent component
    onContentChange(content); // Reset to the content passed to this component
    setIsTranslated(false);
  };

  // Mock function to simulate Urdu translation
  // In a real implementation, this would call a translation API
  const translateToUrdu = async (originalContent) => {
    // This is a mock implementation - in reality, this would call an API
    // that translates the content to Urdu using a service like Google Translate API
    // or a dedicated Urdu translation model

    // For demonstration purposes, we'll just return a placeholder
    // In a real implementation, you would send the content to your backend
    // which would use a proper translation service
    try {
      const response = await fetch('/api/translate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('access_token')}`
        },
        body: JSON.stringify({
          text: originalContent,
          target_language: 'ur'
        })
      });

      if (response.ok) {
        const data = await response.json();
        return data.translated_text;
      } else {
        // Fallback: return original content with a note
        return `[Urdu translation would appear here]\n\n${originalContent}`;
      }
    } catch (error) {
      console.error('Translation API error:', error);
      // Fallback: return original content with a note
      return `[Urdu translation would appear here]\n\n${originalContent}`;
    }
  };

  if (!userProfile) {
    return (
      <div className="translation-prompt">
        <div className="translation-message">
          <h3>Sign in to access Urdu translation</h3>
          <p>Log in to translate this content to Urdu.</p>
          <a href="/signin" className="auth-link">Sign In</a> or <a href="/signup" className="auth-link">Sign Up</a>
        </div>
      </div>
    );
  }

  return (
    <div className="translation-controls">
      <div className="controls-header">
        <h3>Translate to Urdu</h3>
        <p>Convert this content to Urdu for easier understanding</p>
      </div>

      <div className="controls-form">
        <div className="button-group">
          <button
            onClick={handleTranslateToUrdu}
            disabled={isTranslating}
            className="translate-btn"
          >
            {isTranslating ? 'Translating...' : isTranslated ? 'Re-translate' : 'Translate to Urdu'}
          </button>

          {isTranslated && (
            <button
              onClick={handleReset}
              className="reset-btn"
            >
              Reset to Original
            </button>
          )}
        </div>
      </div>

      {isTranslated && (
        <div className="translation-status">
          <span className="status-indicator active"></span>
          <span>Content translated to Urdu</span>
        </div>
      )}
    </div>
  );
};

export default UrduTranslationControls;