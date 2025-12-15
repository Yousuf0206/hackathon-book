import React, { useState, useEffect } from 'react';
import './personalization.css';

const PersonalizationControls = ({ content, onContentChange }) => {
  const [isPersonalized, setIsPersonalized] = useState(false);
  const [personalizationLevel, setPersonalizationLevel] = useState('default');
  const [userProfile, setUserProfile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  // Check if user is authenticated and get their profile
  useEffect(() => {
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

  const handlePersonalize = async () => {
    if (!userProfile) {
      alert('Please sign in to use personalization features');
      return;
    }

    setIsLoading(true);

    try {
      // In a real implementation, this would call an API to personalize the content
      // based on the user's profile and selected personalization level
      const personalizedContent = await personalizeContent(content, personalizationLevel);
      onContentChange(personalizedContent);
      setIsPersonalized(true);
    } catch (error) {
      console.error('Error personalizing content:', error);
      alert('Error personalizing content. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    onContentChange(content); // Reset to original content
    setIsPersonalized(false);
    setPersonalizationLevel('default');
  };

  // Mock function to simulate content personalization
  // In a real implementation, this would call the backend API
  const personalizeContent = async (originalContent, level) => {
    // This is a mock implementation - in reality, this would call an API
    // that adjusts the content based on user profile and personalization level
    if (level === 'simplified') {
      // Simulate simplified content (this is just a mock)
      return originalContent.replace(/complex/gi, 'simple')
                           .replace(/advanced/gi, 'basic')
                           .replace(/sophisticated/gi, 'basic');
    } else if (level === 'advanced') {
      // Simulate advanced content (this is just a mock)
      return originalContent.replace(/basic/gi, 'advanced')
                           .replace(/simple/gi, 'complex')
                           .replace(/introductory/gi, 'advanced');
    }
    return originalContent; // default level returns original content
  };

  if (!userProfile) {
    return (
      <div className="personalization-prompt">
        <div className="personalization-message">
          <h3>Sign in to personalize your learning experience</h3>
          <p>Log in to customize content based on your background and preferences.</p>
          <a href="/signin" className="auth-link">Sign In</a> or <a href="/signup" className="auth-link">Sign Up</a>
        </div>
      </div>
    );
  }

  return (
    <div className="personalization-controls">
      <div className="controls-header">
        <h3>Personalize Content</h3>
        <p>Adjust the content to match your experience level and preferences</p>
      </div>

      <div className="controls-form">
        <div className="form-group">
          <label htmlFor="personalization-level">Experience Level:</label>
          <select
            id="personalization-level"
            value={personalizationLevel}
            onChange={(e) => setPersonalizationLevel(e.target.value)}
            disabled={isLoading}
          >
            <option value="default">Default (as written)</option>
            <option value="simplified">Simplified (for beginners)</option>
            <option value="advanced">Advanced (for experienced users)</option>
          </select>
        </div>

        <div className="button-group">
          <button
            onClick={handlePersonalize}
            disabled={isLoading}
            className="personalize-btn"
          >
            {isLoading ? 'Personalizing...' : isPersonalized ? 'Update Personalization' : 'Personalize Content'}
          </button>

          {isPersonalized && (
            <button
              onClick={handleReset}
              className="reset-btn"
            >
              Reset to Original
            </button>
          )}
        </div>
      </div>

      {isPersonalized && (
        <div className="personalization-status">
          <span className="status-indicator active"></span>
          <span>Content personalized based on your profile</span>
        </div>
      )}
    </div>
  );
};

export default PersonalizationControls;