import React, { useState } from 'react';

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
      <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-4">
        <div className="text-center">
          <h3 className="text-lg font-medium text-green-800 dark:text-green-200 mb-2">Sign in to access Urdu translation</h3>
          <p className="text-green-600 dark:text-green-300 mb-3">Log in to translate this content to Urdu.</p>
          <div className="flex justify-center space-x-4">
            <a href="/signin" className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors">Sign In</a>
            <a href="/signup" className="px-4 py-2 bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors">Sign Up</a>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="text-center">
        <h3 className="text-lg font-medium text-gray-800 dark:text-gray-200">Translate to Urdu</h3>
        <p className="text-sm text-gray-600 dark:text-gray-400">Convert this content to Urdu for easier understanding</p>
      </div>

      <div className="space-y-4">
        <div className="flex flex-wrap gap-2">
          <button
            onClick={handleTranslateToUrdu}
            disabled={isTranslating}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              isTranslating
                ? 'bg-gray-300 dark:bg-gray-600 text-gray-500 dark:text-gray-400 cursor-not-allowed'
                : 'bg-green-600 hover:bg-green-700 text-white'
            }`}
          >
            {isTranslating ? 'Translating...' : isTranslated ? 'Re-translate' : 'Translate to Urdu'}
          </button>

          {isTranslated && (
            <button
              onClick={handleReset}
              className="px-4 py-2 rounded-lg font-medium bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200 hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
            >
              Reset to Original
            </button>
          )}
        </div>
      </div>

      {isTranslated && (
        <div className="flex items-center space-x-2 text-green-600 dark:text-green-400">
          <div className="w-2 h-2 bg-green-500 rounded-full"></div>
          <span>Content translated to Urdu</span>
        </div>
      )}
    </div>
  );
};

export default UrduTranslationControls;