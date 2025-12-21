import React, { useState, useEffect } from 'react';

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
      <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
        <div className="text-center">
          <h3 className="text-lg font-medium text-blue-800 dark:text-blue-200 mb-2">Sign in to personalize your learning experience</h3>
          <p className="text-blue-600 dark:text-blue-300 mb-3">Log in to customize content based on your background and preferences.</p>
          <div className="flex justify-center space-x-4">
            <a href="/signin" className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">Sign In</a>
            <a href="/signup" className="px-4 py-2 bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors">Sign Up</a>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="text-center">
        <h3 className="text-lg font-medium text-gray-800 dark:text-gray-200">Personalize Content</h3>
        <p className="text-sm text-gray-600 dark:text-gray-400">Adjust the content to match your experience level and preferences</p>
      </div>

      <div className="space-y-4">
        <div className="space-y-2">
          <label htmlFor="personalization-level" className="block text-sm font-medium text-gray-700 dark:text-gray-300">
            Experience Level:
          </label>
          <select
            id="personalization-level"
            value={personalizationLevel}
            onChange={(e) => setPersonalizationLevel(e.target.value)}
            disabled={isLoading}
            className="w-full p-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value="default">Default (as written)</option>
            <option value="simplified">Simplified (for beginners)</option>
            <option value="advanced">Advanced (for experienced users)</option>
          </select>
        </div>

        <div className="flex flex-wrap gap-2">
          <button
            onClick={handlePersonalize}
            disabled={isLoading}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              isLoading
                ? 'bg-gray-300 dark:bg-gray-600 text-gray-500 dark:text-gray-400 cursor-not-allowed'
                : 'bg-blue-600 hover:bg-blue-700 text-white'
            }`}
          >
            {isLoading ? 'Personalizing...' : isPersonalized ? 'Update Personalization' : 'Personalize Content'}
          </button>

          {isPersonalized && (
            <button
              onClick={handleReset}
              className="px-4 py-2 rounded-lg font-medium bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200 hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
            >
              Reset to Original
            </button>
          )}
        </div>
      </div>

      {isPersonalized && (
        <div className="flex items-center space-x-2 text-green-600 dark:text-green-400">
          <div className="w-2 h-2 bg-green-500 rounded-full"></div>
          <span>Content personalized based on your profile</span>
        </div>
      )}
    </div>
  );
};

export default PersonalizationControls;