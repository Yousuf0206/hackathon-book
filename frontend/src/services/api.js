/**
 * API service for interacting with the backend
 */

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || '/api';

class ApiService {
  constructor() {
    this.baseURL = API_BASE_URL;
  }

  async chat(query, selectedText = null, sessionId = null, targetLanguage = "en") {
    try {
      // Get user ID from localStorage if available
      const userStr = localStorage.getItem('user');
      let userId = null;
      if (userStr) {
        try {
          const user = JSON.parse(userStr);
          userId = user.id || null;
        } catch (e) {
          console.warn('Could not parse user data from localStorage');
        }
      }

      const response = await fetch(`${this.baseURL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query,
          selected_text: selectedText,
          session_id: sessionId,
          user_id: userId,  // Pass user ID for personalization
          target_language: targetLanguage  // Pass target language for multilingual support
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Chat API error:', error);
      throw error;
    }
  }

  async query(query, topK = 5, selectedText = null) {
    try {
      const response = await fetch(`${this.baseURL}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query,
          top_k: topK,
          selected_text: selectedText
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Query API error:', error);
      throw error;
    }
  }

  async health() {
    try {
      const response = await fetch(`${this.baseURL}/health`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Health check error:', error);
      throw error;
    }
  }
}

export default new ApiService();