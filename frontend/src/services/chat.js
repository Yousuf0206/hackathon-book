/**
 * Chat session management service
 */

class ChatService {
  constructor() {
    this.sessions = new Map();
    this.currentSessionId = null;
  }

  createSession(sessionId = null) {
    const id = sessionId || this.generateSessionId();
    const session = {
      id,
      createdAt: new Date(),
      messages: [],
      metadata: {}
    };

    this.sessions.set(id, session);
    this.currentSessionId = id;
    return session;
  }

  getSession(sessionId) {
    if (sessionId) {
      return this.sessions.get(sessionId);
    }
    return this.sessions.get(this.currentSessionId);
  }

  addMessage(sessionId, message) {
    const session = this.getSession(sessionId);
    if (session) {
      session.messages.push({
        ...message,
        timestamp: new Date()
      });
    }
  }

  getMessages(sessionId) {
    const session = this.getSession(sessionId);
    return session ? session.messages : [];
  }

  clearSession(sessionId) {
    if (sessionId) {
      this.sessions.delete(sessionId);
      if (this.currentSessionId === sessionId) {
        this.currentSessionId = null;
      }
    } else {
      this.sessions.clear();
      this.currentSessionId = null;
    }
  }

  generateSessionId() {
    return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
  }

  getCurrentSessionId() {
    return this.currentSessionId;
  }

  setCurrentSessionId(sessionId) {
    if (this.sessions.has(sessionId)) {
      this.currentSessionId = sessionId;
    }
  }
}

export default new ChatService();