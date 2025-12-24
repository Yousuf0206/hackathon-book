import React from 'react';

const FuturisticCTA = () => {
  return (
    <section className="cta-section padding-top--xl padding-bottom--xl">
      <div className="container">
        <div className="cta-container robotic-card neon-glow">
          <div className="cta-content">
            <h2 className="cta-title cyberpunk-title">
              Ready to <span className="cta-highlight">Shape</span> the Future?
            </h2>
            <p className="cta-subtitle">
              Join the next generation of robotics engineers and AI specialists. Start your journey in physical AI and humanoid robotics today.
            </p>
          </div>

          <div className="cta-buttons">
            <button className="futuristic-btn neon-glow cta-primary-btn">
              Begin Learning Journey
            </button>
            <button className="futuristic-btn neon-glow cta-secondary-btn">
              Explore Curriculum
            </button>
          </div>

          {/* Trust indicators */}
          <div className="cta-trust-indicators">
            <div className="trust-item">
              <div className="trust-icon icon-certified">✓</div>
              <span className="trust-title">Industry Certified</span>
              <span className="trust-subtitle">Program</span>
            </div>
            <div className="trust-item">
              <div className="trust-icon icon-support">📚</div>
              <span className="trust-title">24/7 Expert</span>
              <span className="trust-subtitle">Support</span>
            </div>
            <div className="trust-item">
              <div className="trust-icon icon-lifetime">♾️</div>
              <span className="trust-title">Lifetime</span>
              <span className="trust-subtitle">Access</span>
            </div>
            <div className="trust-item">
              <div className="trust-icon icon-curriculum">⚡</div>
              <span className="trust-title">Cutting-Edge</span>
              <span className="trust-subtitle">Curriculum</span>
            </div>
          </div>

          {/* Stats grid */}
          <div className="cta-stats-grid">
            <div className="stat-item">
              <div className="stat-number stat-number-1">50+</div>
              <div className="stat-label">Learning Modules</div>
            </div>
            <div className="stat-item">
              <div className="stat-number stat-number-2">200+</div>
              <div className="stat-label">Projects</div>
            </div>
            <div className="stat-item">
              <div className="stat-number stat-number-3">10k+</div>
              <div className="stat-label">Engineers</div>
            </div>
            <div className="stat-item">
              <div className="stat-number stat-number-4">∞</div>
              <div className="stat-label">Possibilities</div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default FuturisticCTA;