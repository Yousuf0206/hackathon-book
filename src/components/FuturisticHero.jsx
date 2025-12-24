import React, { useState, useEffect } from 'react';

const FuturisticHero = () => {
  const [animatedText, setAnimatedText] = useState('');
  const fullText = "Physical AI & Humanoid Robotics";
  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    if (currentIndex < fullText.length) {
      const timeout = setTimeout(() => {
        setAnimatedText(prev => prev + fullText[currentIndex]);
        setCurrentIndex(prev => prev + 1);
      }, 100);
      return () => clearTimeout(timeout);
    }
  }, [currentIndex, fullText]);

  return (
    <section className="futuristic-hero-container">
      {/* Animated background elements */}
      <div className="futuristic-background">
        <div className="futuristic-bg-element element-1"></div>
        <div className="futuristic-bg-element element-2"></div>
        <div className="futuristic-bg-element element-3"></div>

        {/* Cyber grid overlay */}
        <div className="cyber-grid-overlay"></div>

        {/* Scanning line effect */}
        <div className="scanning-line"></div>
      </div>

      {/* Floating elements */}
      <div className="floating-elements-container">
        {[...Array(20)].map((_, i) => (
          <div
            key={i}
            className="floating-element"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              animationDelay: `${Math.random() * 3}s`,
              animationDuration: `${3 + Math.random() * 2}s`
            }}
          ></div>
        ))}
      </div>

      <div className="futuristic-content-container">
        <div className="hero-content">
          <div className="hero-text-section">
            <h1 className="hero-title cyberpunk-title">
              <span className="hero-title-gradient">
                {animatedText}
              </span>
              <span className="cursor-blink">|</span>
            </h1>
            <p className="hero-subtitle">
              Master the convergence of artificial intelligence and humanoid robotics with our cutting-edge curriculum
            </p>
          </div>

          <div className="hero-buttons-section">
            <button className="futuristic-btn neon-glow hero-btn">
              Start Learning
            </button>
            <button className="futuristic-btn neon-glow hero-btn hero-btn-secondary">
              View Curriculum
            </button>
          </div>

          {/* Stats bar */}
          <div className="hero-stats">
            <div className="stat-item">
              <div className="stat-number stat-number-1">50+</div>
              <div className="stat-label">Learning Modules</div>
            </div>
            <div className="stat-item">
              <div className="stat-number stat-number-2">200+</div>
              <div className="stat-label">Practical Projects</div>
            </div>
            <div className="stat-item">
              <div className="stat-number stat-number-3">10k+</div>
              <div className="stat-label">Engineers Trained</div>
            </div>
          </div>
        </div>
      </div>

      {/* Scroll indicator */}
      <div className="scroll-indicator">
        <div className="scroll-indicator-wrapper">
          <div className="scroll-indicator-dot"></div>
        </div>
      </div>
    </section>
  );
};

export default FuturisticHero;