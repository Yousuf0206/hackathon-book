import React, { useState } from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';

// Homepage components
import FuturisticHero from '../components/FuturisticHero.jsx';
import FuturisticFeatures from '../components/FuturisticFeatures';
import FuturisticTestimonials from '../components/FuturisticTestimonials';
import TechStack from '../components/TechStack';
import AIAssistantPreview from '../components/AIAssistantPreview';
import FuturisticCTA from '../components/FuturisticCTA';
import ModernFooter from '../components/ModernFooter';

export default function Home() {
  const { siteConfig } = useDocusaurusContext();
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [isLogin, setIsLogin] = useState(true);
  const [authData, setAuthData] = useState({
    email: '',
    password: '',
    name: '',
  });

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setAuthData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleAuthSubmit = (e) => {
    e.preventDefault();
    console.log(
      isLogin ? 'Login attempt' : 'Registration attempt',
      authData
    );
    setShowAuthModal(false);
  };

  return (
    <Layout
      title={siteConfig.title}
      description="A Comprehensive Guide to Physical AI & Humanoid Robotics"
    >
      {/* AUTH MODAL */}
      {showAuthModal && (
        <div className="authOverlay" onClick={() => setShowAuthModal(false)}>
          <div
            className="authModal"
            onClick={(e) => e.stopPropagation()}
          >
            <header className="authHeader">
              <h2>{isLogin ? 'Sign In' : 'Create Account'}</h2>
              <button
                className="authClose"
                onClick={() => setShowAuthModal(false)}
              >
                ×
              </button>
            </header>

            <form onSubmit={handleAuthSubmit} className="authForm">
              {!isLogin && (
                <input
                  type="text"
                  name="name"
                  placeholder="Full Name"
                  value={authData.name}
                  onChange={handleInputChange}
                  required
                />
              )}

              <input
                type="email"
                name="email"
                placeholder="Email"
                value={authData.email}
                onChange={handleInputChange}
                required
              />

              <input
                type="password"
                name="password"
                placeholder="Password"
                value={authData.password}
                onChange={handleInputChange}
                required
                minLength={6}
              />

              <button type="submit">
                {isLogin ? 'Sign In' : 'Create Account'}
              </button>
            </form>

            <p className="authSwitch">
              {isLogin
                ? "Don't have an account?"
                : 'Already have an account?'}{' '}
              <span onClick={() => setIsLogin(!isLogin)}>
                {isLogin ? 'Sign Up' : 'Sign In'}
              </span>
            </p>
          </div>
        </div>
      )}

      {/* MAIN CONTENT */}
      <main className="homepage">
        <FuturisticHero />
        <FuturisticFeatures />
        <FuturisticTestimonials />
        <TechStack />
        <AIAssistantPreview />
        <FuturisticCTA />
      </main>

      <ModernFooter />
    </Layout>
  );
}
