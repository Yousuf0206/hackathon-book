import React, { useState } from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import CorporateBanner from '../components/CorporateBanner';
import CorporateFeatures from '../components/CorporateFeatures';
import CorporateStats from '../components/CorporateStats';
import LearningPaths from '../components/LearningPaths';
import Testimonials from '../components/Testimonials';
import HomepageFooter from '../components/HomepageFooter';

export default function Home() {
  const { siteConfig } = useDocusaurusContext();
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [isLogin, setIsLogin] = useState(true);
  const [authData, setAuthData] = useState({
    email: '',
    password: '',
    name: ''
  });

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setAuthData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleAuthSubmit = (e) => {
    e.preventDefault();
    // In a real implementation, this would call your auth API
    console.log(isLogin ? 'Login attempt' : 'Registration attempt', authData);
    setShowAuthModal(false);
  };

  return (
    <Layout
      title={`${siteConfig.title}`}
      description="A Comprehensive Guide to Physical AI & Humanoid Robotics">
      {/* Authentication Modal */}
      {showAuthModal && (
        <div className="auth-modal-overlay" onClick={() => setShowAuthModal(false)}>
          <div className="auth-modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="auth-modal-header">
              <h2>{isLogin ? 'Sign In' : 'Create Account'}</h2>
              <button className="close-auth-modal" onClick={() => setShowAuthModal(false)}>×</button>
            </div>
            <form onSubmit={handleAuthSubmit} className="auth-form">
              {!isLogin && (
                <div className="form-group">
                  <label htmlFor="name">Full Name</label>
                  <input
                    type="text"
                    id="name"
                    name="name"
                    value={authData.name}
                    onChange={handleInputChange}
                    required={!isLogin}
                    className="auth-input"
                  />
                </div>
              )}
              <div className="form-group">
                <label htmlFor="email">Email</label>
                <input
                  type="email"
                  id="email"
                  name="email"
                  value={authData.email}
                  onChange={handleInputChange}
                  required
                  className="auth-input"
                />
              </div>
              <div className="form-group">
                <label htmlFor="password">Password</label>
                <input
                  type="password"
                  id="password"
                  name="password"
                  value={authData.password}
                  onChange={handleInputChange}
                  required
                  minLength="6"
                  className="auth-input"
                />
              </div>
              <button type="submit" className="auth-submit-button">
                {isLogin ? 'Sign In' : 'Create Account'}
              </button>
            </form>
            <div className="auth-switch">
              <p>
                {isLogin ? "Don't have an account?" : "Already have an account?"}{' '}
                <button
                  className="auth-switch-link"
                  onClick={() => setIsLogin(!isLogin)}
                >
                  {isLogin ? 'Sign Up' : 'Sign In'}
                </button>
              </p>
            </div>
          </div>
        </div>
      )}

      <main className="home-main">
        <CorporateBanner />
        <CorporateFeatures />
        <LearningPaths />
        <Testimonials />
        <CorporateStats />
      </main>
      <HomepageFooter />
    </Layout>
  );
}