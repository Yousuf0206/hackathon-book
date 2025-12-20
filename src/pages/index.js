import React, { useState } from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';

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
        {/* Large Futuristic Banner */}
        <div className="futuristic-banner bg-gradient-to-br from-slate-900 via-blue-900 to-indigo-900 min-h-screen flex items-center justify-center relative overflow-hidden">
          {/* Animated Background Elements */}
          <div className="absolute inset-0">
            <div className="absolute top-20 left-10 w-72 h-72 bg-blue-500/10 rounded-full blur-3xl animate-pulse"></div>
            <div className="absolute bottom-20 right-10 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl animate-pulse" style={{animationDelay: '2s'}}></div>
            <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-indigo-500/5 rounded-full blur-3xl"></div>
          </div>

          {/* Floating Grid Pattern */}
          <div className="absolute inset-0 bg-grid-pattern bg-[length:60px_60px] opacity-20"></div>

          {/* Main Content */}
          <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
            {/* Left Side - Text Content */}
            <div className="text-content text-center lg:text-left">
              <div className="badge bg-gradient-to-r from-blue-500 to-purple-600 text-white px-4 py-2 rounded-full text-sm font-medium inline-block mb-6">
                AI & Robotics Education Platform
              </div>

              <h1 className="title-text text-5xl lg:text-7xl font-bold mb-6 leading-tight">
                <span className="block bg-clip-text text-transparent bg-gradient-to-r from-blue-300 to-purple-300">Physical AI</span>
                <span className="block bg-clip-text text-transparent bg-gradient-to-r from-cyan-300 to-blue-300">Humanoid Robotics</span>
              </h1>

              <p className="subtitle-text text-xl lg:text-2xl text-blue-100 mb-8 max-w-2xl">
                Master the convergence of artificial intelligence and humanoid robotics with our comprehensive guide to ROS 2, NVIDIA Isaac, Unity, and Vision-Language-Action technologies.
              </p>

              <div className="cta-buttons flex flex-col sm:flex-row gap-4 justify-center lg:justify-start">
                <button
                  onClick={() => setShowAuthModal(true)}
                  className="primary-cta bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white font-bold py-4 px-8 rounded-full text-lg transition-all duration-300 transform hover:scale-105 shadow-2xl hover:shadow-blue-500/25"
                >
                  Start Learning
                </button>

                <Link
                  className="secondary-cta bg-white/10 backdrop-blur-sm hover:bg-white/20 text-white font-semibold py-4 px-8 rounded-full text-lg transition-all duration-300 border border-white/20 hover:border-white/40"
                  to="/docs/intro">
                  Explore Curriculum
                </Link>
              </div>

              <div className="auth-prompt mt-8 text-blue-200">
                <p>Already have an account?{' '}
                  <button
                    className="auth-link text-blue-300 hover:text-white underline transition-colors"
                    onClick={() => {
                      setIsLogin(true);
                      setShowAuthModal(true);
                    }}
                  >
                    Sign in
                  </button>
                </p>
              </div>
            </div>

            {/* Right Side - Futuristic Robot/Human Interaction Visual */}
            <div className="visual-content flex justify-center">
              <div className="interaction-scene relative">
                {/* Robot Character */}
                <div className="robot-character absolute left-0 top-1/2 transform -translate-y-1/2">
                  <div className="robot-body bg-gradient-to-b from-gray-800 to-gray-900 rounded-2xl p-8 border-2 border-blue-400/30 shadow-2xl shadow-blue-500/25">
                    <div className="robot-head bg-gradient-to-b from-blue-400 to-blue-600 rounded-full w-24 h-24 mx-auto mb-4 flex items-center justify-center border-2 border-blue-300/50">
                      <div className="robot-eye flex space-x-2">
                        <div className="w-3 h-3 bg-cyan-300 rounded-full animate-pulse"></div>
                        <div className="w-3 h-3 bg-cyan-300 rounded-full animate-pulse" style={{animationDelay: '0.5s'}}></div>
                      </div>
                    </div>
                    <div className="robot-antenna">
                      <div className="w-1 h-8 bg-blue-400 mx-auto rounded-t-full"></div>
                      <div className="w-3 h-3 bg-cyan-300 rounded-full -mt-1 mx-auto animate-pulse"></div>
                    </div>
                    <div className="robot-display mt-4 bg-black/50 rounded p-2">
                      <div className="text-cyan-300 text-xs font-mono text-center">AI MODE</div>
                    </div>
                  </div>
                  <div className="robot-base mt-4 bg-gradient-to-b from-gray-700 to-gray-800 rounded-lg p-2 border border-gray-600/50">
                    <div className="flex justify-center space-x-1">
                      <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                      <div className="w-2 h-2 bg-yellow-400 rounded-full animate-pulse" style={{animationDelay: '0.2s'}}></div>
                      <div className="w-2 h-2 bg-red-400 rounded-full animate-pulse" style={{animationDelay: '0.4s'}}></div>
                    </div>
                  </div>
                </div>

                {/* Interaction Elements */}
                <div className="interaction-arrows absolute left-1/2 top-1/2 transform -translate-x-1/2 -translate-y-1/2">
                  <div className="arrow-wave flex flex-col items-center">
                    <div className="arrow-circle w-16 h-16 rounded-full border-4 border-blue-400/30 flex items-center justify-center mb-2 animate-ping">
                      <div className="w-8 h-8 rounded-full bg-gradient-to-r from-blue-400 to-cyan-400"></div>
                    </div>
                    <div className="arrow-lines">
                      <div className="w-1 h-16 bg-gradient-to-b from-blue-400/50 to-transparent mx-auto"></div>
                    </div>
                  </div>
                </div>

                {/* Human Character */}
                <div className="human-character absolute right-0 top-1/2 transform -translate-y-1/2">
                  <div className="human-body bg-gradient-to-b from-blue-600/20 to-purple-600/20 rounded-2xl p-8 border-2 border-purple-400/30 shadow-2xl shadow-purple-500/25">
                    <div className="human-head bg-gradient-to-b from-amber-200 to-amber-300 rounded-full w-24 h-24 mx-auto mb-4 flex items-center justify-center border-2 border-amber-100/50">
                      <div className="human-face">
                        <div className="eyes flex justify-center space-x-4 mb-2">
                          <div className="w-3 h-3 bg-gray-800 rounded-full"></div>
                          <div className="w-3 h-3 bg-gray-800 rounded-full"></div>
                        </div>
                        <div className="mouth w-8 h-1 bg-gray-800 rounded-full mx-auto"></div>
                      </div>
                    </div>
                    <div className="human-hat bg-gradient-to-b from-red-500 to-red-600 rounded-full w-12 h-8 mx-auto -mt-6"></div>
                    <div className="human-display mt-4 bg-black/50 rounded p-2">
                      <div className="text-purple-300 text-xs font-mono text-center">LEARNING</div>
                    </div>
                  </div>
                  <div className="human-base mt-4 bg-gradient-to-b from-purple-700/50 to-purple-800/50 rounded-lg p-2 border border-purple-600/50">
                    <div className="flex justify-center space-x-1">
                      <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                      <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse" style={{animationDelay: '0.2s'}}></div>
                      <div className="w-2 h-2 bg-purple-400 rounded-full animate-pulse" style={{animationDelay: '0.4s'}}></div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Floating Elements */}
          <div className="floating-elements absolute inset-0 pointer-events-none">
            <div className="floating-element floating-1 w-4 h-4 bg-blue-400/60 rounded-full animate-float" style={{left: '10%', top: '20%'}}></div>
            <div className="floating-element floating-2 w-3 h-3 bg-purple-400/60 rounded-full animate-float" style={{left: '85%', top: '15%', animationDelay: '1s'}}></div>
            <div className="floating-element floating-3 w-2 h-2 bg-cyan-400/60 rounded-full animate-float" style={{left: '70%', top: '70%', animationDelay: '2s'}}></div>
            <div className="floating-element floating-4 w-3 h-3 bg-indigo-400/60 rounded-full animate-float" style={{left: '20%', top: '65%', animationDelay: '0.5s'}}></div>
          </div>
        </div>

        {/* Features Section */}
        <div className="features-section container mx-auto px-4 py-20">
          <div className="text-center mb-16">
            <h2 className="section-title text-4xl font-bold text-gray-900 dark:text-white mb-4">Cutting-Edge Technologies</h2>
            <p className="section-subtitle text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
              Master the tools and frameworks powering the future of AI and robotics
            </p>
          </div>

          <div className="features-grid grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="feature-card bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-xl hover:shadow-2xl transition-all duration-300 border border-gray-200 dark:border-gray-700 transform hover:-translate-y-2">
              <div className="feature-icon text-6xl mb-6">🤖</div>
              <h3 className="feature-title text-2xl font-bold mb-4 text-gray-900 dark:text-white">ROS 2 & Gazebo</h3>
              <p className="feature-description text-gray-600 dark:text-gray-300">
                Learn robot operating system fundamentals and advanced simulation techniques for developing and testing robotic applications in realistic environments.
              </p>
            </div>

            <div className="feature-card bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-xl hover:shadow-2xl transition-all duration-300 border border-gray-200 dark:border-gray-700 transform hover:-translate-y-2">
              <div className="feature-icon text-6xl mb-6">🎮</div>
              <h3 className="feature-title text-2xl font-bold mb-4 text-gray-900 dark:text-white">Unity & NVIDIA Isaac</h3>
              <p className="feature-description text-gray-600 dark:text-gray-300">
                Build immersive digital twins and leverage NVIDIA's AI-accelerated robotics platform for training and deploying intelligent robotic systems.
              </p>
            </div>

            <div className="feature-card bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-xl hover:shadow-2xl transition-all duration-300 border border-gray-200 dark:border-gray-700 transform hover:-translate-y-2">
              <div className="feature-icon text-6xl mb-6">👁️</div>
              <h3 className="feature-title text-2xl font-bold mb-4 text-gray-900 dark:text-white">Vision-Language-Action</h3>
              <p className="feature-description text-gray-600 dark:text-gray-300">
                Explore state-of-the-art VLA models that enable robots to perceive, understand, and act in complex real-world environments using multimodal AI.
              </p>
            </div>
          </div>
        </div>

        {/* Stats Section */}
        <div className="stats-section bg-gradient-to-r from-gray-50 to-blue-50 dark:from-gray-800 dark:to-gray-900 py-20">
          <div className="container mx-auto px-4">
            <div className="stats-grid grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-8 text-center">
              <div className="stat-item">
                <div className="stat-number text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-500 to-purple-600">50+</div>
                <div className="stat-label text-lg text-gray-600 dark:text-gray-300">Learning Modules</div>
              </div>
              <div className="stat-item">
                <div className="stat-number text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-purple-500 to-pink-600">200+</div>
                <div className="stat-label text-lg text-gray-600 dark:text-gray-300">Hands-on Projects</div>
              </div>
              <div className="stat-item">
                <div className="stat-number text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-cyan-500 to-blue-600">10k+</div>
                <div className="stat-label text-lg text-gray-600 dark:text-gray-300">Active Learners</div>
              </div>
              <div className="stat-item">
                <div className="stat-number text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-green-500 to-teal-600">∞</div>
                <div className="stat-label text-lg text-gray-600 dark:text-gray-300">AI Assistance</div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </Layout>
  );
}