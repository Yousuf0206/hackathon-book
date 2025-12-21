import React from 'react';

const CorporateBanner = () => {
  return (
    <div className="corporate-hero min-h-screen flex items-center justify-center relative overflow-hidden bg-slate-900">
      {/* Background image */}
      <div
        className="absolute inset-0 bg-cover bg-center bg-no-repeat"
        style={{
          backgroundImage: "url('/img/robotic-human-banner.jpg')",
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          zIndex: 0
        }}
      >
        {/* Dark overlay for better text visibility */}
        <div className="absolute inset-0 bg-gradient-to-br from-slate-900/80 via-slate-800/70 to-slate-900/80"></div>
      </div>

      {/* Main content */}
      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
        <div className="company-badge bg-slate-800/60 backdrop-blur-sm border border-slate-600/50 text-cyan-400 px-6 py-3 rounded-full text-sm font-medium inline-block mb-8">
          AI & HUMAN INTELLIGENCE CONVERGENCE
        </div>

        <h1 className="hero-title text-5xl lg:text-7xl font-bold leading-tight mb-6">
          <span className="block text-white mb-4">Physical AI &</span>
          <span className="block text-4xl lg:text-6xl bg-gradient-to-r from-cyan-400 via-purple-400 to-blue-400 bg-clip-text text-transparent">
            Humanoid Robotics
          </span>
        </h1>

        <p className="hero-subtitle text-xl text-slate-300 leading-relaxed max-w-3xl mx-auto mb-12">
          Master the convergence of artificial intelligence and humanoid robotics.
          Explore ROS 2, NVIDIA Isaac, Unity, and Vision-Language-Action technologies.
        </p>

        <div className="cta-container flex flex-col sm:flex-row gap-6 justify-center">
          <button className="primary-cta bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700 text-white font-semibold py-4 px-10 rounded-xl text-lg transition-all duration-300 transform hover:scale-105 shadow-2xl shadow-cyan-500/25 backdrop-blur-sm">
            Start Learning
          </button>
          <button className="secondary-cta bg-slate-800/60 backdrop-blur-sm hover:bg-slate-700/60 text-white font-semibold py-4 px-10 rounded-xl text-lg transition-all duration-300 border border-slate-600/50 hover:border-cyan-400/50">
            View Curriculum
          </button>
        </div>

        {/* Stats bar */}
        <div className="stats-bar flex justify-center space-x-12 pt-16">
          <div className="stat-item text-center">
            <div className="stat-number text-4xl font-bold text-cyan-400">50+</div>
            <div className="stat-label text-sm text-slate-400">Learning Modules</div>
          </div>
          <div className="stat-item text-center">
            <div className="stat-number text-4xl font-bold text-purple-400">200+</div>
            <div className="stat-label text-sm text-slate-400">Projects</div>
          </div>
          <div className="stat-item text-center">
            <div className="stat-number text-4xl font-bold text-blue-400">10k+</div>
            <div className="stat-label text-sm text-slate-400">Engineers</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CorporateBanner;