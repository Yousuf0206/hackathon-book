import React from 'react';

const CorporateBanner = () => {
  return (
    <div className="corporate-hero min-h-screen flex items-center justify-center relative overflow-hidden bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      {/* Animated background elements */}
      <div className="absolute inset-0 overflow-hidden">
        {/* Animated grid pattern */}
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,rgba(120,119,198,0.1)_0%,transparent_70%)]"></div>
        <div className="absolute inset-0" style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%239C92AC' fill-opacity='0.05' fill-rule='nonzero'%3E%3Ccircle cx='30' cy='30' r='2'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`,
        }}></div>

        {/* Floating particles */}
        {[...Array(15)].map((_, i) => (
          <div
            key={i}
            className="absolute rounded-full bg-gradient-to-r from-cyan-500/10 to-blue-500/10"
            style={{
              width: `${Math.random() * 20 + 5}px`,
              height: `${Math.random() * 20 + 5}px`,
              top: `${Math.random() * 100}%`,
              left: `${Math.random() * 100}%`,
              animation: `float-particle ${Math.random() * 10 + 10}s infinite ease-in-out`,
              animationDelay: `${Math.random() * 5}s`
            }}
          ></div>
        ))}
      </div>

      {/* Main content */}
      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
        <div className="company-badge bg-white/10 backdrop-blur-lg border border-white/20 text-cyan-400 px-6 py-3 rounded-full text-sm font-medium inline-block mb-8 shadow-lg shadow-cyan-500/10">
          <span className="relative z-10">AI & HUMAN INTELLIGENCE CONVERGENCE</span>
          <div className="absolute inset-0 rounded-full bg-gradient-to-r from-cyan-500/20 to-blue-500/20 blur-xl -z-10"></div>
        </div>

        <h1 className="hero-title text-5xl lg:text-7xl font-bold leading-tight mb-6 relative">
          <span className="block text-white mb-4 relative z-10">Physical AI &</span>
          <span className="block text-4xl lg:text-6xl bg-gradient-to-r from-cyan-400 via-purple-400 to-blue-400 bg-clip-text text-transparent relative z-10">
            Humanoid Robotics
          </span>
          <div className="absolute -bottom-4 left-1/2 transform -translate-x-1/2 w-32 h-1 bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full"></div>
        </h1>

        <p className="hero-subtitle text-xl text-slate-300 leading-relaxed max-w-3xl mx-auto mb-12 relative z-10">
          Master the convergence of artificial intelligence and humanoid robotics.
          Explore ROS 2, NVIDIA Isaac, Unity, and Vision-Language-Action technologies.
        </p>

        <div className="cta-container flex flex-col sm:flex-row gap-6 justify-center relative z-10">
          <button className="primary-cta bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700 text-white font-semibold py-4 px-10 rounded-xl text-lg transition-all duration-300 transform hover:scale-105 shadow-2xl shadow-cyan-500/25 backdrop-blur-sm relative overflow-hidden group">
            <span className="relative z-10">Start Learning</span>
            <div className="absolute inset-0 bg-gradient-to-r from-white/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 -skew-x-12 -translate-x-full group-hover:translate-x-0"></div>
          </button>
          <button className="secondary-cta bg-white/10 backdrop-blur-lg hover:bg-white/20 text-white font-semibold py-4 px-10 rounded-xl text-lg transition-all duration-300 border border-white/30 hover:border-cyan-400/50 relative overflow-hidden group">
            <span className="relative z-10">View Curriculum</span>
            <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/10 to-blue-500/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
          </button>
        </div>

        {/* Stats bar */}
        <div className="stats-bar grid grid-cols-1 md:grid-cols-3 gap-8 pt-16 max-w-4xl mx-auto">
          <div className="stat-item text-center group">
            <div className="stat-number text-5xl font-bold bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent group-hover:scale-110 transition-transform duration-300">50+</div>
            <div className="stat-label text-lg text-slate-400 group-hover:text-cyan-400 transition-colors duration-300">Learning Modules</div>
          </div>
          <div className="stat-item text-center group">
            <div className="stat-number text-5xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent group-hover:scale-110 transition-transform duration-300">200+</div>
            <div className="stat-label text-lg text-slate-400 group-hover:text-purple-400 transition-colors duration-300">Projects</div>
          </div>
          <div className="stat-item text-center group">
            <div className="stat-number text-5xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent group-hover:scale-110 transition-transform duration-300">10k+</div>
            <div className="stat-label text-lg text-slate-400 group-hover:text-blue-400 transition-colors duration-300">Engineers</div>
          </div>
        </div>
      </div>

      {/* Bottom decorative elements */}
      <div className="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-cyan-500/30 to-transparent"></div>
    </div>
  );
};

export default CorporateBanner;