import React from 'react';

const CorporateStats = () => {
  const stats = [
    {
      number: "50+",
      label: "AI-Human Projects",
      description: "Collaborative intelligence implementations",
      icon: "🤖",
      color: "text-cyan-500",
      gradient: "from-cyan-500 to-blue-500"
    },
    {
      number: "200+",
      label: "Robotic Systems",
      description: "Intelligent machines with human oversight",
      icon: "🤖",
      color: "text-purple-500",
      gradient: "from-purple-500 to-pink-500"
    },
    {
      number: "10k+",
      label: "Engineers Trained",
      description: "AI-Human collaboration specialists",
      icon: "👨‍💻",
      color: "text-blue-500",
      gradient: "from-blue-500 to-cyan-500"
    },
    {
      number: "∞",
      label: "Possibilities",
      description: "Limitless AI-Human potential",
      icon: "⚡",
      color: "text-slate-500",
      gradient: "from-slate-500 to-gray-500"
    }
  ];

  return (
    <div className="corporate-stats-section py-20 bg-gradient-to-b from-slate-100 to-slate-200 dark:from-slate-800 dark:to-slate-900">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-8">
          {stats.map((stat, index) => (
            <div
              key={index}
              className="stat-item group relative bg-white/80 dark:bg-slate-800/60 backdrop-blur-xl rounded-3xl p-10 shadow-2xl hover:shadow-2xl transition-all duration-700 border border-white/20 dark:border-slate-700/50 hover:border-cyan-400/30 dark:hover:border-cyan-500/30 overflow-hidden hover:-translate-y-3"
            >
              {/* Animated background gradient */}
              <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/5 via-transparent to-purple-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-700"></div>

              {/* Floating elements */}
              <div className="absolute top-4 right-4 w-16 h-16 bg-gradient-to-r from-cyan-500/5 to-blue-500/5 rounded-full blur-lg animate-pulse"></div>
              <div className="absolute bottom-4 left-4 w-12 h-12 bg-gradient-to-r from-purple-500/5 to-pink-500/5 rounded-full blur-lg animate-pulse" style={{animationDelay: '0.5s'}}></div>

              <div className="relative z-10 text-center">
                <div className="stat-icon text-4xl mb-6 relative z-10">{stat.icon}</div>
                <div className={`stat-number text-6xl lg:text-7xl font-bold bg-gradient-to-r ${stat.gradient} bg-clip-text text-transparent mb-4 group-hover:scale-110 transition-transform duration-500`}>
                  {stat.number}
                </div>
                <div className="stat-label text-xl font-bold text-slate-900 dark:text-white mb-3 relative z-10">
                  {stat.label}
                </div>
                <div className="stat-description text-slate-600 dark:text-slate-400 relative z-10">
                  {stat.description}
                </div>
              </div>

              {/* Enhanced connection indicator */}
              <div className="absolute top-6 right-6 w-3 h-3 bg-gradient-to-r from-cyan-400 to-purple-400 rounded-full animate-pulse shadow-lg shadow-cyan-400/30"></div>

              {/* Hover animation element */}
              <div className="absolute top-0 right-0 w-24 h-24 bg-gradient-to-br from-cyan-400/10 to-purple-500/10 rounded-full opacity-0 group-hover:opacity-100 transition-opacity duration-700 -translate-y-12 translate-x-12"></div>
            </div>
          ))}
        </div>

        {/* Enhanced AI-Human collaboration visualization */}
        <div className="ai-human-connection mt-24 text-center relative">
          <div className="connection-title text-3xl font-bold text-slate-900 dark:text-white mb-12 relative z-10">
            The Future of AI-Human Intelligence
          </div>

          <div className="relative">
            {/* Background connection line */}
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="w-full max-w-2xl h-px bg-gradient-to-r from-transparent via-cyan-400/30 to-transparent"></div>
            </div>

            <div className="flex items-center justify-center space-x-16 relative z-10">
              <div className="ai-node text-center group">
                <div className="w-24 h-24 bg-gradient-to-br from-cyan-500/10 to-blue-500/10 rounded-2xl flex items-center justify-center border-2 border-cyan-400/30 mx-auto mb-4 group-hover:border-cyan-400/60 transition-colors duration-500 shadow-lg group-hover:shadow-xl group-hover:shadow-cyan-400/20">
                  <span className="text-cyan-400 text-4xl">🤖</span>
                </div>
                <div className="text-base font-medium text-slate-700 dark:text-slate-300 group-hover:text-cyan-600 dark:group-hover:text-cyan-400 transition-colors duration-300">
                  Artificial Intelligence
                </div>
              </div>

              <div className="connection-center relative">
                <div className="w-16 h-16 bg-gradient-to-br from-cyan-500 to-purple-500 rounded-full flex items-center justify-center animate-spin-slow shadow-lg shadow-cyan-500/30">
                  <span className="text-white text-xl">🤝</span>
                </div>
                <div className="absolute -inset-4 rounded-full border border-cyan-400/20 animate-ping-slow"></div>
              </div>

              <div className="human-node text-center group">
                <div className="w-24 h-24 bg-gradient-to-br from-purple-500/10 to-pink-500/10 rounded-2xl flex items-center justify-center border-2 border-purple-400/30 mx-auto mb-4 group-hover:border-purple-400/60 transition-colors duration-500 shadow-lg group-hover:shadow-xl group-hover:shadow-purple-400/20">
                  <span className="text-purple-400 text-4xl">👤</span>
                </div>
                <div className="text-base font-medium text-slate-700 dark:text-slate-300 group-hover:text-purple-600 dark:group-hover:text-purple-400 transition-colors duration-300">
                  Human Intelligence
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CorporateStats;