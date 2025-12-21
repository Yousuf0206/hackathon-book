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
              className="stat-item group relative bg-white/80 dark:bg-slate-800/60 backdrop-blur-sm rounded-2xl p-8 shadow-lg hover:shadow-2xl transition-all duration-500 border border-slate-200/50 dark:border-slate-700/50 hover:border-cyan-300/30 dark:hover:border-cyan-500/30 overflow-hidden"
            >
              {/* Animated background */}
              <div className="absolute inset-0 bg-gradient-to-br opacity-0 group-hover:opacity-5 transition-opacity duration-500"></div>

              <div className="relative z-10 text-center">
                <div className="stat-icon text-4xl mb-4">{stat.icon}</div>
                <div className={`stat-number text-5xl lg:text-6xl font-bold bg-gradient-to-r ${stat.gradient} bg-clip-text text-transparent mb-2 group-hover:scale-110 transition-transform duration-300`}>
                  {stat.number}
                </div>
                <div className="stat-label text-xl font-bold text-slate-900 dark:text-white mb-2">
                  {stat.label}
                </div>
                <div className="stat-description text-slate-600 dark:text-slate-400">
                  {stat.description}
                </div>
              </div>

              {/* Connection indicator */}
              <div className="absolute top-4 right-4 w-2 h-2 bg-gradient-to-r from-cyan-400 to-purple-400 rounded-full animate-pulse"></div>

              {/* Hover animation element */}
              <div className="absolute top-0 right-0 w-20 h-20 bg-gradient-to-br from-cyan-400/10 to-purple-500/10 rounded-full opacity-0 group-hover:opacity-100 transition-opacity duration-500 -translate-y-10 translate-x-10"></div>
            </div>
          ))}
        </div>

        {/* AI-Human collaboration visualization */}
        <div className="ai-human-connection mt-16 text-center">
          <div className="connection-title text-2xl font-bold text-slate-900 dark:text-white mb-8">
            The Future of AI-Human Intelligence
          </div>
          <div className="connection-flow relative flex justify-center items-center">
            <div className="flex items-center space-x-8">
              <div className="ai-node text-center">
                <div className="w-16 h-16 bg-gradient-to-br from-cyan-500/20 to-blue-500/20 rounded-full flex items-center justify-center border border-cyan-400/30 mx-auto mb-2">
                  <span className="text-cyan-400 text-2xl">🤖</span>
                </div>
                <div className="text-sm text-slate-600 dark:text-slate-300">Artificial Intelligence</div>
              </div>
              <div className="connection-arrow text-2xl animate-pulse">🔄</div>
              <div className="human-node text-center">
                <div className="w-16 h-16 bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-full flex items-center justify-center border border-purple-400/30 mx-auto mb-2">
                  <span className="text-purple-400 text-2xl">👤</span>
                </div>
                <div className="text-sm text-slate-600 dark:text-slate-300">Human Intelligence</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CorporateStats;