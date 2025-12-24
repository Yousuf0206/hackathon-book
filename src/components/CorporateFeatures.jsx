import React from 'react';

const CorporateFeatures = () => {
  const features = [
    {
      title: "AI-Human Collaboration",
      description: "Advanced frameworks for seamless interaction between artificial intelligence and human operators in robotic systems.",
      icon: (
        <svg className="w-8 h-8" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="currentColor" strokeWidth="2" className="text-cyan-400"/>
          <path d="M2 17L12 22L22 17" stroke="currentColor" strokeWidth="2" className="text-cyan-400"/>
          <path d="M2 12L12 17L22 12" stroke="currentColor" strokeWidth="2" className="text-cyan-400"/>
        </svg>
      ),
      gradient: "from-cyan-400 to-blue-500"
    },
    {
      title: "Intelligent Robotics",
      description: "Cognitive robotic systems that learn from human interaction and adapt to complex environments autonomously.",
      icon: (
        <svg className="w-8 h-8" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <rect x="3" y="3" width="18" height="18" rx="2" stroke="currentColor" strokeWidth="2" className="text-purple-400"/>
          <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="2" className="text-purple-400"/>
          <path d="M12 6V9M12 15V18" stroke="currentColor" strokeWidth="2" className="text-purple-400"/>
        </svg>
      ),
      gradient: "from-purple-400 to-pink-500"
    },
    {
      title: "Adaptive Learning",
      description: "Machine learning systems that evolve through human feedback, creating continuously improving robotic behaviors.",
      icon: (
        <svg className="w-8 h-8" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" className="text-blue-400"/>
          <circle cx="12" cy="10" r="3" fill="currentColor" className="text-blue-400"/>
          <path d="M7 20.5C7 18.5 8.5 17 10.5 17H13.5C15.5 17 17 18.5 17 20.5" stroke="currentColor" strokeWidth="2" className="text-blue-400"/>
        </svg>
      ),
      gradient: "from-blue-400 to-cyan-500"
    }
  ];

  return (
    <div className="corporate-features-section py-20 bg-gradient-to-b from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16 relative">
          <h2 className="section-title text-4xl lg:text-5xl font-bold text-slate-900 dark:text-white mb-4 relative z-10">
            AI-Human Intelligence Integration
          </h2>
          <p className="section-subtitle text-xl text-slate-600 dark:text-slate-300 max-w-3xl mx-auto relative z-10">
            Advanced technologies bridging artificial intelligence with human expertise
          </p>

          {/* Decorative element */}
          <div className="absolute -top-8 left-1/2 transform -translate-x-1/2 w-24 h-1 bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full opacity-30"></div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-10">
          {features.map((feature, index) => (
            <div
              key={index}
              className="feature-card group relative bg-white/80 dark:bg-slate-800/60 backdrop-blur-lg rounded-3xl p-10 shadow-xl hover:shadow-2xl transition-all duration-700 border border-white/20 dark:border-slate-700/50 hover:border-cyan-400/30 dark:hover:border-cyan-500/30 overflow-hidden hover:-translate-y-3"
            >
              {/* Animated background gradient */}
              <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/5 via-transparent to-blue-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-700"></div>

              {/* Floating elements */}
              <div className="absolute top-6 right-6 w-20 h-20 bg-gradient-to-r from-cyan-500/10 to-blue-500/10 rounded-full blur-xl animate-pulse"></div>
              <div className="absolute bottom-6 left-6 w-16 h-16 bg-gradient-to-r from-purple-500/10 to-pink-500/10 rounded-full blur-xl animate-pulse" style={{animationDelay: '1s'}}></div>

              <div className="relative z-10">
                <div className="feature-icon mb-6">
                  <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-br from-slate-100/80 to-slate-200/80 dark:from-slate-700/50 dark:to-slate-600/50 rounded-2xl group-hover:from-cyan-500/10 group-hover:to-blue-500/10 transition-all duration-500 shadow-lg group-hover:shadow-xl">
                    <div className={`w-10 h-10 bg-gradient-to-br ${feature.gradient} bg-clip-text text-transparent`}>
                      {feature.icon}
                    </div>
                  </div>
                </div>

                <h3 className="feature-title text-2xl font-bold text-slate-900 dark:text-white mb-4 group-hover:text-cyan-600 dark:group-hover:text-cyan-400 transition-colors duration-300">
                  {feature.title}
                </h3>

                <p className="feature-description text-slate-600 dark:text-slate-300 leading-relaxed mb-6">
                  {feature.description}
                </p>

                {/* Enhanced connection indicator */}
                <div className="pt-6 border-t border-slate-200/30 dark:border-slate-700/30">
                  <div className="flex items-center justify-between">
                    <span className="inline-flex items-center text-sm text-slate-500 dark:text-slate-400 group-hover:text-cyan-600 dark:group-hover:text-cyan-400 transition-colors duration-300">
                      <span className="inline-block w-3 h-3 bg-gradient-to-r from-cyan-400 to-blue-400 rounded-full mr-2 animate-pulse"></span>
                      AI-Human Connection
                    </span>
                    <span className="text-xs px-3 py-1.5 bg-gradient-to-r from-cyan-500/10 to-purple-500/10 rounded-full text-cyan-600 dark:text-cyan-400 font-medium">
                      Intelligent
                    </span>
                  </div>
                </div>
              </div>

              {/* Hover effect overlay */}
              <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/0 via-transparent to-blue-500/0 group-hover:from-cyan-500/10 group-hover:to-blue-500/10 transition-all duration-700 pointer-events-none"></div>
            </div>
          ))}
        </div>

        {/* Enhanced connection visualization between features */}
        <div className="connection-visualization mt-20 relative">
          <div className="relative">
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="w-full max-w-3xl h-px bg-gradient-to-r from-transparent via-cyan-400/20 to-transparent"></div>
            </div>
            <div className="flex justify-center items-center space-x-12 relative z-10">
              <div className="connection-node w-5 h-5 bg-gradient-to-r from-cyan-400 to-blue-400 rounded-full animate-pulse shadow-lg shadow-cyan-400/30"></div>
              <div className="connection-line w-24 h-0.5 bg-gradient-to-r from-cyan-400 to-purple-400 relative">
                <div className="absolute right-0 top-1/2 transform -translate-y-1/2 w-2 h-2 bg-cyan-400 rounded-full animate-ping"></div>
              </div>
              <div className="connection-node w-5 h-5 bg-gradient-to-r from-purple-400 to-pink-400 rounded-full animate-pulse shadow-lg shadow-purple-400/30" style={{animationDelay: '0.5s'}}></div>
              <div className="connection-line w-24 h-0.5 bg-gradient-to-r from-purple-400 to-blue-400 relative">
                <div className="absolute right-0 top-1/2 transform -translate-y-1/2 w-2 h-2 bg-purple-400 rounded-full animate-ping" style={{animationDelay: '0.5s'}}></div>
              </div>
              <div className="connection-node w-5 h-5 bg-gradient-to-r from-blue-400 to-cyan-400 rounded-full animate-pulse shadow-lg shadow-blue-400/30" style={{animationDelay: '1s'}}></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CorporateFeatures;