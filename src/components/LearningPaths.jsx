import React from 'react';

const LearningPaths = () => {
  const paths = [
    {
      title: "ROS 2 Fundamentals",
      description: "Master the Robot Operating System for building complex robotic applications and multi-robot systems.",
      duration: "4 weeks",
      level: "Beginner",
      icon: "🤖",
      color: "from-cyan-400 to-blue-500"
    },
    {
      title: "AI-Human Interaction",
      description: "Explore the cutting-edge intersection of artificial intelligence and human-centered robotics.",
      duration: "6 weeks",
      level: "Intermediate",
      icon: "🤝",
      color: "from-purple-400 to-pink-500"
    },
    {
      title: "Computer Vision & Perception",
      description: "Implement vision systems that enable robots to understand and interact with their environment.",
      duration: "5 weeks",
      level: "Advanced",
      icon: "👁️",
      color: "from-blue-400 to-cyan-500"
    },
    {
      title: "Robotics Simulation",
      description: "Create realistic simulation environments using Unity and NVIDIA Isaac for robot training.",
      duration: "3 weeks",
      level: "Intermediate",
      icon: "🎮",
      color: "from-green-400 to-teal-500"
    }
  ];

  return (
    <div className="learning-paths-section py-20 bg-gradient-to-b from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="section-title text-4xl lg:text-5xl font-bold text-slate-900 dark:text-white mb-4">
            Learning Paths
          </h2>
          <p className="section-subtitle text-xl text-slate-600 dark:text-slate-300 max-w-3xl mx-auto">
            Structured curricula designed to advance your expertise in AI-humanoid robotics
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {paths.map((path, index) => (
            <div
              key={index}
              className="learning-path-card group relative bg-white/80 dark:bg-slate-800/60 backdrop-blur-sm rounded-2xl p-6 shadow-lg hover:shadow-2xl transition-all duration-500 border border-slate-200/50 dark:border-slate-700/50 hover:border-cyan-300/30 dark:hover:border-cyan-500/30 overflow-hidden"
            >
              <div className="absolute inset-0 bg-gradient-to-br opacity-0 group-hover:opacity-5 transition-opacity duration-500"></div>

              <div className="relative z-10">
                <div className="path-icon text-4xl mb-4">{path.icon}</div>

                <h3 className="path-title text-xl font-bold text-slate-900 dark:text-white mb-3 group-hover:text-cyan-600 dark:group-hover:text-cyan-400 transition-colors duration-300">
                  {path.title}
                </h3>

                <p className="path-description text-slate-600 dark:text-slate-300 text-sm mb-4">
                  {path.description}
                </p>

                <div className="path-meta flex justify-between text-xs text-slate-500 dark:text-slate-400 mt-4 pt-4 border-t border-slate-200/30 dark:border-slate-700/30">
                  <span className="flex items-center">
                    <span className="mr-1">⏱️</span> {path.duration}
                  </span>
                  <span className={`px-2 py-1 rounded-full text-xs bg-gradient-to-r ${path.color.includes('cyan') ? 'from-cyan-500/10 to-blue-500/10 text-cyan-500' : path.color.includes('purple') ? 'from-purple-500/10 to-pink-500/10 text-purple-500' : path.color.includes('blue') ? 'from-blue-500/10 to-cyan-500/10 text-blue-500' : 'from-green-500/10 to-teal-500/10 text-green-500'}`}>
                    {path.level}
                  </span>
                </div>
              </div>

              <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/0 via-transparent to-blue-500/0 group-hover:from-cyan-500/5 group-hover:to-blue-500/5 transition-all duration-500 pointer-events-none"></div>
            </div>
          ))}
        </div>

        <div className="curriculum-cta text-center mt-16">
          <button className="curriculum-button bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700 text-white font-semibold py-4 px-10 rounded-xl text-lg transition-all duration-300 transform hover:scale-105 shadow-2xl shadow-cyan-500/25 backdrop-blur-sm">
            Explore Full Curriculum
          </button>
        </div>
      </div>
    </div>
  );
};

export default LearningPaths;