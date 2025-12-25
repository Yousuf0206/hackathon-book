import React, { useState } from 'react';
import './component-styles.css';

const LearningPathways = () => {
  const [selectedPath, setSelectedPath] = useState(0);

  const learningPaths = [
    {
      title: "Beginner's Path",
      description: "Start with fundamentals of robotics and AI",
      duration: "3-6 months",
      level: "Beginner",
      modules: [
        "Introduction to ROS 2",
        "Basic Python for Robotics",
        "Robot Kinematics",
        "Simple Navigation"
      ],
      color: "from-green-500 to-teal-500",
      icon: "🌱"
    },
    {
      title: "Developer's Path",
      description: "Build advanced robotic applications",
      duration: "6-12 months",
      level: "Intermediate",
      modules: [
        "Advanced ROS 2 Concepts",
        "Gazebo Simulation",
        "Computer Vision",
        "Path Planning"
      ],
      color: "from-blue-500 to-cyan-500",
      icon: "💻"
    },
    {
      title: "Expert's Path",
      description: "Master AI-humanoid integration",
      duration: "12+ months",
      level: "Advanced",
      modules: [
        "NVIDIA Isaac Ecosystem",
        "Deep Reinforcement Learning",
        "Human-Robot Interaction",
        "VLA Systems"
      ],
      color: "from-purple-500 to-pink-500",
      icon: "🚀"
    }
  ];

  const resources = [
    {
      title: "Interactive Tutorials",
      description: "Step-by-step guides with live coding",
      icon: "📝",
      count: "100+"
    },
    {
      title: "Video Lectures",
      description: "Expert-led video content",
      icon: "🎥",
      count: "200+"
    },
    {
      title: "Practical Projects",
      description: "Real-world implementation challenges",
      icon: "🎯",
      count: "50+"
    },
    {
      title: "Community Support",
      description: "Access to expert community",
      icon: "👥",
      count: "24/7"
    }
  ];

  return (
    <section className="robotic-learning-pathways-section">
      <div className="robotic-learning-pathways-container">
        <div className="robotic-learning-pathways-header">
          <h2 className="robotic-learning-pathways-title">
            Choose Your Learning Path
          </h2>
          <p className="robotic-learning-pathways-subtitle">
            Tailored learning paths to match your skill level and career goals
          </p>
        </div>

        {/* Learning Path Selection */}
        <div className="robotic-learning-path-grid">
          {learningPaths.map((path, index) => (
            <div
              key={index}
              onClick={() => setSelectedPath(index)}
              className={`robotic-learning-path-card ${selectedPath === index ? 'selected' : ''}`}
            >
              <div className="absolute inset-0 bg-gradient-to-br-cyan-purple opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none"></div>
              <div className="relative z-10">
                <div className={`robotic-path-icon text-4xl mb-6 bg-gradient-to-r ${path.color} bg-clip-text text-transparent`}>
                  {path.icon}
                </div>
                <div className="robotic-path-info">
                  <h3 className="robotic-path-title">
                    {path.title}
                  </h3>
                  <div className="robotic-path-meta">
                    <span className={`robotic-path-level px-3 py-1 rounded-full text-xs font-medium ${
                      path.level === 'Beginner' ? 'bg-gradient-to-r-green-teal-stats text-green-800 dark:bg-gradient-to-r-green-teal-stats dark:text-green-300' :
                      path.level === 'Intermediate' ? 'bg-gradient-to-r-blue-cyan-stats text-blue-800 dark:bg-gradient-to-r-blue-cyan-stats dark:text-blue-300' :
                      'bg-gradient-to-r-purple-pink-stats text-purple-800 dark:bg-gradient-to-r-purple-pink-stats dark:text-purple-300'
                    }`}>
                      {path.level}
                    </span>
                    <span className="robotic-path-duration ml-3 text-slate-500 dark:text-slate-400 text-sm">
                      {path.duration}
                    </span>
                  </div>
                </div>
              </div>
              <p className="robotic-path-description text-slate-600 dark:text-slate-400 text-sm mt-4">
                {path.description}
              </p>
            </div>
          ))}
        </div>

        {/* Selected Path Details */}
        <div className="robotic-selected-path-container bg-slate-800-60 backdrop-blur-xl rounded-3xl p-10 shadow-2xl border border-white-20 dark:border-slate-700-50 mb-16">
          <div className="robotic-selected-path-content">
            <div className="robotic-selected-path-header text-center mb-12">
              <div className={`robotic-selected-path-icon-large inline-block text-7xl mb-6 bg-gradient-to-r ${learningPaths[selectedPath].color} bg-clip-text text-transparent`}>
                {learningPaths[selectedPath].icon}
              </div>
              <h3 className="robotic-selected-path-title-large text-4xl font-bold text-slate-900 dark:text-white mb-4">
                {learningPaths[selectedPath].title}
              </h3>
              <p className="robotic-selected-path-description-large text-xl text-slate-600 dark:text-slate-300">
                {learningPaths[selectedPath].description}
              </p>
            </div>

            <div className="robotic-selected-path-details-grid grid grid-cols-1 md:grid-cols-2 gap-12">
              <div className="robotic-modules-section">
                <h4 className="robotic-section-title text-2xl font-bold text-slate-900 dark:text-white mb-6 relative inline-block">
                  Learning Modules
                  <div className="absolute bottom-0 left-0 w-12 h-1 bg-gradient-to-r-cyan-blue-stats"></div>
                </h4>
                <ul className="robotic-modules-list space-y-4">
                  {learningPaths[selectedPath].modules.map((module, index) => (
                    <li key={index} className="robotic-module-item flex items-start p-4 rounded-xl bg-slate-700-20 hover:bg-slate-700-30 transition-all duration-300 group">
                      <div className="w-3 h-3 bg-gradient-to-r-cyan-blue-stats rounded-full mt-2 mr-4 flex-shrink-0"></div>
                      <span className="robotic-module-text text-slate-700 dark:text-slate-300 group-hover:text-cyan-600 dark:group-hover:text-cyan-400 transition-colors duration-300">{module}</span>
                    </li>
                  ))}
                </ul>
              </div>
              <div className="robotic-achievements-section">
                <h4 className="robotic-section-title text-2xl font-bold text-slate-900 dark:text-white mb-6 relative inline-block">
                  What You'll Achieve
                  <div className="absolute bottom-0 left-0 w-12 h-1 bg-gradient-to-r-purple-pink-stats"></div>
                </h4>
                <ul className="robotic-achievements-list space-y-4">
                  <li className="robotic-achievement-item flex items-start p-4 rounded-xl bg-slate-700-20 hover:bg-slate-700-30 transition-all duration-300 group">
                    <div className="w-3 h-3 bg-gradient-to-r-green-teal-stats rounded-full mt-2 mr-4 flex-shrink-0"></div>
                    <span className="robotic-achievement-text text-slate-700 dark:text-slate-300 group-hover:text-green-600 dark:group-hover:text-green-400 transition-colors duration-300">Build complete robotic systems</span>
                  </li>
                  <li className="robotic-achievement-item flex items-start p-4 rounded-xl bg-slate-700-20 hover:bg-slate-700-30 transition-all duration-300 group">
                    <div className="w-3 h-3 bg-gradient-to-r-blue-cyan-stats rounded-full mt-2 mr-4 flex-shrink-0"></div>
                    <span className="robotic-achievement-text text-slate-700 dark:text-slate-300 group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors duration-300">Deploy AI models on real hardware</span>
                  </li>
                  <li className="robotic-achievement-item flex items-start p-4 rounded-xl bg-slate-700-20 hover:bg-slate-700-30 transition-all duration-300 group">
                    <div className="w-3 h-3 bg-gradient-to-r-purple-pink-stats rounded-full mt-2 mr-4 flex-shrink-0"></div>
                    <span className="robotic-achievement-text text-slate-700 dark:text-slate-300 group-hover:text-purple-600 dark:group-hover:text-purple-400 transition-colors duration-300">Create autonomous applications</span>
                  </li>
                </ul>
              </div>
            </div>

            <div className="text-center mt-12">
              <button className={`robotic-start-path-button px-10 py-4 bg-gradient-to-r ${learningPaths[selectedPath].color} text-white font-bold rounded-2xl text-lg hover:shadow-2xl hover:shadow-cyan-500/30 transition-all duration-300 transform hover:scale-105`}>
                Start Learning Path
              </button>
            </div>
          </div>
        </div>

        {/* Learning Resources */}
        <div className="robotic-resources-grid grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {resources.map((resource, index) => (
            <div
              key={index}
              className="robotic-resource-card bg-slate-800-40 backdrop-blur-lg rounded-2xl p-8 shadow-xl border border-white-20 dark:border-slate-700-50 text-center hover:-translate-y-2 transition-transform duration-300 group"
            >
              <div className="robotic-resource-icon text-5xl mb-6 group-hover:scale-110 transition-transform duration-300">{resource.icon}</div>
              <div className="robotic-resource-count text-4xl font-bold bg-gradient-to-r-cyan-blue-stats bg-clip-text text-transparent mb-4">
                {resource.count}
              </div>
              <h4 className="robotic-resource-title text-xl font-bold text-slate-900 dark:text-white mb-3 group-hover:text-cyan-400 transition-colors duration-300">
                {resource.title}
              </h4>
              <p className="robotic-resource-description text-slate-600 dark:text-slate-400">
                {resource.description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default LearningPathways;