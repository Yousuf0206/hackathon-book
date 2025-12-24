import React, { useState } from 'react';

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
    <section className="py-20 bg-gradient-to-b from-slate-100 to-slate-200 dark:from-slate-800 dark:to-slate-900">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold text-slate-900 dark:text-white mb-4">
            Choose Your Learning Path
          </h2>
          <p className="text-xl text-slate-600 dark:text-slate-300 max-w-3xl mx-auto">
            Tailored learning paths to match your skill level and career goals
          </p>
        </div>

        {/* Learning Path Selection */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-16">
          {learningPaths.map((path, index) => (
            <div
              key={index}
              onClick={() => setSelectedPath(index)}
              className={`cursor-pointer p-6 rounded-2xl border-2 transition-all duration-300 ${
                selectedPath === index
                  ? 'border-cyan-500 bg-white/80 dark:bg-slate-800/60 shadow-xl'
                  : 'border-transparent bg-white/60 dark:bg-slate-800/40 hover:bg-white/80 dark:hover:bg-slate-800/60'
              }`}
            >
              <div className="flex items-center mb-4">
                <div className={`text-3xl mr-4 bg-gradient-to-r ${path.color} bg-clip-text text-transparent`}>
                  {path.icon}
                </div>
                <div>
                  <h3 className="text-xl font-bold text-slate-900 dark:text-white">
                    {path.title}
                  </h3>
                  <div className="flex items-center">
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                      path.level === 'Beginner' ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300' :
                      path.level === 'Intermediate' ? 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300' :
                      'bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300'
                    }`}>
                      {path.level}
                    </span>
                    <span className="ml-2 text-slate-500 dark:text-slate-400 text-sm">
                      {path.duration}
                    </span>
                  </div>
                </div>
              </div>
              <p className="text-slate-600 dark:text-slate-400 text-sm">
                {path.description}
              </p>
            </div>
          ))}
        </div>

        {/* Selected Path Details */}
        <div className="bg-white/80 dark:bg-slate-800/60 backdrop-blur-lg rounded-3xl p-10 shadow-2xl border border-white/20 dark:border-slate-700/50 mb-16">
          <div className="text-center mb-8">
            <div className={`inline-block text-6xl mb-4 bg-gradient-to-r ${learningPaths[selectedPath].color} bg-clip-text text-transparent`}>
              {learningPaths[selectedPath].icon}
            </div>
            <h3 className="text-3xl font-bold text-slate-900 dark:text-white mb-2">
              {learningPaths[selectedPath].title}
            </h3>
            <p className="text-xl text-slate-600 dark:text-slate-300">
              {learningPaths[selectedPath].description}
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div>
              <h4 className="text-xl font-bold text-slate-900 dark:text-white mb-4">
                Learning Modules
              </h4>
              <ul className="space-y-3">
                {learningPaths[selectedPath].modules.map((module, index) => (
                  <li key={index} className="flex items-center">
                    <div className="w-2 h-2 bg-cyan-500 rounded-full mr-3"></div>
                    <span className="text-slate-700 dark:text-slate-300">{module}</span>
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <h4 className="text-xl font-bold text-slate-900 dark:text-white mb-4">
                What You'll Achieve
              </h4>
              <div className="space-y-3">
                <div className="flex items-start">
                  <div className="w-2 h-2 bg-green-500 rounded-full mt-2 mr-3"></div>
                  <span className="text-slate-700 dark:text-slate-300">Build complete robotic systems</span>
                </div>
                <div className="flex items-start">
                  <div className="w-2 h-2 bg-blue-500 rounded-full mt-2 mr-3"></div>
                  <span className="text-slate-700 dark:text-slate-300">Deploy AI models on real hardware</span>
                </div>
                <div className="flex items-start">
                  <div className="w-2 h-2 bg-purple-500 rounded-full mt-2 mr-3"></div>
                  <span className="text-slate-700 dark:text-slate-300">Create autonomous applications</span>
                </div>
              </div>
            </div>
          </div>

          <div className="text-center mt-8">
            <button className={`px-8 py-4 bg-gradient-to-r ${learningPaths[selectedPath].color} text-white font-semibold rounded-xl hover:shadow-lg transition-shadow duration-300`}>
              Start Learning Path
            </button>
          </div>
        </div>

        {/* Learning Resources */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {resources.map((resource, index) => (
            <div
              key={index}
              className="bg-white/80 dark:bg-slate-800/60 backdrop-blur-lg rounded-2xl p-6 shadow-xl border border-white/20 dark:border-slate-700/50 text-center hover:-translate-y-2 transition-transform duration-300"
            >
              <div className="text-4xl mb-4">{resource.icon}</div>
              <div className="text-2xl font-bold bg-gradient-to-r from-cyan-500 to-blue-500 bg-clip-text text-transparent mb-2">
                {resource.count}
              </div>
              <h4 className="text-lg font-bold text-slate-900 dark:text-white mb-2">
                {resource.title}
              </h4>
              <p className="text-slate-600 dark:text-slate-400 text-sm">
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