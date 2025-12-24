import React, { useState } from 'react';

const EducationalPaths = () => {
  const [selectedPath, setSelectedPath] = useState(0);

  const learningPaths = [
    {
      title: "Foundations",
      description: "Start with core principles of robotics and AI",
      duration: "4-6 weeks",
      level: "Beginner",
      modules: [
        "Introduction to Robotics",
        "Basic AI Concepts",
        "Mathematics for Robotics",
        "Programming Fundamentals"
      ],
      color: "from-blue-500 to-cyan-500",
      icon: "🌱"
    },
    {
      title: "Specialization",
      description: "Deep dive into humanoid-specific technologies",
      duration: "8-12 weeks",
      level: "Intermediate",
      modules: [
        "Humanoid Kinematics",
        "Sensor Fusion",
        "Control Systems",
        "Machine Learning for Robotics"
      ],
      color: "from-purple-500 to-pink-500",
      icon: "🎯"
    },
    {
      title: "Mastery",
      description: "Advanced concepts and research applications",
      duration: "12-16 weeks",
      level: "Advanced",
      modules: [
        "Neural Networks",
        "Human-Robot Interaction",
        "Adaptive Systems",
        "Research Project"
      ],
      color: "from-green-500 to-emerald-500",
      icon: "🚀"
    }
  ];

  return (
    <section className="py-20 bg-gradient-to-b from-slate-900 to-slate-800">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
            Structured <span className="bg-gradient-to-r from-purple-400 to-cyan-400 bg-clip-text text-transparent">Learning</span> Paths
          </h2>
          <p className="text-lg text-slate-400 max-w-2xl mx-auto">
            Progress through carefully designed educational tracks that build expertise systematically
          </p>
        </div>

        {/* Path Selection */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-12">
          {learningPaths.map((path, index) => (
            <div
              key={index}
              onClick={() => setSelectedPath(index)}
              className={`cursor-pointer p-4 rounded-lg border-2 transition-all duration-300 text-center ${
                selectedPath === index
                  ? 'border-purple-500 bg-slate-800/50 shadow-lg shadow-purple-500/10'
                  : 'border-slate-700 bg-slate-800/30 hover:border-slate-600'
              }`}
            >
              <div className={`text-2xl mb-2 bg-gradient-to-r ${path.color} bg-clip-text text-transparent`}>
                {path.icon}
              </div>
              <h3 className="font-semibold text-white mb-1">{path.title}</h3>
              <p className="text-sm text-slate-400">{path.duration}</p>
            </div>
          ))}
        </div>

        {/* Selected Path Details */}
        <div className="bg-slate-800/40 backdrop-blur-sm rounded-2xl p-8 border border-slate-700/50">
          <div className="text-center mb-8">
            <div className={`inline-block text-4xl mb-4 bg-gradient-to-r ${learningPaths[selectedPath].color} bg-clip-text text-transparent`}>
              {learningPaths[selectedPath].icon}
            </div>
            <h3 className="text-2xl font-bold text-white mb-2">
              {learningPaths[selectedPath].title} Path
            </h3>
            <p className="text-slate-300 mb-4">
              {learningPaths[selectedPath].description}
            </p>
            <div className="inline-block px-3 py-1 bg-slate-700/50 rounded-full text-sm text-slate-300">
              {learningPaths[selectedPath].level} • {learningPaths[selectedPath].duration}
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div>
              <h4 className="text-lg font-semibold text-white mb-4">Learning Modules</h4>
              <ul className="space-y-2">
                {learningPaths[selectedPath].modules.map((module, index) => (
                  <li key={index} className="flex items-center text-slate-300 text-sm">
                    <div className="w-1.5 h-1.5 bg-purple-500 rounded-full mr-3"></div>
                    {module}
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <h4 className="text-lg font-semibold text-white mb-4">What You'll Achieve</h4>
              <ul className="space-y-2">
                <li className="flex items-start text-slate-300 text-sm">
                  <div className="w-1.5 h-1.5 bg-cyan-500 rounded-full mt-1.5 mr-3"></div>
                  Comprehensive understanding of {learningPaths[selectedPath].title.toLowerCase()} concepts
                </li>
                <li className="flex items-start text-slate-300 text-sm">
                  <div className="w-1.5 h-1.5 bg-green-500 rounded-full mt-1.5 mr-3"></div>
                  Hands-on experience with practical implementations
                </li>
                <li className="flex items-start text-slate-300 text-sm">
                  <div className="w-1.5 h-1.5 bg-purple-500 rounded-full mt-1.5 mr-3"></div>
                  Portfolio of projects demonstrating your skills
                </li>
              </ul>
            </div>
          </div>

          <div className="text-center mt-8">
            <button className={`px-6 py-3 bg-gradient-to-r ${learningPaths[selectedPath].color} text-white font-medium rounded-lg hover:shadow-lg transition-shadow duration-300`}>
              Start Learning Path
            </button>
          </div>
        </div>
      </div>
    </section>
  );
};

export default EducationalPaths;