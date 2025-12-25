import React from 'react';
import './component-styles.css';

const TechStack = () => {
  const technologies = [
    {
      category: "AI Frameworks",
      items: [
        { name: "TensorFlow", icon: "🤖", description: "ML framework for neural networks" },
        { name: "PyTorch", icon: "⚡", description: "Deep learning for research and production" },
        { name: "OpenVLA", icon: "👁️", description: "Vision-language-action models" }
      ],
      color: "from-blue-500 to-cyan-500"
    },
    {
      category: "Robotics Platforms",
      items: [
        { name: "ROS 2", icon: "⚙️", description: "Robot Operating System" },
        { name: "NVIDIA Isaac", icon: "🚀", description: "AI-powered robotics platform" },
        { name: "Gazebo", icon: "🎮", description: "3D simulation environment" }
      ],
      color: "from-purple-500 to-pink-500"
    },
    {
      category: "Development Tools",
      items: [
        { name: "Python", icon: "🐍", description: "Primary programming language" },
        { name: "C++", icon: "⚡", description: "Performance-critical systems" },
        { name: "Docker", icon: "🐳", description: "Containerization for deployment" }
      ],
      color: "from-green-500 to-emerald-500"
    }
  ];

  return (
    <section className="py-20 bg-slate-900">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-3xl md-text-4xl font-bold text-white mb-4">
            Essential <span className="bg-gradient-to-r-from-green-400-to-blue-400 bg-clip-text text-transparent"></span> Technology Stack
          </h2>
          <p className="text-lg text-slate-800 max-w-2xl mx-auto">
                       Industry-standard tools and frameworks used in modern humanoid robotics development
          </p>
        </div>

        <div className="grid grid-cols-1 md-grid-cols-3 gap-8">
          {technologies.map((techGroup, index) => (
            <div key={index} className="bg-slate-800-40 backdrop-blur-sm rounded-xl p-6 border border-slate-700-50">
              <h3 className="text-xl font-semibold text-white mb-6 flex-items-center">
                <span className={`bg-gradient-to-r ${techGroup.color} bg-clip-text text-transparent mr-2`}>
                  {techGroup.items[0].icon}
                </span>
                {techGroup.category}
              </h3>
              <div className="space-y-4">
                {techGroup.items.map((tech, techIndex) => (
                  <div key={techIndex} className="flex-items-center space-x-4 p-3 rounded-lg bg-slate-700-30 hover-bg-slate-700-50 transition-colors duration-300">
                    <div className="text-2xl">{tech.icon}</div>
                    <div className="flex-1">
                      <h4 className="font-medium text-white">{tech.name}</h4>
                      <p className="text-sm text-slate-400">{tech.description}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* Additional resources */}
        <div className="mt-16 bg-slate-800-30 backdrop-blur-sm rounded-2xl p-8 border border-slate-700-50">
          <h3 className="text-2xl font-bold text-white mb-6 text-center">Research & Development Resources</h3>
          <div className="grid grid-cols-1 md-grid-cols-4 gap-6">
            <div className="text-center">
              <div className="text-3xl mb-2">📚</div>
              <h4 className="font-semibold text-white mb-2">Research Papers</h4>
              <p className="text-sm text-slate-400">Curated collection of seminal works</p>
            </div>
            <div className="text-center">
              <div className="text-3xl mb-2">🧪</div>
              <h4 className="font-semibold text-white mb-2">Lab Exercises</h4>
              <p className="text-sm text-slate-400">Hands-on experiments and tests</p>
            </div>
            <div className="text-center">
              <div className="text-3xl mb-2">📊</div>
              <h4 className="font-semibold text-white mb-2">Datasets</h4>
              <p className="text-sm text-slate-400">Real-world robotics data</p>
            </div>
            <div className="text-center">
              <div className="text-3xl mb-2">🔧</div>
              <h4 className="font-semibold text-white mb-2">Tools</h4>
              <p className="text-sm text-slate-400">Development and debugging tools</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default TechStack;