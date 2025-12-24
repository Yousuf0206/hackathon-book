import React, { useState } from 'react';

const CoreConcepts = () => {
  const [hoveredCard, setHoveredCard] = useState(null);

  const concepts = [
    {
      title: "Physical AI Fundamentals",
      description: "Understanding the principles of AI that operate in physical spaces with real-world constraints",
      icon: "🧠",
      gradient: "from-blue-500 to-cyan-500"
    },
    {
      title: "Humanoid Locomotion",
      description: "Advanced techniques for bipedal walking, balance, and movement control systems",
      icon: "🦾",
      gradient: "from-purple-500 to-pink-500"
    },
    {
      title: "Sensor Integration",
      description: "Fusing data from multiple sensors for environmental awareness and navigation",
      icon: "📡",
      gradient: "from-cyan-500 to-blue-500"
    },
    {
      title: "Neural Control Systems",
      description: "Implementing neural networks for real-time robotic control and decision making",
      icon: "⚡",
      gradient: "from-green-500 to-emerald-500"
    },
    {
      title: "Human-Robot Interaction",
      description: "Designing intuitive interfaces for seamless collaboration between humans and robots",
      icon: "🤝",
      gradient: "from-orange-500 to-red-500"
    },
    {
      title: "Adaptive Learning",
      description: "Enabling robots to learn and adapt to new situations and environments",
      icon: "🔄",
      gradient: "from-pink-500 to-purple-500"
    }
  ];

  return (
    <section className="py-20 bg-slate-900">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
            Core <span className="bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">Concepts</span>
          </h2>
          <p className="text-lg text-slate-400 max-w-2xl mx-auto">
            Foundational knowledge areas that form the backbone of humanoid robotics education
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {concepts.map((concept, index) => (
            <div
              key={index}
              className={`group relative bg-slate-800/30 backdrop-blur-sm rounded-xl p-6 border border-slate-700/50 transition-all duration-300 cursor-pointer ${
                hoveredCard === index
                  ? 'bg-slate-800/50 border-blue-500/50 scale-105'
                  : 'hover:bg-slate-800/40 hover:border-slate-600'
              }`}
              onMouseEnter={() => setHoveredCard(index)}
              onMouseLeave={() => setHoveredCard(null)}
            >
              <div className="flex items-start space-x-4">
                <div className={`w-12 h-12 rounded-lg flex items-center justify-center text-xl bg-gradient-to-r ${concept.gradient} bg-clip-text text-transparent`}>
                  {concept.icon}
                </div>
                <div className="flex-1">
                  <h3 className="text-lg font-semibold text-white mb-2 group-hover:text-blue-400 transition-colors duration-300">
                    {concept.title}
                  </h3>
                  <p className="text-slate-400 text-sm leading-relaxed">
                    {concept.description}
                  </p>
                </div>
              </div>

              {/* Subtle hover indicator */}
              <div className={`absolute bottom-0 left-0 h-0.5 bg-gradient-to-r ${concept.gradient} transition-all duration-300 ${
                hoveredCard === index ? 'w-full' : 'w-0'
              }`}></div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default CoreConcepts;