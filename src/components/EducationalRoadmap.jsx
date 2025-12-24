import React from 'react';

const EducationalRoadmap = () => {
  const roadmap = [
    {
      phase: "Foundation",
      title: "Core Principles",
      duration: "Weeks 1-4",
      description: "Build fundamental understanding of robotics and AI concepts",
      topics: [
        "Introduction to Physical AI",
        "Mathematics for Robotics",
        "Basic Programming Concepts",
        "Robotics Hardware Overview"
      ],
      color: "from-blue-500 to-cyan-500"
    },
    {
      phase: "Specialization",
      title: "Humanoid Systems",
      duration: "Weeks 5-12",
      description: "Deep dive into humanoid-specific technologies and challenges",
      topics: [
        "Humanoid Kinematics",
        "Sensor Fusion",
        "Control Systems",
        "Locomotion Algorithms",
        "Machine Learning for Robotics"
      ],
      color: "from-purple-500 to-pink-500"
    },
    {
      phase: "Integration",
      title: "AI-Hardware Interface",
      duration: "Weeks 13-20",
      description: "Connect AI algorithms with physical robotic systems",
      topics: [
        "Neural Network Implementation",
        "Real-time Control Systems",
        "Human-Robot Interaction",
        "Adaptive Learning Systems"
      ],
      color: "from-green-500 to-emerald-500"
    },
    {
      phase: "Mastery",
      title: "Research & Innovation",
      duration: "Weeks 21-24",
      description: "Apply knowledge to cutting-edge research projects",
      topics: [
        "Advanced Research Methods",
        "Innovation Projects",
        "Thesis Development",
        "Industry Applications"
      ],
      color: "from-orange-500 to-red-500"
    }
  ];

  return (
    <section className="py-20 bg-slate-900">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
            Learning <span className="bg-gradient-to-r from-orange-400 to-red-400 bg-clip-text text-transparent">Roadmap</span>
          </h2>
          <p className="text-lg text-slate-400 max-w-2xl mx-auto">
            A structured path from fundamentals to advanced research in humanoid robotics
          </p>
        </div>

        <div className="relative">
          {/* Timeline line */}
          <div className="absolute left-1/2 transform -translate-x-1/2 w-1 h-full bg-gradient-to-b from-blue-500 via-purple-500 to-green-500 rounded-full hidden md:block"></div>

          <div className="space-y-12">
            {roadmap.map((phase, index) => (
              <div key={index} className={`flex flex-col ${index % 2 === 0 ? 'md:flex-row' : 'md:flex-row-reverse'} items-center`}>
                <div className={`w-full md:w-5/12 ${index % 2 === 0 ? 'md:pr-12 md:text-right' : 'md:pl-12 md:text-left'} mb-6 md:mb-0`}>
                  <div className="bg-slate-800/40 backdrop-blur-sm rounded-xl p-6 border border-slate-700/50">
                    <div className="text-sm font-medium bg-gradient-to-r from-slate-600 to-slate-700 text-slate-300 px-3 py-1 rounded-full inline-block mb-3">
                      {phase.duration}
                    </div>
                    <h3 className="text-xl font-bold text-white mb-2">
                      <span className={`bg-gradient-to-r ${phase.color} bg-clip-text text-transparent`}>
                        {phase.phase}
                      </span>
                    </h3>
                    <h4 className="text-lg font-semibold text-slate-200 mb-3">{phase.title}</h4>
                    <p className="text-slate-400 mb-4">{phase.description}</p>
                    <ul className="space-y-2">
                      {phase.topics.map((topic, topicIndex) => (
                        <li key={topicIndex} className="flex items-center text-sm text-slate-300">
                          <div className="w-1.5 h-1.5 bg-cyan-500 rounded-full mr-3"></div>
                          {topic}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>

                {/* Timeline marker */}
                <div className="w-full md:w-2/12 flex justify-center">
                  <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full border-4 border-slate-900 flex items-center justify-center z-10">
                    <span className="text-white font-bold text-sm">{index + 1}</span>
                  </div>
                </div>

                <div className="w-full md:w-5/12"></div>
              </div>
            ))}
          </div>
        </div>

        {/* Call to action */}
        <div className="text-center mt-16">
          <div className="inline-block bg-gradient-to-r from-blue-600/20 to-purple-600/20 rounded-2xl p-1">
            <div className="bg-slate-800/50 rounded-xl p-8">
              <h3 className="text-2xl font-bold text-white mb-4">Ready to Begin Your Journey?</h3>
              <p className="text-slate-300 mb-6 max-w-md mx-auto">
                Start with the foundational concepts and progress through our structured learning roadmap
              </p>
              <button className="px-8 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-medium rounded-lg hover:shadow-lg transition-shadow duration-300">
                Start Learning Today
              </button>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default EducationalRoadmap;