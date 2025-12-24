import React from 'react';

const RoboticFeatures = () => {
  const features = [
    {
      icon: (
        <div className="w-16 h-16 bg-gradient-to-r from-cyan-500 to-blue-500 rounded-2xl flex items-center justify-center">
          <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
          </svg>
        </div>
      ),
      title: "Physical AI Fundamentals",
      description: "Deep dive into the principles of physical AI that power next-generation humanoid robots",
      gradient: "from-cyan-500 to-blue-500"
    },
    {
      icon: (
        <div className="w-16 h-16 bg-gradient-to-r from-purple-500 to-pink-500 rounded-2xl flex items-center justify-center">
          <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
        </div>
      ),
      title: "Humanoid Locomotion",
      description: "Master the complex mechanics of walking, balance, and movement in humanoid systems",
      gradient: "from-purple-500 to-pink-500"
    },
    {
      icon: (
        <div className="w-16 h-16 bg-gradient-to-r from-green-500 to-teal-500 rounded-2xl flex items-center justify-center">
          <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 014.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3M14.25 3.104c.251.023.501.05.75.082M19.8 15.3l-1.57.393A9.065 9.065 0 0112 15a9.065 9.065 0 00-6.23-.693L5 14.5m14.8.8l1.402 1.402c.5.5.798 1.19.798 1.905v1.986a2.25 2.25 0 01-2.25 2.25H2.25A2.25 2.25 0 010 20.25v-1.986c0-.715.298-1.405.798-1.905L2.25 15" />
          </svg>
        </div>
      ),
      title: "AI-Human Interaction",
      description: "Learn how to design intuitive interfaces for seamless human-robot collaboration",
      gradient: "from-green-500 to-teal-500"
    },
    {
      icon: (
        <div className="w-16 h-16 bg-gradient-to-r from-orange-500 to-red-500 rounded-2xl flex items-center justify-center">
          <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 18v-5.25m0 0a6.01 6.01 0 001.5-.189m-1.5.189a6.01 6.01 0 01-1.5-.189m0 0a6.01 6.01 0 00-1.5.189m1.5-.189v5.25m0-6V3.75a3 3 0 10-6 0v14.25a3 3 0 006 0z" />
          </svg>
        </div>
      ),
      title: "Neural Networks",
      description: "Build and deploy neural networks specifically optimized for robotic applications",
      gradient: "from-orange-500 to-red-500"
    },
    {
      icon: (
        <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-2xl flex items-center justify-center">
          <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.288 15.038a5.25 5.25 0 017.424 0M5.106 11.856c3.807-3.808 9.98-3.808 13.788 0M1.924 8.674a9.09 9.09 0 0113.788 0m-13.788 0L1.924 8.674m0 0L8.288 15.038M1.924 8.674L1.924 21l6.364-6.364" />
          </svg>
        </div>
      ),
      title: "Real-time Control",
      description: "Implement real-time control systems for responsive and adaptive robot behavior",
      gradient: "from-blue-500 to-cyan-500"
    },
    {
      icon: (
        <div className="w-16 h-16 bg-gradient-to-r from-pink-500 to-purple-500 rounded-2xl flex items-center justify-center">
          <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.042 21.672L13.684 16.6m0 0l-2.51 2.225.569-9.47 5.227 7.917-3.286-.672ZM12 2.25V4.5m5.834.166l-1.591 1.591M20.25 10.5H18M7.757 14.743l-1.59 1.59M6 10.5H3.75m4.007-4.243l-1.59-1.59" />
          </svg>
        </div>
      ),
      title: "Hardware Integration",
      description: "Seamlessly integrate AI algorithms with physical robotic hardware components",
      gradient: "from-pink-500 to-purple-500"
    }
  ];

  return (
    <section className="py-20 bg-slate-900">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
            Core <span className="bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">Robotic</span> Concepts
          </h2>
          <p className="text-xl text-slate-300 max-w-3xl mx-auto">
            Master the fundamental technologies that power next-generation humanoid robots
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <div
              key={index}
              className="robotic-feature-card group relative p-8 hover:-translate-y-2 hover:shadow-2xl hover:shadow-cyan-500/10"
            >
              <div className="mb-6">
                {feature.icon}
              </div>
              <h3 className="text-xl font-bold text-white mb-4 group-hover:text-cyan-400 transition-colors duration-300">
                {feature.title}
              </h3>
              <p className="text-slate-400 leading-relaxed">
                {feature.description}
              </p>

              {/* Hover effect element */}
              <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/5 to-blue-500/5 rounded-3xl opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none"></div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default RoboticFeatures;