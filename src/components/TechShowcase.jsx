import React from 'react';
import './component-styles.css';

const TechShowcase = () => {
  const technologies = [
    {
      title: "NVIDIA Isaac",
      description: "Advanced robotics platform for AI-powered humanoid systems",
      image: "/img/isaac-platform.webp",
      gradient: "from-green-500 to-emerald-500"
    },
    {
      title: "ROS 2",
      description: "Robot Operating System for building sophisticated robotic applications",
      image: "/img/ros2-logo.webp",
      gradient: "from-blue-500 to-cyan-500"
    },
    {
      title: "OpenVLA",
      description: "Vision-language-action models for intelligent robot manipulation",
      image: "/img/openvla-model.webp",
      gradient: "from-purple-500 to-pink-500"
    },
    {
      title: "Gazebo",
      description: "Advanced 3D simulation environment for robot testing and development",
      image: "/img/gazebo-sim.webp",
      gradient: "from-orange-500 to-red-500"
    }
  ];

  const applications = [
    {
      icon: "🤖",
      title: "Industrial Automation",
      description: "Humanoid robots for manufacturing and industrial applications"
    },
    {
      icon: "🏥",
      title: "Healthcare Assistance",
      description: "Robotic companions for elderly care and medical support"
    },
    {
      icon: "🏠",
      title: "Domestic Helpers",
      description: "AI-powered robots for household management and assistance"
    },
    {
      icon: "🎓",
      title: "Educational Tools",
      description: "Interactive robots for STEM education and learning"
    }
  ];

  return (
    <section className="py-20 bg-gradient-to-b-slate-alt">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-4xl md-text-5xl font-bold text-white mb-4">
            <span className="bg-gradient-to-r-purple-cyan bg-clip-text text-transparent">Technology</span> Stack
          </h2>
          <p className="text-xl text-slate-300 max-w-3xl mx-auto">
            Learn with industry-standard tools and frameworks used in cutting-edge robotics
          </p>
        </div>

        <div className="grid grid-cols-1 md-grid-cols-2 lg-grid-cols-4 gap-6 mb-20">
          {technologies.map((tech, index) => (
            <div
              key={index}
              className="group relative bg-gradient-to-br-slate backdrop-blur-lg rounded-2xl p-6 border border-slate-700-50 hover-border-purple-500-50 transition-all duration-500 hover-translate-y-n2 hover-shadow-xl hover-shadow-purple-500-10 overflow-hidden"
            >
              <div className="relative mb-4">
                <div className={`w-full h-32 bg-gradient-to-r-${tech.gradient} rounded-xl flex-items-center mb-4`}>
                  <div className="w-16 h-16 bg-white-20 rounded-lg flex-items-center">
                    <span className="text-2xl">⚙️</span>
                  </div>
                </div>
              </div>
              <h3 className="text-lg font-bold text-white mb-2 group-hover-text-purple-400 transition-colors duration-300">
                {tech.title}
              </h3>
              <p className="text-slate-400 text-sm">
                {tech.description}
              </p>

              <div className="absolute inset-0 bg-gradient-to-r-purple-cyan-alt rounded-2xl opacity-0 group-hover-opacity-100 transition-opacity duration-500 pointer-events-none"></div>
            </div>
          ))}
        </div>

        <div className="bg-gradient-to-r-slate backdrop-blur-lg rounded-3xl p-10 border border-slate-700-50 mb-16">
          <div className="text-center mb-12">
            <h3 className="text-3xl font-bold text-white mb-4">
              Real-World <span className="bg-gradient-to-r-cyan-blue-text bg-clip-text text-transparent">Applications</span>
            </h3>
            <p className="text-slate-300 max-w-2xl mx-auto">
              Discover how humanoid robots are transforming industries and daily life
            </p>
          </div>

          <div className="grid grid-cols-1 md-grid-cols-2 lg-grid-cols-4 gap-8">
            {applications.map((app, index) => (
              <div
                key={index}
                className="text-center group"
              >
                <div className="text-4xl mb-4 group-hover-scale-110 transition-transform duration-300">
                  {app.icon}
                </div>
                <h4 className="text-lg font-bold text-white mb-2 group-hover-text-cyan-400 transition-colors duration-300">
                  {app.title}
                </h4>
                <p className="text-slate-400 text-sm">
                  {app.description}
                </p>
              </div>
            ))}
          </div>
        </div>

        {/* Interactive timeline */}
        <div className="relative">
          <div className="absolute-left-1-2 transform-n-translate-x-1-2 w-1 h-full bg-gradient-to-b-cyan-purple rounded-full"></div>

          <div className="space-y-12">
            {[
              { year: "2024", title: "Foundation", desc: "Physical AI fundamentals and robotics basics" },
              { year: "2025", title: "Advanced Systems", desc: "Humanoid locomotion and control systems" },
              { year: "2026", title: "AI Integration", desc: "Neural networks and machine learning for robots" },
              { year: "2027", title: "Deployment", desc: "Real-world applications and deployment" }
            ].map((item, index) => (
              <div key={index} className={`flex-items-center ${index % 2 === 0 ? 'flex-row' : 'flex-row-reverse'}`}>
                <div className={`w-5-12 ${index % 2 === 0 ? 'pr-8 text-right' : 'pl-8 text-left'}`}>
                  <div className="bg-gradient-to-r-slate-alt backdrop-blur-lg rounded-2xl p-6 border border-slate-700-50">
                    <div className="text-2xl font-bold text-cyan-400 mb-2">{item.year}</div>
                    <h4 className="text-lg font-bold text-white mb-2">{item.title}</h4>
                    <p className="text-slate-400">{item.desc}</p>
                  </div>
                </div>
                <div className="w-2-12 flex justify-center">
                  <div className="w-4 h-4 bg-cyan-500 rounded-full border-slate-900 z-10"></div>
                </div>
                <div className="w-5-12"></div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

export default TechShowcase;