import React from 'react';
import './component-styles.css';

const StatsShowcase = () => {
  const stats = [
    {
      number: "50+",
      label: "Learning Modules",
      description: "Comprehensive coverage of robotics and AI topics",
      icon: "📚",
      gradient: "from-cyan-500 to-blue-500"
    },
    {
      number: "200+",
      label: "Projects",
      description: "Hands-on projects to reinforce learning",
      icon: "🚀",
      gradient: "from-purple-500 to-pink-500"
    },
    {
      number: "10k+",
      label: "Engineers",
      description: "Professionals trained with our curriculum",
      icon: "👨‍💻",
      gradient: "from-blue-500 to-cyan-500"
    },
    {
      number: "∞",
      label: "Possibilities",
      description: "Limitless potential for innovation",
      icon: "⚡",
      gradient: "from-green-500 to-teal-500"
    }
  ];

  const achievements = [
    {
      title: "Industry Recognition",
      description: "Featured in top robotics conferences and journals",
      icon: "🏆"
    },
    {
      title: "Research Impact",
      description: "Cited in over 500 academic papers",
      icon: "🔬"
    },
    {
      title: "Career Outcomes",
      description: "95% of learners report career advancement",
      icon: "📈"
    },
    {
      title: "Global Reach",
      description: "Used by institutions in 50+ countries",
      icon: "🌍"
    }
  ];

  return (
    <section className="py-20 bg-gradient-to-b-slate-stats dark-bg-gradient-to-b-slate-stats">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Main Stats */}
        <div className="grid grid-cols-1 md-grid-cols-2 lg-grid-cols-4 gap-8 mb-16">
          {stats.map((stat, index) => (
            <div
              key={index}
              className="group relative bg-white-80 dark-bg-slate-800-60 backdrop-blur-lg rounded-2xl p-8 shadow-xl hover-shadow-2xl transition-all duration-500 border border-white-20 dark-border-slate-700-50 hover-border-cyan-400-30 dark-hover-border-cyan-500-30 overflow-hidden hover-translate-y-n2"
            >
              <div className="absolute inset-0 bg-gradient-to-br-cyan-purple opacity-0 group-hover-opacity-100 transition-opacity duration-500"></div>

              <div className="relative z-10 text-center">
                <div className="text-5xl md-text-6xl font-bold bg-gradient-to-r bg-clip-text text-transparent mb-4 group-hover-scale-110 transition-transform duration-300">
                  <span className={`bg-gradient-to-r ${stat.gradient} bg-clip-text text-transparent`}>
                    {stat.number}
                  </span>
                </div>
                <div className="text-4xl mb-4">{stat.icon}</div>
                <div className="text-xl font-bold text-slate-900 dark:text-white mb-2">
                  {stat.label}
                </div>
                <div className="text-slate-600 dark:text-slate-400 text-sm">
                  {stat.description}
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Achievements */}
        <div className="bg-white-80 dark-bg-slate-800-60 backdrop-blur-lg rounded-3xl p-10 shadow-2xl border border-white-20 dark-border-slate-700-50">
          <div className="text-center mb-12">
            <h3 className="text-3xl md-text-4xl font-bold text-slate-900 dark:text-white mb-4">
              Proven Impact & Recognition
            </h3>
            <p className="text-xl text-slate-600 dark:text-slate-300">
              Our curriculum has made a significant impact in the robotics community
            </p>
          </div>

          <div className="grid grid-cols-1 md-grid-cols-2 lg-grid-cols-4 gap-8">
            {achievements.map((achievement, index) => (
              <div
                key={index}
                className="text-center group"
              >
                <div className="text-4xl mb-4 group-hover-scale-110 transition-transform duration-300">
                  {achievement.icon}
                </div>
                <h4 className="text-lg font-bold text-slate-900 dark:text-white mb-2 group-hover-text-cyan-600 dark-group-hover-text-cyan-400 transition-colors duration-300">
                  {achievement.title}
                </h4>
                <p className="text-slate-600 dark:text-slate-400 text-sm">
                  {achievement.description}
                </p>
              </div>
            ))}
          </div>
        </div>

        {/* CTA Section */}
        <div className="text-center mt-16">
          <div className="inline-block bg-gradient-to-r-cyan-blue-stats p-1 rounded-2xl">
            <div className="bg-white dark:bg-slate-900 rounded-xl px-8 py-6">
              <h4 className="text-2xl font-bold text-slate-900 dark:text-white mb-2">
                Ready to Join the Next Generation?
              </h4>
              <p className="text-slate-600 dark:text-slate-300 mb-4">
                Start your journey in physical AI and humanoid robotics today
              </p>
              <button className="px-8 py-3 bg-gradient-to-r-cyan-blue-stats text-white font-semibold rounded-xl hover-shadow-lg transition-shadow duration-300">
                Begin Your Learning Journey
              </button>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default StatsShowcase;