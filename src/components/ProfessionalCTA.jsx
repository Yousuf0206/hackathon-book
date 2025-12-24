import React from 'react';

const ProfessionalCTA = () => {
  return (
    <section className="py-20 bg-gradient-to-b from-slate-900 to-slate-800">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
        <h2 className="text-3xl md:text-4xl font-bold text-white mb-6">
          Ready to <span className="bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent">Master</span> Physical AI?
        </h2>
        <p className="text-lg text-slate-300 mb-12 max-w-2xl mx-auto">
          Join thousands of engineers and researchers advancing the field of humanoid robotics
        </p>

        <div className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-16">
          <button className="px-8 py-3 bg-gradient-to-r from-cyan-600 to-blue-600 text-white font-medium rounded-lg hover:shadow-lg hover:shadow-cyan-500/25 transition-all duration-300">
            Start Learning
          </button>
          <button className="px-8 py-3 bg-slate-800/50 text-slate-300 font-medium rounded-lg border border-slate-700 hover:border-cyan-500 hover:text-white transition-all duration-300">
            View Curriculum
          </button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8 mb-12">
          <div>
            <div className="text-2xl font-bold text-cyan-400 mb-2">50+</div>
            <div className="text-slate-400 text-sm">Modules</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-purple-400 mb-2">200+</div>
            <div className="text-slate-400 text-sm">Projects</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-green-400 mb-2">10k+</div>
            <div className="text-slate-400 text-sm">Engineers</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-orange-400 mb-2">∞</div>
            <div className="text-slate-400 text-sm">Possibilities</div>
          </div>
        </div>

        {/* Trust indicators */}
        <div className="flex flex-wrap justify-center items-center gap-8 text-slate-400 text-sm">
          <div className="flex items-center">
            <div className="w-2 h-2 bg-green-500 rounded-full mr-2"></div>
            <span>Industry Certified</span>
          </div>
          <div className="flex items-center">
            <div className="w-2 h-2 bg-blue-500 rounded-full mr-2"></div>
            <span>Research-Backed</span>
          </div>
          <div className="flex items-center">
            <div className="w-2 h-2 bg-purple-500 rounded-full mr-2"></div>
            <span>Hands-On Learning</span>
          </div>
          <div className="flex items-center">
            <div className="w-2 h-2 bg-cyan-500 rounded-full mr-2"></div>
            <span>Expert Instructors</span>
          </div>
        </div>
      </div>
    </section>
  );
};

export default ProfessionalCTA;