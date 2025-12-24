import React from 'react';

const ModernFooter = () => {
  return (
    <footer className="bg-slate-900 text-slate-300 py-16">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-12">
          {/* Brand */}
          <div>
            <div className="text-2xl font-bold text-white mb-4">
              Physical AI & Humanoid Robotics
            </div>
            <p className="text-slate-400 mb-6">
              The comprehensive guide to mastering artificial intelligence and humanoid robotics.
            </p>
            <div className="flex space-x-4">
              <a href="#" className="w-10 h-10 bg-slate-800 rounded-full flex items-center justify-center hover:bg-cyan-500 transition-colors duration-300">
                <span className="text-sm">f</span>
              </a>
              <a href="#" className="w-10 h-10 bg-slate-800 rounded-full flex items-center justify-center hover:bg-cyan-500 transition-colors duration-300">
                <span className="text-sm">t</span>
              </a>
              <a href="#" className="w-10 h-10 bg-slate-800 rounded-full flex items-center justify-center hover:bg-cyan-500 transition-colors duration-300">
                <span className="text-sm">in</span>
              </a>
              <a href="#" className="w-10 h-10 bg-slate-800 rounded-full flex items-center justify-center hover:bg-cyan-500 transition-colors duration-300">
                <span className="text-sm">gh</span>
              </a>
            </div>
          </div>

          {/* Quick Links */}
          <div>
            <h3 className="text-white font-semibold mb-6 text-lg">Quick Links</h3>
            <ul className="space-y-3">
              <li><a href="#" className="text-slate-400 hover:text-cyan-400 transition-colors duration-300">Home</a></li>
              <li><a href="#" className="text-slate-400 hover:text-cyan-400 transition-colors duration-300">Modules</a></li>
              <li><a href="#" className="text-slate-400 hover:text-cyan-400 transition-colors duration-300">Projects</a></li>
              <li><a href="#" className="text-slate-400 hover:text-cyan-400 transition-colors duration-300">Resources</a></li>
              <li><a href="#" className="text-slate-400 hover:text-cyan-400 transition-colors duration-300">Community</a></li>
            </ul>
          </div>

          {/* Learning Paths */}
          <div>
            <h3 className="text-white font-semibold mb-6 text-lg">Learning Paths</h3>
            <ul className="space-y-3">
              <li><a href="#" className="text-slate-400 hover:text-cyan-400 transition-colors duration-300">Beginner's Path</a></li>
              <li><a href="#" className="text-slate-400 hover:text-cyan-400 transition-colors duration-300">Developer's Path</a></li>
              <li><a href="#" className="text-slate-400 hover:text-cyan-400 transition-colors duration-300">Expert's Path</a></li>
              <li><a href="#" className="text-slate-400 hover:text-cyan-400 transition-colors duration-300">Certification</a></li>
              <li><a href="#" className="text-slate-400 hover:text-cyan-400 transition-colors duration-300">Career Services</a></li>
            </ul>
          </div>

          {/* Contact */}
          <div>
            <h3 className="text-white font-semibold mb-6 text-lg">Contact Us</h3>
            <ul className="space-y-3">
              <li className="flex items-start">
                <svg className="w-5 h-5 text-cyan-400 mr-3 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 4.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                </svg>
                <span className="text-slate-400">contact@physicalai-book.com</span>
              </li>
              <li className="flex items-start">
                <svg className="w-5 h-5 text-cyan-400 mr-3 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
                <span className="text-slate-400">San Francisco, CA</span>
              </li>
              <li className="flex items-start">
                <svg className="w-5 h-5 text-cyan-400 mr-3 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z" />
                </svg>
                <span className="text-slate-400">+1 (555) 123-4567</span>
              </li>
            </ul>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="border-t border-slate-800 mt-12 pt-8 flex flex-col md:flex-row justify-between items-center">
          <div className="text-slate-400 text-sm mb-4 md:mb-0">
            © 2025 Physical AI & Humanoid Robotics. All rights reserved.
          </div>
          <div className="flex space-x-6 text-sm">
            <a href="#" className="text-slate-400 hover:text-cyan-400 transition-colors duration-300">Privacy Policy</a>
            <a href="#" className="text-slate-400 hover:text-cyan-400 transition-colors duration-300">Terms of Service</a>
            <a href="#" className="text-slate-400 hover:text-cyan-400 transition-colors duration-300">Cookie Policy</a>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default ModernFooter;