import React from 'react';

const HomepageFooter = () => {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="homepage-footer bg-slate-900 text-slate-300 py-16">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-12">
          <div className="footer-section">
            <h3 className="footer-title text-xl font-bold text-white mb-6">AI-Human Intelligence</h3>
            <p className="footer-description text-slate-400">
              Bridging artificial intelligence with human expertise in robotics and autonomous systems.
              Advancing the future of collaborative intelligence.
            </p>
          </div>

          <div className="footer-section">
            <h3 className="footer-title text-xl font-bold text-white mb-6">Resources</h3>
            <ul className="footer-links space-y-3">
              <li><a href="/docs/intro" className="footer-link text-slate-400 hover:text-cyan-400 transition-colors">Documentation</a></li>
              <li><a href="/docs/category/getting-started" className="footer-link text-slate-400 hover:text-cyan-400 transition-colors">Getting Started</a></li>
              <li><a href="/docs/category/advanced-topics" className="footer-link text-slate-400 hover:text-cyan-400 transition-colors">Advanced Topics</a></li>
              <li><a href="/blog" className="footer-link text-slate-400 hover:text-cyan-400 transition-colors">Blog</a></li>
            </ul>
          </div>

          <div className="footer-section">
            <h3 className="footer-title text-xl font-bold text-white mb-6">Community</h3>
            <ul className="footer-links space-y-3">
              <li><a href="#" className="footer-link text-slate-400 hover:text-cyan-400 transition-colors">GitHub</a></li>
              <li><a href="#" className="footer-link text-slate-400 hover:text-cyan-400 transition-colors">Discord</a></li>
              <li><a href="#" className="footer-link text-slate-400 hover:text-cyan-400 transition-colors">Forums</a></li>
              <li><a href="#" className="footer-link text-slate-400 hover:text-cyan-400 transition-colors">Events</a></li>
            </ul>
          </div>

          <div className="footer-section">
            <h3 className="footer-title text-xl font-bold text-white mb-6">Connect</h3>
            <ul className="footer-links space-y-3">
              <li><a href="#" className="footer-link text-slate-400 hover:text-cyan-400 transition-colors">Contact Us</a></li>
              <li><a href="#" className="footer-link text-slate-400 hover:text-cyan-400 transition-colors">Newsletter</a></li>
              <li><a href="#" className="footer-link text-slate-400 hover:text-cyan-400 transition-colors">Partners</a></li>
              <li><a href="#" className="footer-link text-slate-400 hover:text-cyan-400 transition-colors">Careers</a></li>
            </ul>
          </div>
        </div>

        <div className="footer-bottom border-t border-slate-800 mt-12 pt-8 flex flex-col md:flex-row justify-between items-center">
          <div className="footer-copyright text-slate-500">
            © {currentYear} AI-Human Intelligence Convergence. All rights reserved.
          </div>
          <div className="footer-links mt-4 md:mt-0">
            <a href="/privacy" className="footer-link text-slate-500 hover:text-cyan-400 mx-4 transition-colors">Privacy Policy</a>
            <a href="/terms" className="footer-link text-slate-500 hover:text-cyan-400 mx-4 transition-colors">Terms of Service</a>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default HomepageFooter;