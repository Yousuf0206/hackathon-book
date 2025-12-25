import React from 'react';
import './component-styles.css';

const ModernFooter = () => {
  return (
    <footer className="modern-footer">
      <div className="modern-footer-container">
        <div className="modern-footer-grid">
          {/* Brand */}
          <div className="modern-footer-brand">
            <div className="modern-footer-title">
              Physical AI & Humanoid Robotics
            </div>
            <p className="modern-footer-description">
              The comprehensive guide to mastering artificial intelligence and humanoid robotics.
            </p>
            <div className="modern-footer-social">
              <a href="#" className="modern-footer-social-link">
                <span className="modern-footer-social-icon">f</span>
              </a>
              <a href="#" className="modern-footer-social-link">
                <span className="modern-footer-social-icon">t</span>
              </a>
              <a href="#" className="modern-footer-social-link">
                <span className="modern-footer-social-icon">in</span>
              </a>
              <a href="#" className="modern-footer-social-link">
                <span className="modern-footer-social-icon">gh</span>
              </a>
            </div>
          </div>

          {/* Quick Links */}
          <div>
            <h3 className="modern-footer-section-title">Quick Links</h3>
            <ul className="modern-footer-links">
              <li><a href="#" className="modern-footer-link">Home</a></li>
              <li><a href="#" className="modern-footer-link">Modules</a></li>
              <li><a href="#" className="modern-footer-link">Projects</a></li>
              <li><a href="#" className="modern-footer-link">Resources</a></li>
              <li><a href="#" className="modern-footer-link">Community</a></li>
            </ul>
          </div>

          {/* Learning Paths */}
          <div>
            <h3 className="modern-footer-section-title">Learning Paths</h3>
            <ul className="modern-footer-links">
              <li><a href="#" className="modern-footer-link">Beginner's Path</a></li>
              <li><a href="#" className="modern-footer-link">Developer's Path</a></li>
              <li><a href="#" className="modern-footer-link">Expert's Path</a></li>
              <li><a href="#" className="modern-footer-link">Certification</a></li>
              <li><a href="#" className="modern-footer-link">Career Services</a></li>
            </ul>
          </div>

          {/* Contact */}
          <div>
            <h3 className="modern-footer-section-title">Contact Us</h3>
            <ul>
              <li className="modern-footer-contact-item">
                <svg className="modern-footer-contact-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 4.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                </svg>
                <span className="modern-footer-contact-text">muhammad.yousuf02@gmail.com</span>
              </li>
              <li className="modern-footer-contact-item">
                <svg className="modern-footer-contact-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
                <span className="modern-footer-contact-text">Karachi, Sindh, Pakistan</span>
              </li>
              <li className="modern-footer-contact-item">
                <svg className="modern-footer-contact-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z" />
                </svg>
                <span className="modern-footer-contact-text">+92 300 2554389</span>
              </li>
            </ul>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="modern-footer-bottom">
          <div className="modern-footer-copyright">
            © 2025 Physical AI & Humanoid Robotics. All rights reserved.
          </div>
          <div className="modern-footer-bottom-links">
            <a href="#" className="modern-footer-bottom-link">Privacy Policy</a>
            <a href="#" className="modern-footer-bottom-link">Terms of Service</a>
            <a href="#" className="modern-footer-bottom-link">Cookie Policy</a>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default ModernFooter;