import React, { useState } from 'react';
import './auth.css';

const SignupForm = ({ onSignupSuccess }) => {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    password: '',
    confirmPassword: '',
    softwareExperience: '',
    hardwareFamiliarity: '',
    learningGoals: ''
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    // Validation
    if (formData.password !== formData.confirmPassword) {
      setError('Passwords do not match');
      return;
    }

    if (!formData.softwareExperience || !formData.hardwareFamiliarity) {
      setError('Please complete all required background information');
      return;
    }

    setIsLoading(true);
    setError('');

    try {
      const response = await fetch('/api/auth/signup', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name: formData.name,
          email: formData.email,
          password: formData.password,
          background: {
            software_experience: formData.softwareExperience,
            hardware_familiarity: formData.hardwareFamiliarity,
            learning_goals: formData.learningGoals
          }
        })
      });

      const data = await response.json();

      if (response.ok) {
        // Store the token in localStorage
        localStorage.setItem('access_token', data.access_token);
        localStorage.setItem('user', JSON.stringify({
          id: data.user_id,
          name: data.name,
          email: data.email
        }));

        // Call the success callback
        if (onSignupSuccess) {
          onSignupSuccess();
        }
      } else {
        setError(data.detail || 'Signup failed. Please try again.');
      }
    } catch (err) {
      setError('Network error. Please check your connection and try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="auth-container">
      <div className="auth-form">
        <h2>Create Account</h2>
        <p>Join our community to personalize your learning experience</p>

        {error && (
          <div className="error-message" role="alert">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="name">Full Name</label>
            <input
              type="text"
              id="name"
              name="name"
              value={formData.name}
              onChange={handleChange}
              required
              placeholder="Enter your full name"
              aria-required="true"
            />
          </div>

          <div className="form-group">
            <label htmlFor="email">Email Address</label>
            <input
              type="email"
              id="email"
              name="email"
              value={formData.email}
              onChange={handleChange}
              required
              placeholder="Enter your email"
              aria-required="true"
            />
          </div>

          <div className="form-group">
            <label htmlFor="password">Password</label>
            <input
              type="password"
              id="password"
              name="password"
              value={formData.password}
              onChange={handleChange}
              required
              minLength="6"
              placeholder="Create a password"
              aria-required="true"
            />
          </div>

          <div className="form-group">
            <label htmlFor="confirmPassword">Confirm Password</label>
            <input
              type="password"
              id="confirmPassword"
              name="confirmPassword"
              value={formData.confirmPassword}
              onChange={handleChange}
              required
              placeholder="Confirm your password"
              aria-required="true"
            />
          </div>

          <div className="form-section">
            <h3>Tell us about your background</h3>
            <p>We'll use this information to personalize your learning experience</p>

            <div className="form-group">
              <label htmlFor="softwareExperience">Software Experience</label>
              <select
                id="softwareExperience"
                name="softwareExperience"
                value={formData.softwareExperience}
                onChange={handleChange}
                required
                aria-required="true"
              >
                <option value="">Select your experience level</option>
                <option value="beginner">Beginner (just starting out)</option>
                <option value="intermediate">Intermediate (some experience)</option>
                <option value="advanced">Advanced (extensive experience)</option>
              </select>
            </div>

            <div className="form-group">
              <label htmlFor="hardwareFamiliarity">Hardware Familiarity</label>
              <select
                id="hardwareFamiliarity"
                name="hardwareFamiliarity"
                value={formData.hardwareFamiliarity}
                onChange={handleChange}
                required
                aria-required="true"
              >
                <option value="">Select your familiarity level</option>
                <option value="low-end">Low-end systems (basic hardware)</option>
                <option value="mid-range">Mid-range systems (standard PC/laptop)</option>
                <option value="high-performance">High-performance systems (advanced hardware)</option>
              </select>
            </div>

            <div className="form-group">
              <label htmlFor="learningGoals">Learning Goals (Optional)</label>
              <textarea
                id="learningGoals"
                name="learningGoals"
                value={formData.learningGoals}
                onChange={handleChange}
                placeholder="What do you hope to learn from this book?"
                rows="3"
              />
            </div>
          </div>

          <button
            type="submit"
            className="submit-button"
            disabled={isLoading}
            aria-busy={isLoading}
          >
            {isLoading ? 'Creating Account...' : 'Sign Up'}
          </button>
        </form>

        <div className="auth-footer">
          <p>Already have an account? <a href="/signin">Sign in</a></p>
        </div>
      </div>
    </div>
  );
};

export default SignupForm;