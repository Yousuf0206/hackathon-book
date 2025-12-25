import React, { useState } from 'react';

const FuturisticTestimonials = () => {
  const [activeIndex, setActiveIndex] = useState(0);

  const testimonials = [
    {
      name: "Dr. Sarah Chen",
      role: "Senior Robotics Engineer at Tesla",
      content: "This curriculum transformed my understanding of humanoid robotics. The hands-on projects with real hardware were invaluable for my career advancement.",
      avatar: "👩‍🔬",
      rating: 5
    },
    {
      name: "Marcus Rodriguez",
      role: "AI Research Scientist at NVIDIA",
      content: "The integration of physical AI concepts with practical implementation was exceptional. I've applied these concepts directly to my research projects.",
      avatar: "👨‍💻",
      rating: 5
    },
    {
      name: "Dr. Aisha Patel",
      role: "Professor of Robotics at MIT",
      content: "As an educator, I appreciate how this curriculum bridges the gap between theory and practice. It's become my go-to resource for advanced robotics courses.",
      avatar: "👩‍🏫",
      rating: 5
    }
  ];

  const nextTestimonial = () => {
    setActiveIndex((prev) => (prev + 1) % testimonials.length);
  };

  const prevTestimonial = () => {
    setActiveIndex((prev) => (prev - 1 + testimonials.length) % testimonials.length);
  };

  return (
    <section className="testimonials-section padding-top--xl padding-bottom--xl">
      <div className="container">
        <div className="text--center margin-bottom--lg">
          <h2 className="section-title cyberpunk-title">Expert Endorsements</h2>
          <p className="section-subtitle">Join thousands of professionals advancing in AI and robotics</p>
        </div>

        <div className="testimonials-container">
          <div className="testimonial-card robotic-card neon-glow">
            <div className="testimonial-content">
              <div className="testimonial-header">
                <div className="testimonial-avatar">
                  <span className="avatar-emoji">{testimonials[activeIndex].avatar}</span>
                </div>
                <div className="testimonial-info">
                  <h3 className="author-name">{testimonials[activeIndex].name}</h3>
                  <p className="author-role">{testimonials[activeIndex].role}</p>
                </div>
              </div>

              <div className="testimonial-rating">
                {[...Array(testimonials[activeIndex].rating)].map((_, i) => (
                  <span key={i} className="rating-star">★</span>
                ))}
              </div>

              <blockquote className="testimonial-text">
                "{testimonials[activeIndex].content}"
              </blockquote>
            </div>

            {/* Navigation buttons */}
            <div className="testimonial-navigation">
              <button
                onClick={prevTestimonial}
                className="nav-button nav-button-prev"
                aria-label="Previous testimonial"
              >
                &lt;
              </button>

              <div className="testimonial-indicators">
                {testimonials.map((_, index) => (
                  <button
                    key={index}
                    onClick={() => setActiveIndex(index)}
                    className={`indicator ${index === activeIndex ? 'active' : ''}`}
                    aria-label={`Go to testimonial ${index + 1}`}
                  />
                ))}
              </div>

              <button
                onClick={nextTestimonial}
                className="nav-button nav-button-next"
                aria-label="Next testimonial"
              >
                &gt;
              </button>
            </div>
          </div>

          {/* Stats bar */}
          <div className="testimonials-stats">
            <div className="stat-item">
              <div className="stat-number stat-number-1">95%</div>
              <div className="stat-label">Career Advancement</div>
            </div>
            <div className="stat-item">
              <div className="stat-number stat-number-2">500+</div>
              <div className="stat-label">Academic Citations</div>
            </div>
            <div className="stat-item">
              <div className="stat-number stat-number-3">50+</div>
              <div className="stat-label">Industry Partners</div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default FuturisticTestimonials;