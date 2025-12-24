import React, { useState } from 'react';

const TestimonialsSlider = () => {
  const [currentIndex, setCurrentIndex] = useState(0);

  const testimonials = [
    {
      name: "Dr. Sarah Chen",
      role: "Senior Robotics Engineer",
      company: "Boston Dynamics",
      content: "This comprehensive guide transformed my understanding of AI-humanoid integration. The practical examples and real-world applications made complex concepts accessible.",
      avatar: "👩‍🔬",
      rating: 5
    },
    {
      name: "Marcus Rodriguez",
      role: "AI Research Lead",
      company: "NVIDIA",
      content: "The depth of coverage on ROS 2 and NVIDIA Isaac is unparalleled. This book became my go-to reference for advanced robotics development.",
      avatar: "👨‍💻",
      rating: 5
    },
    {
      name: "Dr. Aisha Patel",
      role: "Professor of Robotics",
      company: "MIT",
      content: "An excellent resource for both students and professionals. The blend of theoretical concepts with hands-on projects is perfect for learning.",
      avatar: "👩‍🏫",
      rating: 5
    }
  ];

  const nextTestimonial = () => {
    setCurrentIndex((prevIndex) => (prevIndex + 1) % testimonials.length);
  };

  const prevTestimonial = () => {
    setCurrentIndex((prevIndex) => (prevIndex - 1 + testimonials.length) % testimonials.length);
  };

  const currentTestimonial = testimonials[currentIndex];

  return (
    <section className="py-20 bg-gradient-to-b from-slate-100 to-slate-200 dark:from-slate-800 dark:to-slate-900">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold text-slate-900 dark:text-white mb-4">
            What Experts Say
          </h2>
          <p className="text-xl text-slate-600 dark:text-slate-300 max-w-3xl mx-auto">
            Hear from industry leaders and academics who have used this comprehensive guide
          </p>
        </div>

        <div className="relative max-w-4xl mx-auto">
          {/* Testimonial Card */}
          <div className="bg-white/80 dark:bg-slate-800/60 backdrop-blur-lg rounded-3xl p-10 shadow-2xl border border-white/20 dark:border-slate-700/50 relative overflow-hidden">
            {/* Background gradient */}
            <div className="absolute top-0 right-0 w-64 h-64 bg-gradient-to-br from-cyan-500/10 to-blue-500/10 rounded-full -translate-y-32 translate-x-32"></div>

            <div className="relative z-10">
              {/* Quote icon */}
              <div className="text-6xl text-cyan-400 mb-6 opacity-30">“</div>

              {/* Content */}
              <p className="text-xl md:text-2xl text-slate-700 dark:text-slate-300 leading-relaxed mb-8">
                {currentTestimonial.content}
              </p>

              {/* Author info */}
              <div className="flex items-center">
                <div className="text-4xl mr-4">{currentTestimonial.avatar}</div>
                <div>
                  <div className="font-bold text-slate-900 dark:text-white text-lg">
                    {currentTestimonial.name}
                  </div>
                  <div className="text-slate-600 dark:text-slate-400">
                    {currentTestimonial.role}, {currentTestimonial.company}
                  </div>

                  {/* Rating */}
                  <div className="flex mt-2">
                    {[...Array(currentTestimonial.rating)].map((_, i) => (
                      <svg key={i} className="w-5 h-5 text-yellow-400" fill="currentColor" viewBox="0 0 20 20">
                        <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461c.969 0 1.371-1.24.588-1.81L9.049 2.927z" />
                      </svg>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Navigation buttons */}
          <button
            onClick={prevTestimonial}
            className="absolute left-0 top-1/2 transform -translate-y-1/2 -translate-x-8 w-12 h-12 bg-white/80 dark:bg-slate-800/60 backdrop-blur-lg rounded-full shadow-lg border border-white/20 dark:border-slate-700/50 flex items-center justify-center hover:bg-cyan-500 hover:text-white transition-all duration-300"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
          </button>

          <button
            onClick={nextTestimonial}
            className="absolute right-0 top-1/2 transform -translate-y-1/2 translate-x-8 w-12 h-12 bg-white/80 dark:bg-slate-800/60 backdrop-blur-lg rounded-full shadow-lg border border-white/20 dark:border-slate-700/50 flex items-center justify-center hover:bg-cyan-500 hover:text-white transition-all duration-300"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </button>

          {/* Dots indicator */}
          <div className="flex justify-center mt-8 space-x-2">
            {testimonials.map((_, index) => (
              <button
                key={index}
                onClick={() => setCurrentIndex(index)}
                className={`w-3 h-3 rounded-full transition-all duration-300 ${
                  index === currentIndex
                    ? 'bg-cyan-500 w-8'
                    : 'bg-slate-300 dark:bg-slate-600 hover:bg-slate-400 dark:hover:bg-slate-500'
                }`}
              />
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

export default TestimonialsSlider;