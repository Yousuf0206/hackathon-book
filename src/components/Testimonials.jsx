import React from 'react';

const Testimonials = () => {
  const testimonials = [
    {
      quote: "The AI-Human collaboration modules transformed how I approach robotics development. The practical applications are game-changing.",
      author: "Dr. Sarah Chen",
      role: "Senior Robotics Engineer",
      company: "Tech Innovations Inc.",
      avatar: "👩‍🔬"
    },
    {
      quote: "An exceptional resource for understanding the intersection of artificial intelligence and human-centered robotics. Highly recommended for professionals.",
      author: "Marcus Rodriguez",
      role: "AI Research Lead",
      company: "Future Robotics Lab",
      avatar: "👨‍💻"
    },
    {
      quote: "The hands-on projects provided real insight into implementing AI-human interfaces. My team's productivity increased by 40% after applying these concepts.",
      author: "Dr. Aisha Patel",
      role: "Director of Engineering",
      company: "Human-Robot Systems",
      avatar: "👩‍💼"
    }
  ];

  return (
    <div className="testimonials-section py-20 bg-gradient-to-b from-slate-100 to-slate-200 dark:from-slate-800 dark:to-slate-900">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="section-title text-4xl lg:text-5xl font-bold text-slate-900 dark:text-white mb-4">
            What Experts Say
          </h2>
          <p className="section-subtitle text-xl text-slate-600 dark:text-slate-300 max-w-3xl mx-auto">
            Insights from leading professionals in AI-humanoid robotics
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {testimonials.map((testimonial, index) => (
            <div
              key={index}
              className="testimonial-card group relative bg-white/80 dark:bg-slate-800/60 backdrop-blur-sm rounded-2xl p-8 shadow-lg hover:shadow-2xl transition-all duration-500 border border-slate-200/50 dark:border-slate-700/50 hover:border-cyan-300/30 dark:hover:border-cyan-500/30 overflow-hidden"
            >
              <div className="absolute inset-0 bg-gradient-to-br opacity-0 group-hover:opacity-5 transition-opacity duration-500"></div>

              <div className="relative z-10">
                <div className="testimonial-quote text-5xl text-cyan-400 mb-4">"</div>

                <p className="testimonial-text text-slate-600 dark:text-slate-300 mb-6 text-lg italic">
                  {testimonial.quote}
                </p>

                <div className="testimonial-author flex items-center">
                  <div className="author-avatar text-3xl mr-4">
                    {testimonial.avatar}
                  </div>
                  <div className="author-info">
                    <div className="author-name font-bold text-slate-900 dark:text-white">
                      {testimonial.author}
                    </div>
                    <div className="author-role text-sm text-slate-500 dark:text-slate-400">
                      {testimonial.role}, {testimonial.company}
                    </div>
                  </div>
                </div>
              </div>

              <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/0 via-transparent to-blue-500/0 group-hover:from-cyan-500/5 group-hover:to-blue-500/5 transition-all duration-500 pointer-events-none"></div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Testimonials;