import React from 'react';

const FuturisticFeatures = () => {
  const features = [
    {
      title: "Physical AI Fundamentals",
      description: "Learn the core principles of embodied artificial intelligence and how AI systems interact with the physical world.",
      icon: "🤖"
    },
    {
      title: "Humanoid Robotics",
      description: "Explore the design, control, and intelligence of humanoid robots that mimic human movement and behavior.",
      icon: "🦾"
    },
    {
      title: "Neural Control Systems",
      description: "Deep dive into neural networks that control complex robotic systems and enable adaptive behavior.",
      icon: "🧠"
    },
    {
      title: "Sensor Integration",
      description: "Master the integration of multiple sensors for perception, navigation, and environmental interaction.",
      icon: "📡"
    },
    {
      title: "AI Safety & Ethics",
      description: "Understand the critical safety and ethical considerations in deploying AI and robotic systems.",
      icon: "🔒"
    },
    {
      title: "Real-world Applications",
      description: "Explore practical implementations of AI and robotics in industry, healthcare, and daily life.",
      icon: "🏭"
    }
  ];

  return (
    <section className="features-section padding-top--xl padding-bottom--xl">
      <div className="container">
        <div className="text--center margin-bottom--lg">
          <h2 className="section-title cyberpunk-title">Core Learning Modules</h2>
          <p className="section-subtitle">Cutting-edge curriculum designed for the future of robotics</p>
        </div>

        <div className="row">
          {features.map((feature, index) => (
            <div key={index} className="col col--4 margin-bottom--lg">
              <div className="feature-card robotic-card neon-glow">
                <div className="feature-icon">
                  <span className="icon-emoji">{feature.icon}</span>
                </div>
                <h3 className="feature-title">{feature.title}</h3>
                <p className="feature-description">{feature.description}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default FuturisticFeatures;