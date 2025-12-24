import React, { useState } from 'react';

const AIAssistantPreview = () => {
  const [messages, setMessages] = useState([
    { type: 'bot', text: 'Hello! I\'m your Physical AI & Robotics assistant. How can I help you understand humanoid robotics today?' },
    { type: 'user', text: 'What are the key challenges in humanoid locomotion?' },
    { type: 'bot', text: 'The main challenges in humanoid locomotion include: 1) Balance control on two legs, 2) Dynamic walking on uneven terrain, 3) Real-time adaptation to disturbances, and 4) Energy-efficient movement patterns.' }
  ]);

  const [inputValue, setInputValue] = useState('');

  const handleSendMessage = (e) => {
    e.preventDefault();
    if (inputValue.trim()) {
      setMessages([...messages, { type: 'user', text: inputValue }]);
      setInputValue('');

      // Simulate bot response after a delay
      setTimeout(() => {
        setMessages(prev => [...prev, {
          type: 'bot',
          text: 'That\'s an excellent question! In our curriculum, we cover this topic in depth in the "Humanoid Locomotion" module. Would you like to know more about the specific techniques we teach?'
        }]);
      }, 1000);
    }
  };

  return (
    <section className="py-20 bg-gradient-to-b from-slate-800 to-slate-900">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
            Interactive <span className="bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent">AI</span> Assistant
          </h2>
          <p className="text-lg text-slate-400 max-w-2xl mx-auto">
            Get instant answers to your questions about physical AI and humanoid robotics
          </p>
        </div>

        <div className="bg-slate-800/40 backdrop-blur-sm rounded-2xl border border-slate-700/50 overflow-hidden">
          <div className="p-6 border-b border-slate-700/50">
            <div className="flex items-center space-x-3">
              <div className="w-3 h-3 bg-red-500 rounded-full"></div>
              <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
              <div className="w-3 h-3 bg-green-500 rounded-full"></div>
              <span className="text-sm text-slate-400 ml-4">AI Assistant - Ready to help</span>
            </div>
          </div>

          <div className="p-6 h-96 overflow-y-auto">
            <div className="space-y-4">
              {messages.map((message, index) => (
                <div
                  key={index}
                  className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                      message.type === 'user'
                        ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white'
                        : 'bg-slate-700/50 text-slate-300'
                    }`}
                  >
                    {message.text}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="p-6 border-t border-slate-700/50">
            <form onSubmit={handleSendMessage} className="flex space-x-4">
              <input
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                placeholder="Ask about physical AI, robotics, or humanoid systems..."
                className="flex-1 bg-slate-700/50 border border-slate-600 rounded-lg px-4 py-3 text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
              <button
                type="submit"
                className="px-6 py-3 bg-gradient-to-r from-cyan-600 to-blue-600 text-white font-medium rounded-lg hover:shadow-lg transition-shadow duration-300"
              >
                Send
              </button>
            </form>
          </div>
        </div>

        <div className="mt-12 grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center p-6 bg-slate-800/30 rounded-xl border border-slate-700/50">
            <div className="text-2xl mb-3">🧠</div>
            <h3 className="font-semibold text-white mb-2">Concept Explanations</h3>
            <p className="text-sm text-slate-400">Get detailed explanations of complex robotics concepts</p>
          </div>
          <div className="text-center p-6 bg-slate-800/30 rounded-xl border border-slate-700/50">
            <div className="text-2xl mb-3">📚</div>
            <h3 className="font-semibold text-white mb-2">Curriculum Guidance</h3>
            <p className="text-sm text-slate-400">Navigate through learning paths and modules</p>
          </div>
          <div className="text-center p-6 bg-slate-800/30 rounded-xl border border-slate-700/50">
            <div className="text-2xl mb-3">🔧</div>
            <h3 className="font-semibold text-white mb-2">Technical Support</h3>
            <p className="text-sm text-slate-400">Get help with implementation and troubleshooting</p>
          </div>
        </div>
      </div>
    </section>
  );
};

export default AIAssistantPreview;