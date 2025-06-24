import React, { useRef, useEffect } from 'react';
import { logoUrl } from '../utils/constants.js';

// --- GSAP Animation Library ---
// We'll load this from a CDN to ensure it's available
const loadGsap = (callback) => {
    if (window.gsap) {
        callback();
        return;
    }
    const script = document.createElement('script');
    script.src = 'https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.2/gsap.min.js';
    script.onload = callback;
    document.head.appendChild(script);
};

// --- IMPROVED: WELCOME PAGE COMPONENT with Glass UI Design ---
const WelcomePage = ({ onGetStarted }) => {
  const containerRef = useRef(null);
  const docRef = useRef(null);
  const buttonRef = useRef(null);
  const leftContentRef = useRef(null);
  const rightContentRef = useRef(null);

  useEffect(() => {
      loadGsap(() => {
          const tl = window.gsap.timeline();
          const piiItems = ".pii-item";
          const shields = ".shield-item";

          // Set initial states
          window.gsap.set(leftContentRef.current, { x: -100, opacity: 0 });
          window.gsap.set(rightContentRef.current, { x: 100, opacity: 0 });
          window.gsap.set(docRef.current, { y: -200, opacity: 0, scale: 0.8 });
          window.gsap.set(piiItems, { scale: 0, opacity: 0 });
          window.gsap.set(shields, { scale: 0, opacity: 0 });
          window.gsap.set(buttonRef.current, { opacity: 0, y: 30, scale: 0.9 });

          // Animation sequence - Button appears earlier
          tl.to(leftContentRef.current, { x: 0, opacity: 1, duration: 1, ease: "power3.out" })
            .to(rightContentRef.current, { x: 0, opacity: 1, duration: 1, ease: "power3.out" }, "-=0.8")
            .to(buttonRef.current, { opacity: 1, y: 0, scale: 1, duration: 0.8, ease: "power3.out" }, "-=0.5") // Moved earlier
            .to(docRef.current, { y: 0, opacity: 1, scale: 1, duration: 1.2, ease: "power2.out" }, "-=0.6")
            .to(piiItems, { scale: 1, opacity: 1, duration: 0.6, stagger: 0.15, ease: "back.out(1.7)" }, "-=0.4")
            .to(shields, { scale: 1, opacity: 1, duration: 0.6, stagger: 0.15, ease: "elastic.out(1, 0.5)" }, "+=0.3")
            .to(piiItems, { opacity: 0.4, duration: 0.7 }, "<");

          // Floating animation for PII items
          window.gsap.to(piiItems, {
              y: "random(-10, 10)",
              rotation: "random(-5, 5)",
              duration: 3,
              ease: "sine.inOut",
              stagger: 0.2,
              repeat: -1,
              yoyo: true
          });

          // Subtle pulse for shields
          window.gsap.to(shields, {
              scale: 1.1,
              duration: 2,
              ease: "sine.inOut",
              stagger: 0.3,
              repeat: -1,
              yoyo: true
          });
      });
  }, []);

  const piiData = [
      { icon: "👤", label: "Name", top: '15%', left: '25%' },
      { icon: "📞", label: "Phone", top: '35%', left: '70%' },
      { icon: "✉️", label: "Email", top: '55%', left: '15%' },
      { icon: "💳", label: "Card", top: '75%', left: '60%' },
      { icon: "📍", label: "Address", top: '20%', left: '80%' },
      { icon: "🏢", label: "Company", top: '65%', left: '10%' },
      { icon: "🔢", label: "SSN", top: '45%', left: '85%' },
      { icon: "🏦", label: "Bank", top: '85%', left: '25%' },
  ];

  return (
      <div ref={containerRef} className="min-h-screen w-full flex items-center justify-center bg-gradient-to-br from-blue-50 via-indigo-50 to-black-50 p-4 font-sans overflow-hidden relative">
          {/* Background Elements */}
          <div className="absolute inset-0 overflow-hidden">
              <div className="absolute -top-40 -right-40 w-80 h-80 bg-gradient-to-br from-blue-400/20 to-black-400/20 rounded-full blur-3xl"></div>
              <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-gradient-to-br from-indigo-400/20 to-pink-400/20 rounded-full blur-3xl"></div>
          </div>

          <div className="flex flex-col lg:flex-row items-center justify-center gap-16 lg:gap-24 w-full max-w-7xl relative z-10">
              
              {/* Left Side - Animation & Logo */}
              <div ref={leftContentRef} className="flex flex-col items-center lg:items-start space-y-8">
                  <div className="text-center lg:text-left">
                      <div className="flex items-center">
                      <div className="flex items-center justify-center lg:justify-start mb-6">
                          <div className="relative">
                              <img src={logoUrl} alt="DocuShield Logo" className="h-20 w-20 rounded-2xl shadow-2xl border-4 border-white/50" />
                              <div className="absolute inset-0 bg-gradient-to-br from-blue-500/20 to-black-500/20 rounded-2xl"></div>
                          </div>
                      </div>
                      
                      <h1 className="ml-4 text-5xl lg:text-6xl font-bold bg-gradient-to-r from-blue-600 via-black-600 to-indigo-600 bg-clip-text text-transparent mb-4">
                          DocuShield AI
                      </h1>
                      </div>
                      <p className="text-xl lg:text-2xl text-gray-600 font-medium">
                          Protecting Your Privacy with Intelligence
                      </p>
                      <div className="flex flex-wrap justify-center lg:justify-start gap-3 mt-6">
                          <span className="px-4 py-2 bg-white/60 backdrop-blur-sm rounded-full text-sm font-semibold text-gray-700 border border-white/20">🛡️ GDPR Compliant</span>
                          <span className="px-4 py-2 bg-white/60 backdrop-blur-sm rounded-full text-sm font-semibold text-gray-700 border border-white/20">⚡ AI Powered</span>
                          <span className="px-4 py-2 bg-white/60 backdrop-blur-sm rounded-full text-sm font-semibold text-gray-700 border border-white/20">🔒 Secure</span>
                      </div>
                  </div>

                  {/* Interactive Document Animation */}
                  <div className="relative w-80 h-96 lg:w-96 lg:h-[420px]">
                      <div ref={docRef} className="absolute inset-0 bg-white/80 backdrop-blur-md rounded-2xl shadow-2xl border border-white/20 p-6 flex flex-col">
                          {/* Document Header */}
                          <div className="flex items-center mb-4">
                              <div className="w-3 h-3 bg-red-400 rounded-full mr-2"></div>
                              <div className="w-3 h-3 bg-yellow-400 rounded-full mr-2"></div>
                              <div className="w-3 h-3 bg-green-400 rounded-full"></div>
                              <div className="ml-auto text-xs text-gray-500 font-mono">document.pdf</div>
                          </div>
                          
                          {/* Document Content */}
                          <div className="space-y-3 flex-1">
                              <div className="h-3 bg-gradient-to-r from-gray-300 to-gray-200 w-3/4 rounded-full"></div>
                              <div className="h-3 bg-gradient-to-r from-gray-300 to-gray-200 w-full rounded-full"></div>
                              <div className="h-3 bg-gradient-to-r from-gray-300 to-gray-200 w-1/2 rounded-full"></div>
                              <div className="h-3 bg-gradient-to-r from-gray-300 to-gray-200 w-full rounded-full"></div>
                              <div className="h-3 bg-gradient-to-r from-gray-300 to-gray-200 w-5/6 rounded-full"></div>
                              <div className="h-3 bg-gradient-to-r from-gray-300 to-gray-200 w-3/4 rounded-full"></div>
                              <div className="h-3 bg-gradient-to-r from-gray-300 to-gray-200 w-full rounded-full"></div>
                              <div className="h-3 bg-gradient-to-r from-gray-300 to-gray-200 w-2/3 rounded-full"></div>
                          </div>
                      </div>
                      
                      {/* Floating PII Elements */}
                      {piiData.map((item, index) => (
                          <React.Fragment key={index}>
                              <div 
                                  className="pii-item absolute text-2xl z-10 filter drop-shadow-lg" 
                                  style={{ top: item.top, left: item.left }}
                              >
                                  <div className="bg-white/90 backdrop-blur-sm rounded-full p-2 border border-red-200/50">
                                      {item.icon}
                                  </div>
                              </div>
                              <div 
                                  className="shield-item absolute text-3xl z-20 filter drop-shadow-lg" 
                                  style={{ top: `calc(${item.top} - 4px)`, left: `calc(${item.left} - 4px)` }}
                              >
                                  <div className="bg-gradient-to-br from-blue-500/90 to-black-500/90 backdrop-blur-sm rounded-full p-1 border border-white/30">
                                      🛡️
                                  </div>
                              </div>
                          </React.Fragment>
                      ))}
                  </div>
              </div>

              {/* Right Side - Call to Action */}
              <div ref={rightContentRef} className="flex flex-col items-center lg:items-start space-y-8 text-center lg:text-left">
                  <div className="space-y-6">
                      <h2 className="text-3xl lg:text-4xl font-bold text-gray-800 leading-tight">
                          Ready to Secure Your
                          <span className="block bg-gradient-to-r from-blue-600 to-red-600 bg-clip-text text-transparent">
                              Sensitive Documents?
                          </span>
                      </h2>
                      
                      <div className="space-y-4 text-gray-600">
                          <div className="flex items-center justify-center lg:justify-start space-x-3">
                              <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-500 rounded-full flex items-center justify-center text-white text-sm font-bold">1</div>
                              <span className="text-lg">Upload your PDF document</span>
                          </div>
                          <div className="flex items-center justify-center lg:justify-start space-x-3">
                              <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-indigo-500 rounded-full flex items-center justify-center text-white text-sm font-bold">2</div>
                              <span className="text-lg">AI detects sensitive information</span>
                          </div>
                          <div className="flex items-center justify-center lg:justify-start space-x-3">
                              <div className="w-8 h-8 bg-gradient-to-br from-indigo-500 to-blue-500 rounded-full flex items-center justify-center text-white text-sm font-bold">3</div>
                              <span className="text-lg">Download your protected document</span>
                          </div>
                      </div>

                      <div className="bg-white/60 backdrop-blur-md rounded-2xl p-6 border border-white/20 shadow-xl">
                          <h3 className="text-lg font-semibold text-gray-800 mb-3">✨ AI-Powered Features</h3>
                          <ul className="space-y-2 text-sm text-gray-600">
                              <li className="flex items-center space-x-2">
                                  <span className="w-2 h-2 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full"></span>
                                  <span>Smart content analysis and suggestions</span>
                              </li>
                              <li className="flex items-center space-x-2">
                                  <span className="w-2 h-2 bg-gradient-to-r from-purple-500 to-indigo-500 rounded-full"></span>
                                  <span>Automated privacy compliance reporting</span>
                              </li>
                              <li className="flex items-center space-x-2">
                                  <span className="w-2 h-2 bg-gradient-to-r from-indigo-500 to-blue-500 rounded-full"></span>
                                  <span>Multi-language document support</span>
                              </li>
                          </ul>
                      </div>
                  </div>

                  {/* Enhanced Get Started Button */}
                  <div ref={buttonRef} className="relative group">
                      <div className="absolute -inset-1 bg-gradient-to-r from-blue-500 via-black-500 to-indigo-500 rounded-2xl blur opacity-75 group-hover:opacity-100 transition duration-300"></div>
                      <button 
                          onClick={onGetStarted} 
                          className="relative px-12 py-4 bg-gradient-to-r from-blue-600 via-black-600 to-indigo-600 text-white font-bold text-xl rounded-2xl shadow-2xl transform transition-all duration-300 hover:scale-105 active:scale-95 flex items-center space-x-3"
                      >
                          <span>Get Started</span>
                          <svg className="w-6 h-6 transform transition-transform group-hover:translate-x-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                          </svg>
                      </button>
                      
                      {/* Floating particles effect */}
                      <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none">
                          <div className="absolute top-2 left-4 w-1 h-1 bg-white rounded-full animate-ping"></div>
                          <div className="absolute bottom-3 right-6 w-1 h-1 bg-white rounded-full animate-ping animation-delay-150"></div>
                          <div className="absolute top-1/2 right-2 w-1 h-1 bg-white rounded-full animate-ping animation-delay-300"></div>
                      </div>
                  </div>

                  <p className="text-sm text-gray-500 max-w-md">
                      Your documents never leave your control. All processing is done securely with enterprise-grade privacy protection.
                  </p>
              </div>
          </div>

          {/* Additional CSS for animations */}
          <style>{`
              .animation-delay-150 {
                  animation-delay: 150ms;
              }
              .animation-delay-300 {
                  animation-delay: 300ms;
              }
              @keyframes float {
                  0%, 100% { transform: translateY(0px) rotate(0deg); }
                  50% { transform: translateY(-10px) rotate(2deg); }
              }
          `}</style>
      </div>
  );
};

export default WelcomePage;