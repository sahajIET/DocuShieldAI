import React, { useEffect, useRef } from 'react';
import { gsap } from 'gsap';

const WelcomePage = ({ onEnterApp }) => {
  const titleRef = useRef(null);
  const subtitleRef = useRef(null);
  const buttonRef = useRef(null);
  const videoRef = useRef(null); // Ref for the video element

  useEffect(() => {
    // GSAP Animations
    // Ensure video plays and loops automatically
    if (videoRef.current) {
      videoRef.current.play().catch(error => {
        console.log("Autoplay prevented:", error);
        // Fallback for autoplay restrictions: show button to manually play
        // Or just allow it to play silently until user interacts.
        // Modern browsers often block autoplay with sound.
      });
    }

    // Timeline for sequential animations
    const tl = gsap.timeline({ defaults: { ease: "power3.out" } });

    // Animate title letters
    tl.from(titleRef.current.children, {
      y: 50,
      opacity: 0,
      stagger: 0.05,
      duration: 0.8,
      delay: 0.5 // Small delay to allow video to start
    })
    // Animate subtitle
    .from(subtitleRef.current, {
      y: 20,
      opacity: 0,
      duration: 0.6
    }, "-=0.3") // Start subtitle animation slightly before title finishes
    // Animate button
    .from(buttonRef.current, {
      scale: 0.8,
      opacity: 0,
      duration: 0.5,
      ease: "back.out(1.7)"
    }, "-=0.2"); // Start button animation slightly before subtitle finishes

    // Optional: Animate video overlay/opacity if you want it to fade in
    gsap.fromTo(".video-overlay", 
      { opacity: 1 }, 
      { opacity: 0.6, duration: 2, ease: "power2.inOut" }
    );

  }, []);

  return (
    <div className="relative min-h-screen w-full flex items-center justify-center text-white overflow-hidden bg-black">
      {/* Video Background - Placeholder */}
      {/* You need to replace 'your-tech-video.mp4' with a path to your actual royalty-free video.
          For example, from sites like Pexels, Pixabay, Videvo, etc.
          Choose a dark, abstract, tech-themed video for the best effect.
          Ensure it's optimized for web (e.g., MP4 format, reasonable size). */}
      <video
        ref={videoRef}
        autoPlay
        loop
        muted // Muted is often required for autoplay in browsers
        playsInline // Important for iOS
        className="absolute z-0 w-full h-full object-cover opacity-70" // Adjust opacity for desired effect
        src="https://assets.mixkit.co/videos/preview/mixkit-abstract-visual-of-a-technological-network-24016-large.mp4" 
        // This is a sample video from mixkit.co, check their license.
        // It's recommended to host your video locally for production by placing it in `frontend/public/videos/`
        // and then using `src="/videos/your-video-name.mp4"`
      >
        Your browser does not support the video tag.
      </video>

      {/* Dark Overlay for better text readability */}
      <div className="absolute z-10 w-full h-full bg-gradient-to-t from-black via-black/80 to-transparent video-overlay"></div>

      {/* Content */}
      <div className="relative z-20 flex flex-col items-center p-4 text-center">
        <h1 ref={titleRef} className="text-5xl sm:text-7xl lg:text-8xl font-extrabold mb-4 drop-shadow-lg">
          {"DocuShield AI".split("").map((char, index) => (
            <span key={index} className="inline-block">{char}</span>
          ))}
        </h1>
        <p ref={subtitleRef} className="text-lg sm:text-xl lg:text-2xl mb-8 font-light max-w-2xl">
          Securely redact sensitive information from your documents with advanced AI.
        </p>
        <button
          ref={buttonRef}
          onClick={onEnterApp}
          className="px-8 py-4 bg-blue-600 hover:bg-blue-700 text-white text-xl font-semibold rounded-full shadow-lg transition-all duration-300 ease-in-out transform hover:scale-105 focus:outline-none focus:ring-4 focus:ring-blue-500 focus:ring-opacity-50"
        >
          Get Started
        </button>
      </div>
    </div>
  );
};

export default WelcomePage;
