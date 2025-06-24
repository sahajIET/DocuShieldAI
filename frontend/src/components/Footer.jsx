import React from 'react';

const Footer = () => (
  <footer className="mt-12 mb-6 text-gray-500 text-sm text-center font-medium">
    <div className="bg-white/40 backdrop-blur-sm rounded-full px-6 py-3 border border-white/20 shadow-lg inline-block">
      &copy; {new Date().getFullYear()} DocuShield AI. Secure document redaction.
    </div>
  </footer>
);

export default Footer;