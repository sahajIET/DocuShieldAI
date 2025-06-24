import React from 'react';

// Simple footer component
const Footer = () => {
  return (
    <footer className="mt-10 mb-6 text-gray-500 text-sm text-center">
      &copy; {new Date().getFullYear()} DocuShield AI. Secure document redaction.
    </footer>
  );
};

export default Footer;
