import React from 'react';

// --- COMPONENT: BackendStatus & Footer (Updated with Glass UI) ---
const BackendStatus = ({ status, url }) => {
  const statusConfig = { 
    connected: { style: 'bg-green-500/20 text-green-800 border-green-200/30', text: `✓ Backend Connected (${url})` }, 
    disconnected: { style: 'bg-red-500/20 text-red-800 border-red-200/30', text: '✗ Backend Disconnected' }, 
    checking: { style: 'bg-yellow-500/20 text-yellow-800 border-yellow-200/30', text: '⏳ Checking Backend Connection...' } 
  };
  const { style, text } = statusConfig[status] || statusConfig.checking;
  
  return (
    <div className="w-full mb-6 flex justify-center">
      <div className={`p-4 rounded-2xl text-sm font-semibold text-center max-w-md backdrop-blur-md border shadow-xl ${style}`}>
        {text}
      </div>
    </div>
  );
};

export default BackendStatus;