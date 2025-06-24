import React from 'react';

// Component to display the current backend connection status
const BackendStatus = ({ status, url }) => {
  const statusConfig = {
    connected: {
      style: 'bg-green-100 text-green-800',
      text: `✓ Backend Connected (${url})`,
    },
    disconnected: {
      style: 'bg-red-100 text-red-800',
      text: '✗ Backend Disconnected',
    },
    checking: {
      style: 'bg-yellow-100 text-yellow-800',
      text: '⏳ Checking Backend Connection...',
    },
  };

  const { style, text } = statusConfig[status] || statusConfig.checking;

  return (
    <div className="w-full mb-4 flex justify-center">
      <div className={`p-3 rounded-lg text-sm font-medium text-center max-w-md ${style}`}>
        {text}
      </div>
    </div>
  );
};

export default BackendStatus;
