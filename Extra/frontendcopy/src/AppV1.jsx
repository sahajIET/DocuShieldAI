import React, { useState, useRef, useEffect } from 'react';

// Main App Component for the PDF Redactor Frontend
const App = () => {
  // State to hold the selected file
  const [selectedFile, setSelectedFile] = useState(null);
  // State to manage loading status during API calls
  const [isLoading, setIsLoading] = useState(false);
  // State to store success or error messages
  const [message, setMessage] = useState('');
  
  // --- NEW: State for redaction types fetched from backend ---
  const [allRedactionTypes, setAllRedactionTypes] = useState([]);
  const [selectedRedactionTypes, setSelectedRedactionTypes] = useState([]);
  const [showTypeSelector, setShowTypeSelector] = useState(false);

  const fileInputRef = useRef(null);

  // --- NEW: Fetch all available redaction types from the backend on component mount ---
  useEffect(() => {
    const fetchRedactionTypes = async () => {
      try {
        const response = await fetch('http://127.0.0.1:8000/redaction-types');
        if (!response.ok) {
          throw new Error('Could not fetch redaction types from server.');
        }
        const data = await response.json();
        const types = data.redaction_types || [];
        setAllRedactionTypes(types);
        // Initially select all types by default
        setSelectedRedactionTypes(types); 
      } catch (error) {
        console.error("Failed to fetch redaction types:", error);
        setMessage(`Error: Could not connect to the backend to get settings. Please ensure the server is running.`);
      }
    };
    fetchRedactionTypes();
  }, []); // Empty dependency array means this runs once on mount


  // Handler for file input change (Unchanged)
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);
    setMessage(file ? `Selected: ${file.name}` : '');
  };

  // Handler for redaction type selection change (Unchanged)
  const handleTypeChange = (type) => {
    setSelectedRedactionTypes(prevTypes => {
      if (prevTypes.includes(type)) {
        return prevTypes.filter(t => t !== type);
      } else {
        return [...prevTypes, type];
      }
    });
  };

  // Handler to select/deselect all types (MODIFIED to use state)
  const handleSelectAll = () => {
    if (selectedRedactionTypes.length === allRedactionTypes.length) {
      setSelectedRedactionTypes([]);
    } else {
      setSelectedRedactionTypes(allRedactionTypes);
    }
  };

  // Helper functions for download (Unchanged)
  const extractFilenameFromHeader = (contentDisposition) => {
    if (!contentDisposition) return 'redacted_document.pdf';
    let match = contentDisposition.match(/filename\*=UTF-8''([^;]+)/i);
    if (match?.[1]) return decodeURIComponent(match[1]);
    match = contentDisposition.match(/filename="([^"]*)"/i);
    if (match?.[1]) return match[1];
    return 'redacted_document.pdf';
  };

  const triggerDownload = (blob, filename) => {
    const link = document.createElement('a');
    link.href = window.URL.createObjectURL(blob);
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(link.href);
  };

  // Handler for file upload (Unchanged)
  const handleUpload = async () => {
    if (!selectedFile) {
      setMessage('Please select a PDF file to upload.');
      return;
    }
    if (selectedRedactionTypes.length === 0) {
      setMessage('Please select at least one redaction type.');
      return;
    }

    setIsLoading(true);
    setMessage('Uploading and redacting... This may take a moment.');

    const formData = new FormData();
    formData.append('file', selectedFile);

    const queryParams = new URLSearchParams();
    selectedRedactionTypes.forEach(type => queryParams.append('redaction_types', type));
    const backendUrl = `http://127.0.0.1:8000/redact-pdf/?${queryParams.toString()}`;

    try {
      const response = await fetch(backendUrl, {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const contentDisposition = response.headers.get('Content-Disposition');
        const filename = extractFilenameFromHeader(contentDisposition);
        const blob = await response.blob();
        triggerDownload(blob, filename);
        setMessage(`Success! Your redacted file '${filename}' has started downloading.`);
      } else {
        const errorData = await response.json().catch(() => ({ detail: `Server responded with status ${response.status}` }));
        setMessage(`Error: ${errorData.detail}`);
      }
    } catch (error) {
      setMessage(`Network Error: ${error.message}. Please ensure the backend server is running.`);
    } finally {
      setIsLoading(false);
    }
  };

  // JSX/Render part (MODIFIED to handle dynamic types)
  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center justify-center p-4 font-sans">
      <div className="bg-white p-8 rounded-xl shadow-lg w-full max-w-md text-center">
        <h1 className="text-4xl font-bold text-gray-800 mb-6">DocuShield AI</h1>
        <p className="text-gray-600 mb-8">Securely redact sensitive information from your PDFs.</p>

        <div className="mb-6">
          <input id="pdf-upload" ref={fileInputRef} type="file" accept=".pdf,application/pdf" onChange={handleFileChange}
            className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"/>
        </div>

        <button onClick={() => setShowTypeSelector(!showTypeSelector)}
          className="w-full mb-4 px-4 py-2 rounded-lg font-semibold text-sm bg-gray-200 text-gray-700 hover:bg-gray-300 transition-colors">
          {showTypeSelector ? 'Hide Redaction Options' : 'Show Redaction Options'} ({selectedRedactionTypes.length} / {allRedactionTypes.length} selected)
        </button>

        {showTypeSelector && (
          <div className="border rounded-lg p-4 mb-6 text-left max-h-60 overflow-y-auto">
            <div className="flex justify-between items-center mb-3 pb-2 border-b">
              <h3 className="text-lg font-semibold text-gray-800">Choose Types</h3>
              <button onClick={handleSelectAll} className="text-blue-600 hover:text-blue-800 text-sm font-medium">
                {selectedRedactionTypes.length === allRedactionTypes.length ? 'Deselect All' : 'Select All'}
              </button>
            </div>
            {allRedactionTypes.map(type => (
              <div key={type} className="flex items-center mb-2">
                <input type="checkbox" id={`type-${type}`} checked={selectedRedactionTypes.includes(type)} onChange={() => handleTypeChange(type)}
                  className="h-4 w-4 text-blue-600 rounded focus:ring-blue-500"/>
                <label htmlFor={`type-${type}`} className="ml-2 text-gray-700 text-sm cursor-pointer">
                  {type.replace(/_/g, ' ').replace('PRESIDIO', 'Presidio:').replace('REGEX', 'Regex:').replace('SPACY', 'SpaCy:')}
                </label>
              </div>
            ))}
          </div>
        )}

        <button onClick={handleUpload} disabled={!selectedFile || isLoading || selectedRedactionTypes.length === 0}
          className={`w-full px-4 py-3 rounded-xl font-semibold text-lg transition-all ${!selectedFile || isLoading || selectedRedactionTypes.length === 0 ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700 text-white shadow-md'} ${isLoading ? 'animate-pulse' : ''}`}>
          {isLoading ? 'Processing...' : 'Redact & Download'}
        </button>

        {message && (
          <div className="mt-6">
             <p className={`p-3 rounded-lg text-sm font-medium ${message.toLowerCase().startsWith('error') ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800'}`}>
                {message}
             </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default App;
