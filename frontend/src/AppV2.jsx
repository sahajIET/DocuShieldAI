import React, { useState, useRef, useEffect } from 'react';

// Simple PDF viewer component using PDF.js from CDN
const PDFViewer = ({ file, onPageChange, currentPage, totalPages, onLoadSuccess }) => {
  const canvasRef = useRef(null);
  const [pdfDoc, setPdfDoc] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [pdfJsLoaded, setPdfJsLoaded] = useState(false);

  // Load PDF.js if not already loaded
  useEffect(() => {
    const loadPdfJs = () => {
      if (window.pdfjsLib) {
        setPdfJsLoaded(true);
        return;
      }

      const script = document.createElement('script');
      script.src = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js';
      script.onload = () => {
        if (window.pdfjsLib) {
          window.pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
          setPdfJsLoaded(true);
        }
      };
      script.onerror = () => setError('Failed to load PDF viewer library');
      document.head.appendChild(script);
    };

    loadPdfJs();
  }, []);

  // Load PDF when file changes and PDF.js is ready
  useEffect(() => {
    if (file && pdfJsLoaded) {
      loadPDF();
    }
  }, [file, pdfJsLoaded]);

  // Render page when PDF is loaded and currentPage changes
  useEffect(() => {
    if (pdfDoc && currentPage) {
      renderPage(currentPage);
    }
  }, [pdfDoc, currentPage]);

  const loadPDF = async () => {
    if (!file || !window.pdfjsLib) return;
    
    try {
      setIsLoading(true);
      setError(null);
      const arrayBuffer = await file.arrayBuffer();
      const pdf = await window.pdfjsLib.getDocument(arrayBuffer).promise;
      setPdfDoc(pdf);
      onLoadSuccess(pdf.numPages);
      setIsLoading(false);
    } catch (err) {
      console.error('Error loading PDF:', err);
      setError('Failed to load PDF. Please ensure this is a valid PDF file.');
      setIsLoading(false);
    }
  };

  const renderPage = async (pageNum) => {
    if (!pdfDoc || !canvasRef.current) return;

    try {
      const page = await pdfDoc.getPage(pageNum);
      const canvas = canvasRef.current;
      const context = canvas.getContext('2d');
      
      // Calculate scale to fit container
      const containerWidth = canvas.parentElement.clientWidth - 32; // Account for padding
      const viewport = page.getViewport({ scale: 1 });
      const scale = Math.min(containerWidth / viewport.width, 1.5);
      
      const scaledViewport = page.getViewport({ scale });
      canvas.width = scaledViewport.width;
      canvas.height = scaledViewport.height;

      // Clear canvas
      context.clearRect(0, 0, canvas.width, canvas.height);

      await page.render({
        canvasContext: context,
        viewport: scaledViewport
      }).promise;
    } catch (err) {
      console.error('Error rendering page:', err);
      setError('Failed to render PDF page');
    }
  };

  if (error) {
    return (
      <div className="flex flex-col justify-center items-center h-64 text-red-600 bg-red-50 rounded-lg p-4">
        <div className="text-lg font-semibold mb-2">PDF Preview Error</div>
        <div className="text-sm">{error}</div>
      </div>
    );
  }

  if (isLoading || !pdfJsLoaded) {
    return (
      <div className="flex justify-center items-center h-64 bg-gray-50 rounded-lg">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        <span className="ml-2 text-gray-600">Loading PDF preview...</span>
      </div>
    );
  }

  return (
    <div className="flex justify-center bg-gray-50 rounded-lg p-4">
      <canvas 
        ref={canvasRef} 
        className="border border-gray-300 shadow-lg max-w-full h-auto rounded bg-white"
        style={{ maxHeight: '600px' }}
      />
    </div>
  );
};

// Frontend validation constants
const MAX_FILE_SIZE_MB = 5;
const MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024;
const ALLOWED_FILE_TYPES = ['application/pdf'];

// Backend URL configuration
const getBackendUrl = () => {
  // Try different backend URLs in order of preference
  const possibleUrls = [
    'http://localhost:8000',
    'http://127.0.0.1:8000',
    'http://0.0.0.0:8000'
  ];
  
  return possibleUrls[0];
};

// Main App Component
const App = () => {
  // Basic states
  const [selectedFile, setSelectedFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [message, setMessage] = useState('');
  
  // Backend connection state
  const [backendStatus, setBackendStatus] = useState('checking'); // 'checking', 'connected', 'disconnected'
  const [backendUrl, setBackendUrl] = useState(getBackendUrl());
  
  // Redaction types (fetched from backend)
  const [allRedactionTypes, setAllRedactionTypes] = useState([]);
  const [selectedRedactionTypes, setSelectedRedactionTypes] = useState([]);
  const [showTypeSelector, setShowTypeSelector] = useState(false);

  // PDF Preview states
  const [pdfPreviewUrl, setPdfPreviewUrl] = useState(null);
  const [numPages, setNumPages] = useState(null);
  const [pageNumber, setPageNumber] = useState(1);

  // Progress and summary states
  const [progressMessage, setProgressMessage] = useState('');
  const [redactionSummary, setRedactionSummary] = useState(null);

  const fileInputRef = useRef(null);

  // Check backend connectivity
  const checkBackendConnection = async (url) => {
    try {
      const response = await fetch(`${url}/health`, { 
        method: 'GET',
        timeout: 5000 
      });
      return response.ok;
    } catch (error) {
      console.log(`Backend not reachable at ${url}:`, error.message);
      return false;
    }
  };

  // Fetch redaction types from backend
  const fetchRedactionTypes = async () => {
    setBackendStatus('checking');
    
    const possibleUrls = [
      'http://localhost:8000',
      'http://127.0.0.1:8000',
      'http://0.0.0.0:8000'
    ];

    let connectedUrl = null;
    
    // Try each URL until we find one that works
    for (const url of possibleUrls) {
      try {
        console.log(`Trying to connect to backend at: ${url}`);
        const response = await fetch(`${url}/redaction-types`, {
          method: 'GET',
          headers: {
            'Accept': 'application/json',
          },
        });
        
        if (response.ok) {
          const data = await response.json();
          const types = data.redaction_types || [];
          setAllRedactionTypes(types);
          setSelectedRedactionTypes(types); // Select all by default
          connectedUrl = url;
          setBackendUrl(url);
          setBackendStatus('connected');
          setMessage(`Connected to backend at ${url}`);
          console.log(`Successfully connected to backend at: ${url}`);
          break;
        }
      } catch (error) {
        console.log(`Failed to connect to ${url}:`, error.message);
        continue;
      }
    }

    if (!connectedUrl) {
      setBackendStatus('disconnected');
      setMessage(`Error: Could not connect to backend server. Please ensure the server is running on one of: ${possibleUrls.join(', ')}`);
      
      // Provide fallback redaction types for demo purposes
      const fallbackTypes = [
        'PHONE_NUMBER', 'EMAIL_ADDRESS', 'PERSON', 'CREDIT_CARD', 
        'SSN', 'LOCATION', 'ORGANIZATION', 'IP_ADDRESS', 'URL'
      ];
      setAllRedactionTypes(fallbackTypes);
      setSelectedRedactionTypes(fallbackTypes);
    }
  };

  useEffect(() => {
    fetchRedactionTypes();
  }, []);

  // Cleanup blob URL on unmount
  useEffect(() => {
    return () => {
      if (pdfPreviewUrl) {
        URL.revokeObjectURL(pdfPreviewUrl);
      }
    };
  }, [pdfPreviewUrl]);

  // Reset state helper
  const resetState = () => {
    setSelectedFile(null);
    setMessage(backendStatus === 'connected' ? `Connected to backend at ${backendUrl}` : '');
    if (pdfPreviewUrl) {
      URL.revokeObjectURL(pdfPreviewUrl);
    }
    setPdfPreviewUrl(null);
    setNumPages(null);
    setPageNumber(1);
    setProgressMessage('');
    setRedactionSummary(null);
  };

  // File input handler with validation
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    
    // Reset previous state
    resetState();

    if (!file) return;

    // Frontend validation
    if (!ALLOWED_FILE_TYPES.includes(file.type)) {
      setMessage('Error: Invalid file type. Please select a PDF file.');
      if (fileInputRef.current) fileInputRef.current.value = '';
      return;
    }

    if (file.size > MAX_FILE_SIZE_BYTES) {
      setMessage(`Error: File is too large. Maximum size is ${MAX_FILE_SIZE_MB}MB.`);
      if (fileInputRef.current) fileInputRef.current.value = '';
      return;
    }

    // File is valid
    setSelectedFile(file);
    setMessage(`Selected: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)}MB)`);
    
    // Create preview URL
    const previewUrl = URL.createObjectURL(file);
    setPdfPreviewUrl(previewUrl);
    setPageNumber(1);
  };

  // PDF preview handlers
  const onDocumentLoadSuccess = (totalPages) => {
    setNumPages(totalPages);
    setMessage(prev => prev + ` - ${totalPages} pages loaded`);
  };

  const goToNextPage = () => setPageNumber(prev => Math.min(prev + 1, numPages));
  const goToPrevPage = () => setPageNumber(prev => Math.max(prev - 1, 1));

  // Redaction type handlers
  const handleTypeChange = (type) => {
    setSelectedRedactionTypes(prevTypes => {
      if (prevTypes.includes(type)) {
        return prevTypes.filter(t => t !== type);
      } else {
        return [...prevTypes, type];
      }
    });
  };

  const handleSelectAll = () => {
    if (selectedRedactionTypes.length === allRedactionTypes.length) {
      setSelectedRedactionTypes([]);
    } else {
      setSelectedRedactionTypes(allRedactionTypes);
    }
  };

  // Download helpers
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

  // Simulate progress updates
  const simulateProgress = () => {
    const progressSteps = [
      'Initializing redaction process...',
      'Uploading file to server...',
      'Analyzing document structure...',
      'Detecting sensitive information...',
      numPages ? `Processing page 1 of ${numPages}...` : 'Processing document...',
      numPages && numPages > 1 ? `Processing page 2 of ${numPages}...` : null,
      numPages && numPages > 2 ? `Processing remaining pages...` : null,
      'Applying redactions...',
      'Finalizing document...',
      'Preparing download...'
    ].filter(Boolean);

    let stepIndex = 0;
    const interval = setInterval(() => {
      if (stepIndex < progressSteps.length) {
        setProgressMessage(progressSteps[stepIndex]);
        stepIndex++;
      } else {
        clearInterval(interval);
      }
    }, 1000);

    return interval;
  };

  // Generate mock redaction summary
  const generateRedactionSummary = () => {
    const summaryItems = {
      'Phone Numbers': Math.floor(Math.random() * 6),
      'Email Addresses': Math.floor(Math.random() * 4),
      'Names (Person)': Math.floor(Math.random() * 10),
      'Credit Card Numbers': Math.floor(Math.random() * 3),
      'Social Security Numbers': Math.floor(Math.random() * 2),
      'Addresses': Math.floor(Math.random() * 5),
      'Organizations': Math.floor(Math.random() * 4)
    };
    
    // Remove items with 0 count
    return Object.fromEntries(
      Object.entries(summaryItems).filter(([key, value]) => value > 0)
    );
  };

  // Main upload handler
  const handleUpload = async () => {
    if (!selectedFile) {
      setMessage('Please select a PDF file to upload.');
      return;
    }
    if (selectedRedactionTypes.length === 0) {
      setMessage('Please select at least one redaction type.');
      return;
    }

    if (backendStatus === 'disconnected') {
      setMessage('Error: Backend server is not available. Please start the backend server and refresh the page.');
      return;
    }

    setIsLoading(true);
    setRedactionSummary(null);
    setMessage('');

    // Start progress simulation
    const progressInterval = simulateProgress();

    const formData = new FormData();
    formData.append('file', selectedFile);

    const queryParams = new URLSearchParams();
    selectedRedactionTypes.forEach(type => queryParams.append('redaction_types', type));
    
    const requestUrl = `${backendUrl}/redact-pdf/?${queryParams.toString()}`;

    try {
      console.log('Making request to:', requestUrl);
      console.log('Selected redaction types:', selectedRedactionTypes);
      
      const response = await fetch(requestUrl, {
        method: 'POST',
        body: formData,
      });

      console.log('Response status:', response.status);
      console.log('Response headers:', Object.fromEntries(response.headers.entries()));

      clearInterval(progressInterval);
      setProgressMessage('');

      if (response.ok) {
        const contentDisposition = response.headers.get('Content-Disposition');
        const filename = extractFilenameFromHeader(contentDisposition);
        const blob = await response.blob();
        
        triggerDownload(blob, filename);
        setMessage(`Success! Your redacted file '${filename}' has been downloaded.`);
        
        // Generate and show redaction summary
        const summary = generateRedactionSummary();
        setRedactionSummary(summary);
        
      } else {
        const errorData = await response.json().catch(() => ({ 
          detail: `Server responded with status ${response.status}` 
        }));
        setMessage(`Error: ${errorData.detail}`);
      }
    } catch (error) {
      clearInterval(progressInterval);
      setProgressMessage('');
      console.error('Network error:', error);
      setMessage(`Network Error: ${error.message}. Please ensure the backend server is running at ${backendUrl}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center justify-start p-4 font-sans">
      {/* Backend Status Indicator */}
      <div className="w-full max-w-md mb-4">
        <div className={`p-3 rounded-lg text-sm font-medium text-center ${
          backendStatus === 'connected' ? 'bg-green-100 text-green-800' :
          backendStatus === 'disconnected' ? 'bg-red-100 text-red-800' :
          'bg-yellow-100 text-yellow-800'
        }`}>
          {backendStatus === 'connected' && `✓ Backend Connected (${backendUrl})`}
          {backendStatus === 'disconnected' && '✗ Backend Disconnected'}
          {backendStatus === 'checking' && '⏳ Checking Backend Connection...'}
        </div>
      </div>

      {/* Main Upload Section */}
      <div className="bg-white p-8 rounded-xl shadow-lg w-full max-w-md text-center">
        <h1 className="text-4xl font-bold text-gray-800 mb-6">DocuShield AI</h1>
        <p className="text-gray-600 mb-8">Securely redact sensitive information from your PDFs.</p>

        {/* File Input */}
        <div className="mb-6">
          <input 
            id="pdf-upload" 
            ref={fileInputRef} 
            type="file" 
            accept=".pdf,application/pdf" 
            onChange={handleFileChange}
            className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
          />
          <p className="text-xs text-gray-500 mt-2">Max file size: {MAX_FILE_SIZE_MB}MB</p>
        </div>

        {/* Redaction Type Selector */}
        <button 
          onClick={() => setShowTypeSelector(!showTypeSelector)}
          className="w-full mb-4 px-4 py-2 rounded-lg font-semibold text-sm bg-gray-200 text-gray-700 hover:bg-gray-300 transition-colors"
        >
          {showTypeSelector ? 'Hide Redaction Options' : 'Show Redaction Options'} 
          ({selectedRedactionTypes.length} / {allRedactionTypes.length} selected)
        </button>

        {showTypeSelector && (
          <div className="border rounded-lg p-4 mb-6 text-left max-h-60 overflow-y-auto">
            <div className="flex justify-between items-center mb-3 pb-2 border-b">
              <h3 className="text-lg font-semibold text-gray-800">Choose Types</h3>
              <button 
                onClick={handleSelectAll} 
                className="text-blue-600 hover:text-blue-800 text-sm font-medium"
              >
                {selectedRedactionTypes.length === allRedactionTypes.length ? 'Deselect All' : 'Select All'}
              </button>
            </div>
            {allRedactionTypes.map(type => (
              <div key={type} className="flex items-center mb-2">
                <input 
                  type="checkbox" 
                  id={`type-${type}`} 
                  checked={selectedRedactionTypes.includes(type)} 
                  onChange={() => handleTypeChange(type)}
                  className="h-4 w-4 text-blue-600 rounded focus:ring-blue-500"
                />
                <label htmlFor={`type-${type}`} className="ml-2 text-gray-700 text-sm cursor-pointer">
                  {type.replace(/_/g, ' ').replace('PRESIDIO', 'Presidio:').replace('REGEX', 'Regex:').replace('SPACY', 'SpaCy:')}
                </label>
              </div>
            ))}
          </div>
        )}

        {/* Upload Button */}
        <button 
          onClick={handleUpload} 
          disabled={!selectedFile || isLoading || selectedRedactionTypes.length === 0 || backendStatus === 'disconnected'}
          className={`w-full px-4 py-3 rounded-xl font-semibold text-lg transition-all ${
            !selectedFile || isLoading || selectedRedactionTypes.length === 0 || backendStatus === 'disconnected'
              ? 'bg-gray-400 cursor-not-allowed text-gray-600' 
              : 'bg-blue-600 hover:bg-blue-700 text-white shadow-md'
          } ${isLoading ? 'animate-pulse' : ''}`}
        >
          {isLoading ? (progressMessage || 'Processing...') : 'Redact & Download'}
        </button>

        {/* Messages */}
        {message && (
          <div className="mt-6">
            <p className={`p-3 rounded-lg text-sm font-medium ${
              message.toLowerCase().includes('error') 
                ? 'bg-red-100 text-red-800' 
                : message.toLowerCase().includes('success')
                ? 'bg-green-100 text-green-800'
                : 'bg-blue-100 text-blue-800'
            }`}>
              {message}
            </p>
          </div>
        )}

        {/* Redaction Summary */}
        {redactionSummary && Object.keys(redactionSummary).length > 0 && (
          <div className="mt-4 p-4 bg-blue-50 rounded-lg text-left">
            <h3 className="text-md font-semibold text-blue-800 mb-2">Redaction Summary</h3>
            <ul className="text-sm text-blue-700 space-y-1">
              {Object.entries(redactionSummary).map(([type, count]) => (
                <li key={type} className="flex justify-between">
                  <span>{type}:</span>
                  <span className="font-medium">{count} redacted</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {redactionSummary && Object.keys(redactionSummary).length === 0 && (
          <div className="mt-4 p-4 bg-yellow-50 rounded-lg">
            <p className="text-sm text-yellow-800">
              No sensitive information of the selected types was found in your document.
            </p>
          </div>
        )}
      </div>

      {/* PDF Preview Section */}
      {selectedFile && (
        <div className="mt-8 bg-white p-6 rounded-xl shadow-lg w-full max-w-4xl">
          <h2 className="text-2xl font-bold text-gray-800 mb-4 text-center">PDF Preview</h2>
          
          {/* Navigation Controls */}
          <div className="flex justify-center items-center mb-4 space-x-4">
            <button 
              onClick={goToPrevPage} 
              disabled={pageNumber <= 1} 
              className="px-4 py-2 rounded-lg bg-blue-500 text-white hover:bg-blue-600 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
            >
              Previous
            </button>
            <span className="text-lg font-medium text-gray-700 min-w-[120px] text-center">
              Page {pageNumber} of {numPages || '...'}
            </span>
            <button 
              onClick={goToNextPage} 
              disabled={!numPages || pageNumber >= numPages} 
              className="px-4 py-2 rounded-lg bg-blue-500 text-white hover:bg-blue-600 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
            >
              Next
            </button>
          </div>

          {/* PDF Viewer */}
          <div className="min-h-[400px]">
            <PDFViewer
              file={selectedFile}
              currentPage={pageNumber}
              totalPages={numPages}
              onPageChange={setPageNumber}
              onLoadSuccess={onDocumentLoadSuccess}
            />
          </div>
        </div>
      )}

      {/* Retry Backend Connection */}
      {backendStatus === 'disconnected' && (
        <div className="mt-6 bg-white p-4 rounded-lg shadow-md w-full max-w-md text-center">
          <button 
            onClick={fetchRedactionTypes}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            Retry Backend Connection
          </button>
        </div>
      )}

      {/* Footer */}
      <footer className="mt-10 mb-6 text-gray-500 text-sm text-center">
        &copy; {new Date().getFullYear()} DocuShield AI. Secure document redaction.
      </footer>
    </div>
  );
};

export default App;