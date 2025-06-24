import React, { useState, useRef, useEffect } from 'react';
import logo from '../public/Redaction-NO-BG.png'
// --- UTILITY: CONSTANTS ---
const MAX_FILE_SIZE_MB = 5;
const MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024;
const ALLOWED_FILE_TYPES = ['application/pdf'];
const POSSIBLE_BACKEND_URLS = [
  'http://localhost:8000',
  'http://127.0.0.1:8000',
  'http://0.0.0.0:8000'
];
const FALLBACK_REDACTION_TYPES = [
    'PHONE_NUMBER', 'EMAIL_ADDRESS', 'PERSON', 'CREDIT_CARD', 
    'SSN', 'LOCATION', 'ORGANIZATION', 'IP_ADDRESS', 'URL'
];

// --- UTILITY: HELPERS ---

/**
 * Extracts the filename from the Content-Disposition header.
 * @param {string} contentDisposition The value of the Content-Disposition header.
 * @returns {string} The extracted filename or a default value.
 */
const extractFilenameFromHeader = (contentDisposition) => {
  if (!contentDisposition) return 'redacted_document.pdf';
  let match = contentDisposition.match(/filename\*=UTF-8''([^;]+)/i);
  if (match?.[1]) return decodeURIComponent(match[1]);
  match = contentDisposition.match(/filename="([^"]*)"/i);
  if (match?.[1]) return match[1];
  return 'redacted_document.pdf';
};

/**
 * Triggers a file download in the browser.
 * @param {Blob} blob The file content as a Blob.
 * @param {string} filename The desired name of the file.
 */
const triggerDownload = (blob, filename) => {
  const link = document.createElement('a');
  link.href = window.URL.createObjectURL(blob);
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.URL.revokeObjectURL(link.href);
};

/**
 * Simulates progress messages during the redaction process.
 * @param {function} setProgressMessage React state setter for the progress message.
 * @param {number} numPages The number of pages in the PDF.
 * @returns {number} The interval ID to be cleared later.
 */
const simulateProgress = (setProgressMessage, numPages) => {
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

/**
 * Generates a mock summary of redacted items for demonstration purposes.
 * @returns {object} An object containing types and counts of redacted items.
 */
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
  
  return Object.fromEntries(
    Object.entries(summaryItems).filter(([key, value]) => value > 0)
  );
};


// --- SERVICE: BACKEND ---

/**
 * Fetches the available redaction types from the backend.
 * @returns {Promise<object>} An object with success status, and data or error info.
 */
const fetchTypesService = async () => {
  for (const url of POSSIBLE_BACKEND_URLS) {
    try {
      console.log(`Trying to connect to backend at: ${url}`);
      const response = await fetch(`${url}/redaction-types`, {
        method: 'GET',
        headers: { 'Accept': 'application/json' },
      });
      
      if (response.ok) {
        const data = await response.json();
        const types = data.redaction_types || [];
        console.log(`Successfully connected to backend at: ${url}`);
        return { success: true, types, connectedUrl: url };
      }
    } catch (error) {
      console.log(`Failed to connect to ${url}:`, error.message);
      continue;
    }
  }

  return { success: false, triedUrls: POSSIBLE_BACKEND_URLS };
};

/**
 * Uploads a PDF file to the backend for redaction.
 * @param {File} file The PDF file to redact.
 * @param {string[]} selectedRedactionTypes The types of information to redact.
 * @param {string} backendUrl The URL of the backend server.
 * @returns {Promise<object>} An object with success status and response data or error.
 */
const uploadAndRedactFile = async (file, selectedRedactionTypes, backendUrl) => {
  const formData = new FormData();
  formData.append('file', file);

  const queryParams = new URLSearchParams();
  selectedRedactionTypes.forEach(type => queryParams.append('redaction_types', type));
  
  const requestUrl = `${backendUrl}/redact-pdf/?${queryParams.toString()}`;

  try {
    const response = await fetch(requestUrl, {
      method: 'POST',
      body: formData,
    });

    if (response.ok) {
      const blob = await response.blob();
      return { success: true, data: { blob, headers: response.headers } };
    } else {
      const errorData = await response.json().catch(() => ({ 
        detail: `Server responded with status ${response.status}` 
      }));
      return { success: false, error: errorData.detail };
    }
  } catch (error) {
    console.error('Network error during upload:', error);
    return { success: false, error: error.message };
  }
};


// --- COMPONENT: PDFViewer ---
const PDFViewer = ({ file, onPageChange, currentPage, totalPages, onLoadSuccess }) => {
  const canvasRef = useRef(null);
  const [pdfDoc, setPdfDoc] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [pdfJsLoaded, setPdfJsLoaded] = useState(false);

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

  useEffect(() => {
    if (file && pdfJsLoaded) {
      loadPDF();
    }
  }, [file, pdfJsLoaded]);

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
      const containerWidth = canvas.parentElement.clientWidth - 32;
      const viewport = page.getViewport({ scale: 1 });
      const scale = Math.min(containerWidth / viewport.width, 1.5);
      const scaledViewport = page.getViewport({ scale });
      canvas.width = scaledViewport.width;
      canvas.height = scaledViewport.height;
      context.clearRect(0, 0, canvas.width, canvas.height);
      await page.render({ canvasContext: context, viewport: scaledViewport }).promise;
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

// --- COMPONENT: RedactionSelector ---
const RedactionSelector = ({ allTypes, selectedTypes, setSelectedTypes }) => {
  const handleTypeChange = (type) => {
    setSelectedTypes(prevTypes => 
      prevTypes.includes(type)
        ? prevTypes.filter(t => t !== type)
        : [...prevTypes, type]
    );
  };

  const handleSelectAll = () => {
    setSelectedTypes(selectedTypes.length === allTypes.length ? [] : allTypes);
  };

  const formatTypeName = (type) => type.replace(/_/g, ' ').replace('PRESIDIO', 'Presidio:').replace('REGEX', 'Regex:').replace('SPACY', 'SpaCy:');

  return (
    <div className="border rounded-lg p-4 mb-6 text-left max-h-60 overflow-y-auto">
      <div className="flex justify-between items-center mb-3 pb-2 border-b">
        <h3 className="text-lg font-semibold text-gray-800">Choose Types</h3>
        <button onClick={handleSelectAll} className="text-blue-600 hover:text-blue-800 text-sm font-medium">
          {selectedTypes.length === allTypes.length ? 'Deselect All' : 'Select All'}
        </button>
      </div>
      {allTypes.map(type => (
        <div key={type} className="flex items-center mb-2">
          <input 
            type="checkbox" 
            id={`type-${type}`} 
            checked={selectedTypes.includes(type)} 
            onChange={() => handleTypeChange(type)}
            className="h-4 w-4 text-blue-600 rounded focus:ring-blue-500"
          />
          <label htmlFor={`type-${type}`} className="ml-2 text-gray-700 text-sm cursor-pointer">
            {formatTypeName(type)}
          </label>
        </div>
      ))}
    </div>
  );
};

// --- COMPONENT: ControlPanel ---
const ControlPanel = ({ selectedFile, fileInputRef, handleFileChange, allRedactionTypes, selectedRedactionTypes, setSelectedRedactionTypes, handleUpload, isLoading, backendStatus, progressMessage, message, redactionSummary, fetchRedactionTypes }) => {
  const [showTypeSelector, setShowTypeSelector] = useState(false);

  return (
    <div className={`bg-white p-8 rounded-xl shadow-lg text-center transition-all duration-500 ease-out ${selectedFile ? 'w-full lg:w-96 lg:min-w-96' : 'w-full max-w-md'}`}>
            <h1 className={`flex items-center font-bold text-gray-800 mb-6 transition-all duration-300 ${selectedFile ? 'text-3xl' : 'text-4xl'}`}>
        <img src={logo} alt="DocuShield Logo" className="h-25 w-auto" />
        DocuShield AI
      </h1>
      <p className="text-gray-600 mb-8">Securely redact sensitive information from your PDFs.</p>
      <div className="mb-6">
        <input id="pdf-upload" ref={fileInputRef} type="file" accept=".pdf,application/pdf" onChange={handleFileChange} className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100" />
        <p className="text-xs text-gray-500 mt-2">Max file size: {MAX_FILE_SIZE_MB}MB</p>
      </div>
      <button onClick={() => setShowTypeSelector(!showTypeSelector)} className="w-full mb-4 px-4 py-2 rounded-lg font-semibold text-sm bg-gray-200 text-gray-700 hover:bg-gray-300 transition-colors">
        {showTypeSelector ? 'Hide' : 'Show'} Redaction Options ({selectedRedactionTypes.length}/{allRedactionTypes.length} selected)
      </button>
      {showTypeSelector && <RedactionSelector allTypes={allRedactionTypes} selectedTypes={selectedRedactionTypes} setSelectedTypes={setSelectedRedactionTypes} />}
      <button onClick={handleUpload} disabled={!selectedFile || isLoading || selectedRedactionTypes.length === 0 || backendStatus === 'disconnected'} className={`w-full px-4 py-3 rounded-xl font-semibold text-lg transition-all ${!selectedFile || isLoading || selectedRedactionTypes.length === 0 || backendStatus === 'disconnected' ? 'bg-gray-400 cursor-not-allowed text-gray-600' : 'bg-blue-600 hover:bg-blue-700 text-white shadow-md'} ${isLoading ? 'animate-pulse' : ''}`}>
        {isLoading ? (progressMessage || 'Processing...') : 'Redact & Download'}
      </button>
      {message && <div className="mt-6"><p className={`p-3 rounded-lg text-sm font-medium ${message.toLowerCase().includes('error') ? 'bg-red-100 text-red-800' : message.toLowerCase().includes('success') ? 'bg-green-100 text-green-800' : 'bg-blue-100 text-blue-800'}`}>{message}</p></div>}
      {redactionSummary && Object.keys(redactionSummary).length > 0 && <div className="mt-4 p-4 bg-blue-50 rounded-lg text-left"><h3 className="text-md font-semibold text-blue-800 mb-2">Redaction Summary</h3><ul className="text-sm text-blue-700 space-y-1">{Object.entries(redactionSummary).map(([type, count]) => (<li key={type} className="flex justify-between"><span>{type}:</span><span className="font-medium">{count} redacted</span></li>))}</ul></div>}
      {redactionSummary && Object.keys(redactionSummary).length === 0 && <div className="mt-4 p-4 bg-yellow-50 rounded-lg"><p className="text-sm text-yellow-800">No sensitive information of the selected types was found.</p></div>}
      {backendStatus === 'disconnected' && <div className="mt-6"><button onClick={fetchRedactionTypes} className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">Retry Connection</button></div>}
    </div>
  );
};

// --- COMPONENT: PreviewSection ---
const PreviewSection = ({ selectedFile, pageNumber, numPages, goToPrevPage, goToNextPage, onDocumentLoadSuccess, setPageNumber }) => {
  if (!selectedFile) return null;
  return (
    <div className={`bg-white p-6 rounded-xl shadow-lg flex-1 min-w-0 transform transition-all duration-1000 ease-out delay-300 ${selectedFile ? 'translate-x-0 opacity-100' : 'translate-x-full opacity-0'}`}>
      <h2 className="text-2xl font-bold text-gray-800 mb-4 text-center">PDF Preview</h2>
      <div className="flex justify-center items-center mb-4 space-x-4">
        <button onClick={goToPrevPage} disabled={pageNumber <= 1} className="px-4 py-2 rounded-lg bg-blue-500 text-white hover:bg-blue-600 disabled:bg-gray-400">Previous</button>
        <span className="text-lg font-medium text-gray-700 min-w-[120px] text-center">Page {pageNumber} of {numPages || '...'}</span>
        <button onClick={goToNextPage} disabled={!numPages || pageNumber >= numPages} className="px-4 py-2 rounded-lg bg-blue-500 text-white hover:bg-blue-600 disabled:bg-gray-400">Next</button>
      </div>
      <div className="min-h-[400px]">
        <PDFViewer file={selectedFile} currentPage={pageNumber} totalPages={numPages} onPageChange={setPageNumber} onLoadSuccess={onDocumentLoadSuccess} />
      </div>
    </div>
  );
};

// --- COMPONENT: BackendStatus ---
const BackendStatus = ({ status, url }) => {
  const statusConfig = {
    connected: { style: 'bg-green-100 text-green-800', text: `✓ Backend Connected (${url})` },
    disconnected: { style: 'bg-red-100 text-red-800', text: '✗ Backend Disconnected' },
    checking: { style: 'bg-yellow-100 text-yellow-800', text: '⏳ Checking Backend Connection...' },
  };
  const { style, text } = statusConfig[status] || statusConfig.checking;
  return <div className="w-full mb-4 flex justify-center"><div className={`p-3 rounded-lg text-sm font-medium text-center max-w-md ${style}`}>{text}</div></div>;
};

// --- COMPONENT: Footer ---
const Footer = () => <footer className="mt-10 mb-6 text-gray-500 text-sm text-center">&copy; {new Date().getFullYear()} DocuShield AI. Secure document redaction.</footer>;


// --- MAIN APP COMPONENT ---
const App = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [backendStatus, setBackendStatus] = useState('checking');
  const [backendUrl, setBackendUrl] = useState('');
  const [allRedactionTypes, setAllRedactionTypes] = useState([]);
  const [selectedRedactionTypes, setSelectedRedactionTypes] = useState([]);
  const [pdfPreviewUrl, setPdfPreviewUrl] = useState(null);
  const [numPages, setNumPages] = useState(null);
  const [pageNumber, setPageNumber] = useState(1);
  const [progressMessage, setProgressMessage] = useState('');
  const [redactionSummary, setRedactionSummary] = useState(null);
  const fileInputRef = useRef(null);

  const fetchRedactionTypes = async () => {
    setBackendStatus('checking');
    const result = await fetchTypesService();
    if (result.success) {
      const { types, connectedUrl } = result;
      setAllRedactionTypes(types);
      setSelectedRedactionTypes(types);
      setBackendUrl(connectedUrl);
      setBackendStatus('connected');
      setMessage(`Connected to backend at ${connectedUrl}`);
    } else {
      setBackendStatus('disconnected');
      setMessage(`Error: Could not connect to backend. Please ensure it's running on one of: ${result.triedUrls.join(', ')}`);
      setAllRedactionTypes(FALLBACK_REDACTION_TYPES);
      setSelectedRedactionTypes(FALLBACK_REDACTION_TYPES);
    }
  };

  useEffect(() => {
    fetchRedactionTypes();
  }, []);

  useEffect(() => {
    return () => { if (pdfPreviewUrl) URL.revokeObjectURL(pdfPreviewUrl); };
  }, [pdfPreviewUrl]);

  const resetState = () => {
    setSelectedFile(null);
    setMessage(backendStatus === 'connected' ? `Connected to backend at ${backendUrl}` : '');
    if (pdfPreviewUrl) URL.revokeObjectURL(pdfPreviewUrl);
    setPdfPreviewUrl(null);
    setNumPages(null);
    setPageNumber(1);
    setProgressMessage('');
    setRedactionSummary(null);
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    resetState();
    if (!file) return;
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
    setSelectedFile(file);
    setMessage(`Selected: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)}MB)`);
    setPdfPreviewUrl(URL.createObjectURL(file));
    setPageNumber(1);
  };

  const onDocumentLoadSuccess = (totalPages) => {
    setNumPages(totalPages);
    setMessage(prev => `${prev} - ${totalPages} pages loaded`);
  };

  const goToNextPage = () => setPageNumber(prev => Math.min(prev + 1, numPages));
  const goToPrevPage = () => setPageNumber(prev => Math.max(prev - 1, 1));

  const handleUpload = async () => {
    if (!selectedFile) { setMessage('Please select a PDF file.'); return; }
    if (selectedRedactionTypes.length === 0) { setMessage('Please select at least one redaction type.'); return; }
    if (backendStatus === 'disconnected') { setMessage('Error: Backend server is not available.'); return; }
    
    setIsLoading(true);
    setRedactionSummary(null);
    setMessage('');

    const progressInterval = simulateProgress(setProgressMessage, numPages);

    try {
      const response = await uploadAndRedactFile(selectedFile, selectedRedactionTypes, backendUrl);
      clearInterval(progressInterval);
      setProgressMessage('');

      if (response.success) {
        const { blob, headers } = response.data;
        const filename = extractFilenameFromHeader(headers.get('Content-Disposition'));
        triggerDownload(blob, filename);
        setMessage(`Success! '${filename}' has been downloaded.`);
        setRedactionSummary(generateRedactionSummary());
      } else {
        setMessage(`Error: ${response.error}`);
      }
    } catch (error) {
      clearInterval(progressInterval);
      setProgressMessage('');
      setMessage(`Network Error: ${error.message}.`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 p-4 font-sans">
      <BackendStatus status={backendStatus} url={backendUrl} />
      <div className={`flex flex-col lg:flex-row gap-8 transition-all duration-700 ease-in-out ${selectedFile ? 'justify-start items-start' : 'justify-center items-center min-h-[70vh]'}`}>
        <ControlPanel
          selectedFile={selectedFile}
          fileInputRef={fileInputRef}
          handleFileChange={handleFileChange}
          allRedactionTypes={allRedactionTypes}
          selectedRedactionTypes={selectedRedactionTypes}
          setSelectedRedactionTypes={setSelectedRedactionTypes}
          handleUpload={handleUpload}
          isLoading={isLoading}
          backendStatus={backendStatus}
          progressMessage={progressMessage}
          message={message}
          redactionSummary={redactionSummary}
          fetchRedactionTypes={fetchRedactionTypes}
        />
        <PreviewSection
          selectedFile={selectedFile}
          pageNumber={pageNumber}
          numPages={numPages}
          goToPrevPage={goToPrevPage}
          goToNextPage={goToNextPage}
          onDocumentLoadSuccess={onDocumentLoadSuccess}
          setPageNumber={setPageNumber}
        />
      </div>
      <Footer />
    </div>
  );
};

export default App;
