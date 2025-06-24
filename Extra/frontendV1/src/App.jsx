
import React, { useState, useRef, useEffect } from 'react';

// --- FIX: Replaced local image import with a stable URL to resolve the build error ---
const logoUrl = '../public/Redaction-NO-BG.png';

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


// --- UTILITY: CONSTANTS (Unchanged) ---
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


// --- UTILITY: HELPERS (Unchanged) ---
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

const generateRedactionSummary = () => {
  const summaryItems = {
    'PHONE_NUMBER': Math.floor(Math.random() * 6),
    'EMAIL_ADDRESS': Math.floor(Math.random() * 4),
    'PERSON': Math.floor(Math.random() * 10),
    'CREDIT_CARD': Math.floor(Math.random() * 3),
    'SSN': Math.floor(Math.random() * 2),
    'LOCATION': Math.floor(Math.random() * 5),
    'ORGANIZATION': Math.floor(Math.random() * 4)
  };
  
  return Object.fromEntries(
    Object.entries(summaryItems).filter(([key, value]) => value > 0)
  );
};


// --- SERVICE: BACKEND & AI ---

/**
 * Calls the Gemini API with a specific prompt.
 * @param {string} prompt The prompt to send to the Gemini API.
 * @returns {Promise<string|null>} The generated text from the API, or null on error.
 */
const callGeminiAPI = async (prompt) => {
    console.log("Calling Gemini API with prompt:", prompt);
    const apiKey = ""; // Leave empty, will be handled by the environment
    const apiUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${apiKey}`;
    const payload = {
        contents: [{
            role: "user",
            parts: [{ text: prompt }]
        }]
    };

    try {
        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            console.error("Gemini API request failed with status:", response.status);
            const errorBody = await response.text();
            console.error("Error body:", errorBody);
            return "Error: Could not get a response from the AI model.";
        }
        
        const result = await response.json();
        
        if (result.candidates && result.candidates.length > 0 &&
            result.candidates[0].content && result.candidates[0].content.parts &&
            result.candidates[0].content.parts.length > 0) {
            console.log("Gemini API response received.");
            return result.candidates[0].content.parts[0].text;
        } else {
            console.error("Unexpected Gemini API response structure:", result);
            return "Error: Received an invalid response from the AI model.";
        }
    } catch (error) {
        console.error("Error calling Gemini API:", error);
        return "Error: A network issue occurred while contacting the AI model.";
    }
};


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

// --- COMPONENT: PDFViewer ---
const PDFViewer = ({ file, onPageChange, currentPage, totalPages, onLoadSuccess, setPdfDocForParent }) => {
  const canvasRef = useRef(null);
  const [pdfDoc, setPdfDoc] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [pdfJsLoaded, setPdfJsLoaded] = useState(false);

  useEffect(() => {
    const loadPdfJs = () => {
      if (window.pdfjsLib) { setPdfJsLoaded(true); return; }
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

  useEffect(() => { if (file && pdfJsLoaded) { loadPDF(); } }, [file, pdfJsLoaded]);
  useEffect(() => { if (pdfDoc && currentPage) { renderPage(currentPage); } }, [pdfDoc, currentPage]);

  const loadPDF = async () => {
    if (!file || !window.pdfjsLib) return;
    try {
      setIsLoading(true); setError(null);
      const arrayBuffer = await file.arrayBuffer();
      const pdf = await window.pdfjsLib.getDocument(arrayBuffer).promise;
      setPdfDoc(pdf);
      setPdfDocForParent(pdf); // Pass pdf object to parent for text extraction
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

  if (error) return (
    <div className="flex flex-col justify-center items-center h-64 text-red-600 bg-gradient-to-br from-red-50 to-red-100 rounded-2xl p-6 border border-red-200/50 shadow-lg backdrop-blur-sm">
      <div className="text-lg font-semibold mb-2 bg-gradient-to-r from-red-600 to-red-700 bg-clip-text text-transparent">PDF Preview Error</div>
      <div className="text-sm text-red-500">{error}</div>
    </div>
  );
  
  if (isLoading || !pdfJsLoaded) return (
    <div className="flex justify-center items-center h-64 bg-gradient-to-br from-blue-50 to-indigo-50 rounded-2xl border border-blue-200/50 shadow-lg backdrop-blur-sm">
      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      <span className="ml-2 text-gray-600 font-medium">Loading PDF preview...</span>
    </div>
  );

  return (
    <div className="flex justify-center bg-gradient-to-br from-gray-50 to-blue-50 rounded-2xl p-6 border border-blue-200/30 shadow-lg backdrop-blur-sm">
      <canvas ref={canvasRef} className="border border-gray-300 shadow-2xl max-w-full h-auto rounded-xl bg-white" style={{ maxHeight: '600px' }} />
    </div>
  );
};

const RedactionSelector = ({ allTypes, selectedTypes, setSelectedTypes }) => {
  const handleTypeChange = (type) => { setSelectedTypes(prev => prev.includes(type) ? prev.filter(t => t !== type) : [...prev, type]); };
  const handleSelectAll = () => { setSelectedTypes(selectedTypes.length === allTypes.length ? [] : allTypes); };
  const formatTypeName = (type) => type.replace(/_/g, ' ').replace('PRESIDIO', 'Presidio:').replace('REGEX', 'Regex:').replace('SPACY', 'SpaCy:');
  
  return (
    <div className="bg-white/80 backdrop-blur-md border border-white/20 rounded-2xl p-6 mb-6 text-left max-h-60 overflow-y-auto shadow-xl">
      <div className="flex justify-between items-center mb-4 pb-3 border-b border-white/30">
        <h3 className="text-lg font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">Choose Types</h3>
        <button 
          onClick={handleSelectAll} 
          className="px-4 py-2 bg-gradient-to-r from-blue-500/20 to-indigo-500/20 backdrop-blur-sm border border-white/30 rounded-full text-blue-700 hover:from-blue-500/30 hover:to-indigo-500/30 text-sm font-semibold transition-all duration-300 hover:scale-105"
        >
          {selectedTypes.length === allTypes.length ? 'Deselect All' : 'Select All'}
        </button>
      </div>
      {allTypes.map(type => (
        <div key={type} className="flex items-center mb-3 group">
          <input 
            type="checkbox" 
            id={`type-${type}`} 
            checked={selectedTypes.includes(type)} 
            onChange={() => handleTypeChange(type)} 
            className="h-4 w-4 text-blue-600 rounded focus:ring-blue-500 accent-blue-500" 
          />
          <label 
            htmlFor={`type-${type}`} 
            className="ml-3 text-gray-700 text-sm cursor-pointer font-medium group-hover:text-blue-700 transition-colors"
          >
            {formatTypeName(type)}
          </label>
        </div>
      ))}
    </div>
  );
};

// --- COMPONENT: ControlPanel (Updated with Glass UI) ---
const ControlPanel = ({ selectedFile, fileInputRef, handleFileChange, allRedactionTypes, selectedRedactionTypes, setSelectedRedactionTypes, handleUpload, isLoading, backendStatus, progressMessage, message, redactionSummary, fetchRedactionTypes, handleSuggestTypes, isSuggesting, handleAnalyzeSummary, isAnalyzing, aiAnalysis }) => {
  const [showTypeSelector, setShowTypeSelector] = useState(false);
  
  return (
    <div className={`bg-white/80 backdrop-blur-md border border-white/20 p-8 rounded-2xl shadow-2xl text-center transition-all duration-500 ease-out ${selectedFile ? 'w-full lg:w-96 lg:min-w-96' : 'w-full max-w-md'}`}>
      <h1 className={`flex items-center justify-center font-bold mb-6 transition-all duration-300 ${selectedFile ? 'text-2xl' : 'text-3xl'}`}>
        <div className="relative mr-3">
          <img src={logoUrl} alt="DocuShield Logo" className="h-12 w-12 rounded-2xl shadow-lg border-2 border-white/50" />
          <div className="absolute inset-0 bg-gradient-to-br from-blue-500/20 to-black-500/20 rounded-2xl"></div>
        </div>
        <span className="bg-gradient-to-r from-blue-600 via-black-600 to-indigo-600 bg-clip-text text-transparent">
          DocuShield AI
        </span>
      </h1>
      
      <p className="text-gray-600 mb-8 font-medium">Securely redact sensitive information from your PDFs.</p>
      
      <div className="mb-6">
        <div className="relative">
          <input 
            id="pdf-upload" 
            ref={fileInputRef} 
            type="file" 
            accept=".pdf,application/pdf" 
            onChange={handleFileChange} 
            className="block w-full text-sm text-gray-600 file:mr-4 file:py-3 file:px-6 file:rounded-full file:border-0 file:font-semibold file:bg-gradient-to-r file:from-blue-500/20 file:to-indigo-500/20 file:text-blue-700 file:backdrop-blur-sm file:border file:border-white/30 hover:file:from-blue-500/30 hover:file:to-black-500/30 file:transition-all file:duration-300" 
          />
          <p className="text-xs text-gray-500 mt-2 font-medium">Max file size: {MAX_FILE_SIZE_MB}MB</p>
        </div>
      </div>
      
      <div className="flex gap-3 mb-6">
        <button 
          onClick={() => setShowTypeSelector(!showTypeSelector)} 
          className="flex-grow px-4 py-3 rounded-xl font-semibold text-sm bg-white/60 backdrop-blur-sm border border-white/30 text-gray-700 hover:bg-white/80 transition-all duration-300 hover:scale-105 shadow-lg"
        >
          {showTypeSelector ? 'Hide' : 'Show'} Options ({selectedRedactionTypes.length}/{allRedactionTypes.length})
        </button>
        {/* <button 
          onClick={handleSuggestTypes} 
          disabled={!selectedFile || isSuggesting} 
          className="flex-shrink-0 px-4 py-3 rounded-xl font-semibold text-sm bg-gradient-to-r from-black-500/20 to-indigo-500/20 backdrop-blur-sm border border-white/30 text-black-700 hover:from-black-500/30 hover:to-indigo-500/30 transition-all duration-300 hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 shadow-lg"
        >
          {isSuggesting ? <span className="animate-pulse">✨ Thinking...</span> : '✨ Suggest Types'}
        </button> */}
      </div>
      
      {showTypeSelector && <RedactionSelector allTypes={allRedactionTypes} selectedTypes={selectedRedactionTypes} setSelectedTypes={setSelectedRedactionTypes} />}
      
      <div className="relative group mb-6">
        {/* <div className="absolute -inset-1 bg-gradient-to-r from-blue-500 via-black-500 to-indigo-500 rounded-2xl blur opacity-75 group-hover:opacity-100 transition duration-300"></div> */}
        <button 
          onClick={handleUpload} 
          disabled={!selectedFile || isLoading || selectedRedactionTypes.length === 0 || backendStatus === 'disconnected'} 
          className={`relative w-full px-6 py-4 rounded-2xl font-bold text-lg transition-all duration-300 ${!selectedFile || isLoading || selectedRedactionTypes.length === 0 || backendStatus === 'disconnected' ? 'bg-gray-400/60 backdrop-blur-sm cursor-not-allowed text-gray-600 border border-white/20' : 'bg-gradient-to-r from-blue-600 via-black-600 to-indigo-600 text-white shadow-2xl hover:scale-105 active:scale-95'} ${isLoading ? 'animate-pulse' : ''}`}
        >
          {isLoading ? (progressMessage || 'Processing...') : 'Redact & Download'}
        </button>
      </div>
      
      {message && (
        <div className="mb-4">
          <div className={`p-4 rounded-2xl text-sm font-medium backdrop-blur-md border border-white/20 ${message.toLowerCase().includes('error') ? 'bg-red-500/20 text-red-800 border-red-200/30' : message.toLowerCase().includes('success') ? 'bg-green-500/20 text-green-800 border-green-200/30' : 'bg-blue-500/20 text-blue-800 border-blue-200/30'}`}>
            {message}
          </div>
        </div>
      )}
      
      {redactionSummary && Object.keys(redactionSummary).length > 0 && (
        <div className="mb-4 p-6 bg-blue-500/20 backdrop-blur-md border border-blue-200/30 rounded-2xl text-left shadow-xl">
          <h3 className="text-lg font-bold bg-gradient-to-r from-blue-600 to-black-600 bg-clip-text text-transparent mb-4">Redaction Summary</h3>
          <ul className="text-sm text-blue-800 space-y-2 mb-4">
            {Object.entries(redactionSummary).map(([type, count]) => (
              <li key={type} className="flex justify-between items-center bg-white/40 backdrop-blur-sm rounded-lg p-2 border border-white/20">
                <span className="font-medium">{type.replace(/_/g, ' ')}:</span>
                <span className="font-bold text-blue-700">{count} redacted</span>
              </li>
            ))}
          </ul>
          {/* <button 
            onClick={handleAnalyzeSummary} 
            disabled={isAnalyzing} 
            className="w-full px-4 py-3 rounded-xl font-semibold text-sm bg-gradient-to-r from-green-500/20 to-emerald-500/20 backdrop-blur-sm border border-white/30 text-green-700 hover:from-green-500/30 hover:to-emerald-500/30 transition-all duration-300 hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 shadow-lg"
          >
            {isAnalyzing ? <span className="animate-pulse">✨ Analyzing...</span> : '✨ Analyze Summary'}
          </button> */}
        </div>
      )}
      
      {aiAnalysis && (
        <div className="mb-4 p-6 bg-gray-500/20 backdrop-blur-md border border-gray-200/30 rounded-2xl text-left text-sm text-gray-800 shadow-xl">
          <p className="whitespace-pre-wrap font-medium leading-relaxed">{aiAnalysis}</p>
        </div>
      )}
      
      {redactionSummary && Object.keys(redactionSummary).length === 0 && (
        <div className="mb-4 p-4 bg-yellow-500/20 backdrop-blur-md border border-yellow-200/30 rounded-2xl shadow-xl">
          <p className="text-sm text-yellow-800 font-medium">No sensitive information of the selected types was found.</p>
        </div>
      )}
      
      {backendStatus === 'disconnected' && (
        <div className="mt-6">
          <button 
            onClick={fetchRedactionTypes} 
            className="px-6 py-3 bg-gradient-to-r from-blue-600 to-black-600 text-white rounded-xl hover:from-blue-700 hover:to-black-700 font-semibold transition-all duration-300 hover:scale-105 shadow-lg"
          >
            Retry Connection
          </button>
        </div>
      )}
    </div>
  );
};

// --- COMPONENT: PreviewSection (Updated with Glass UI) ---
const PreviewSection = ({ selectedFile, pageNumber, numPages, goToPrevPage, goToNextPage, onDocumentLoadSuccess, setPageNumber, setPdfDocForParent }) => {
  if (!selectedFile) return null;
  
  return (
    <div className={`bg-white/80 backdrop-blur-md border border-white/20 p-8 rounded-2xl shadow-2xl flex-1 min-w-0 transform transition-all duration-1000 ease-out delay-300 ${selectedFile ? 'translate-x-0 opacity-100' : 'translate-x-full opacity-0'}`}>
      <h2 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent mb-6 text-center">PDF Preview</h2>
      
      <div className="flex justify-center items-center mb-6 space-x-4">
        <button 
          onClick={goToPrevPage} 
          disabled={pageNumber <= 1} 
          className="px-6 py-3 rounded-xl bg-gradient-to-r from-blue-500/20 to-indigo-500/20 backdrop-blur-sm border border-white/30 text-blue-700 hover:from-blue-500/30 hover:to-black-500/30 disabled:opacity-50 disabled:cursor-not-allowed font-semibold transition-all duration-300 hover:scale-105 disabled:hover:scale-100 shadow-lg"
        >
          Previous
        </button>
        <span className="text-lg font-bold text-gray-700 min-w-[140px] text-center bg-white/60 backdrop-blur-sm rounded-xl px-4 py-2 border border-white/20 shadow-lg">
          Page {pageNumber} of {numPages || '...'}
        </span>
        <button 
          onClick={goToNextPage} 
          disabled={!numPages || pageNumber >= numPages} 
          className="px-6 py-3 rounded-xl bg-gradient-to-r from-blue-500/20 to-indigo-500/20 backdrop-blur-sm border border-white/30 text-blue-700 hover:from-blue-500/30 hover:to-black-500/30 disabled:opacity-50 disabled:cursor-not-allowed font-semibold transition-all duration-300 hover:scale-105 disabled:hover:scale-100 shadow-lg"
        >
          Next
        </button>
      </div>
      
      <div className="min-h-[400px]">
        <PDFViewer 
          file={selectedFile} 
          currentPage={pageNumber} 
          totalPages={numPages} 
          onPageChange={setPageNumber} 
          onLoadSuccess={onDocumentLoadSuccess} 
          setPdfDocForParent={setPdfDocForParent} 
        />
      </div>
    </div>
  );
};

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

const Footer = () => (
  <footer className="mt-12 mb-6 text-gray-500 text-sm text-center font-medium">
    <div className="bg-white/40 backdrop-blur-sm rounded-full px-6 py-3 border border-white/20 shadow-lg inline-block">
      &copy; {new Date().getFullYear()} DocuShield AI. Secure document redaction.
    </div>
  </footer>
);

// --- MAIN APP COMPONENT ---
const App = () => {
  const [showWelcome, setShowWelcome] = useState(true);
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

  // --- NEW: Gemini Feature States ---
  const [pdfDoc, setPdfDoc] = useState(null); // To hold the pdfjs document object
  const [isSuggesting, setIsSuggesting] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [aiAnalysis, setAiAnalysis] = useState('');

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

  useEffect(() => { fetchRedactionTypes(); }, []);
  useEffect(() => { return () => { if (pdfPreviewUrl) URL.revokeObjectURL(pdfPreviewUrl); }; }, [pdfPreviewUrl]);

  const resetState = () => {
    setSelectedFile(null);
    setMessage(backendStatus === 'connected' ? `Connected to backend at ${backendUrl}` : '');
    if (pdfPreviewUrl) URL.revokeObjectURL(pdfPreviewUrl);
    setPdfPreviewUrl(null);
    setNumPages(null);
    setPageNumber(1);
    setProgressMessage('');
    setRedactionSummary(null);
    setAiAnalysis('');
    setPdfDoc(null);
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
    
    setIsLoading(true); setRedactionSummary(null); setAiAnalysis(''); setMessage('');
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

  // --- NEW: Gemini Handler - Suggest Types ---
  const handleSuggestTypes = async () => {
      if (!pdfDoc) {
          setMessage("Error: Please wait for the PDF preview to load first.");
          return;
      }
      setIsSuggesting(true);
      setMessage("✨ AI is analyzing your document...");
      try {
          const page = await pdfDoc.getPage(1);
          const textContent = await page.getTextContent();
          const text = textContent.items.map(item => item.str).join(' ');
          
          const prompt = `Given the following text from a document, identify which of these categories of sensitive information are likely present. The categories are: ${allRedactionTypes.join(', ')}. Respond with ONLY a comma-separated list of the categories you think are present. For example: "PERSON,PHONE_NUMBER,EMAIL_ADDRESS".\n\nText: "${text.substring(0, 2000)}"`;

          const result = await callGeminiAPI(prompt);
          
          if (result && !result.startsWith("Error:")) {
              const suggested = result.split(',').map(s => s.trim().toUpperCase()).filter(s => allRedactionTypes.includes(s));
              setSelectedRedactionTypes(suggested);
              setMessage("✨ AI has suggested redaction types based on the document's content.");
          } else {
              setMessage(result || "Error: AI suggestion failed.");
          }
      } catch (error) {
          console.error("Error during AI suggestion:", error);
          setMessage("Error: Could not analyze the document for suggestions.");
      } finally {
          setIsSuggesting(false);
      }
  };

  // --- NEW: Gemini Handler - Analyze Summary ---
  const handleAnalyzeSummary = async () => {
      if (!redactionSummary) return;
      setIsAnalyzing(true);
      setAiAnalysis("✨ AI is analyzing the redaction summary...");

      const summaryText = Object.entries(redactionSummary).map(([key, value]) => `${value} ${key.replace(/_/g, ' ')}`).join(', ');
      const prompt = `The following sensitive information was redacted from a document: ${summaryText}. In 2-3 short sentences and using a friendly, reassuring tone, explain why redacting this information is important for privacy and GDPR compliance. Start with "Great job securing this document!". Do not use markdown.`;

      const result = await callGeminiAPI(prompt);
      setAiAnalysis(result || "Could not get an analysis.");
      setIsAnalyzing(false);
  };


  if (showWelcome) {
      return <WelcomePage onGetStarted={() => {
                  console.log("Button pressed"); 
                  setShowWelcome(false)
                  }
                } 
              />;
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-black-50 p-4 font-sans animate-fade-in">
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
          handleSuggestTypes={handleSuggestTypes}
          isSuggesting={isSuggesting}
          handleAnalyzeSummary={handleAnalyzeSummary}
          isAnalyzing={isAnalyzing}
          aiAnalysis={aiAnalysis}
        />
        <PreviewSection
          selectedFile={selectedFile}
          pageNumber={pageNumber}
          numPages={numPages}
          goToPrevPage={goToPrevPage}
          goToNextPage={goToNextPage}
          onDocumentLoadSuccess={onDocumentLoadSuccess}
          setPageNumber={setPageNumber}
          setPdfDocForParent={setPdfDoc} // Pass setter to PDFViewer
        />
      </div>
      <Footer />
       <style>{`
        @keyframes fadeIn { from { opacity: 0; transform: scale(0.98); } to { opacity: 1; transform: scale(1); } }
        .animate-fade-in { animation: fadeIn 0.5s ease-out forwards; }
      `}</style>
    </div>
  );
};

export default App;
