import React, { useState, useRef, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, useNavigate } from 'react-router-dom';
import WelcomePage from './components/WelcomePage';
import ControlPanel from './components/ControlPanel';
import PreviewSection from './components/PreviewSection';
import BackendStatus from './components/BackendStatus';
import Footer from './components/Footer';
import { fetchTypesService, uploadAndRedactFile } from './services/backendService';
import { callGeminiAPI } from './services/geminiService';
import { 
  MAX_FILE_SIZE_MB, 
  MAX_FILE_SIZE_BYTES, 
  ALLOWED_FILE_TYPES, 
  FALLBACK_REDACTION_TYPES 
} from './utils/constants';
import { 
  extractFilenameFromHeader, 
  triggerDownload, 
  simulateProgress, 
  generateRedactionSummary 
} from './utils/helpers';

// --- WELCOME PAGE WRAPPER ---
const WelcomePageWrapper = () => {
  const navigate = useNavigate();
  
  const handleGetStarted = () => {
    navigate('/app');
  };

  return <WelcomePage onGetStarted={handleGetStarted} />;
};

// --- MAIN APP COMPONENT ---
const MainApp = () => {
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

// --- ROOT APP COMPONENT WITH ROUTING ---
const App = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<WelcomePageWrapper />} />
        <Route path="/app" element={<MainApp />} />
      </Routes>
    </Router>
  );
};

export default App;