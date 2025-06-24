import React, { useState, useRef, useEffect } from 'react';

// The PDFViewer component is now included directly in this file to resolve the import error.
// It encapsulates the PDF.js library logic for rendering a PDF page.
const PDFViewer = ({ file, currentPage, onLoadSuccess }) => {
  const canvasRef = useRef(null);
  const [pdfDoc, setPdfDoc] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [pdfJsLoaded, setPdfJsLoaded] = useState(false);

  // Effect to load the PDF.js library from a CDN
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

  // Effect to load the PDF document when the file or library status changes
  useEffect(() => {
    if (file && pdfJsLoaded) {
      loadPDF();
    }
  }, [file, pdfJsLoaded]);

  // Effect to render a specific page when the document or page number changes
  useEffect(() => {
    if (pdfDoc && currentPage) {
      renderPage(currentPage);
    }
  }, [pdfDoc, currentPage]);

  // Function to load the PDF from the file object
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

  // Function to render a specific page onto the canvas
  const renderPage = async (pageNum) => {
    if (!pdfDoc || !canvasRef.current) return;
    try {
      const page = await pdfDoc.getPage(pageNum);
      const canvas = canvasRef.current;
      const context = canvas.getContext('2d');
      const containerWidth = canvas.parentElement.clientWidth - 32; // Account for padding
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


// Section for displaying the PDF preview and its navigation controls
const PreviewSection = ({
  selectedFile,
  pageNumber,
  numPages,
  goToPrevPage,
  goToNextPage,
  onDocumentLoadSuccess,
  setPageNumber,
}) => {
  if (!selectedFile) {
    return null; // Don't render anything if no file is selected
  }

  return (
    <div className={`bg-white p-6 rounded-xl shadow-lg flex-1 min-w-0 transform transition-all duration-1000 ease-out delay-300 ${
      selectedFile ? 'translate-x-0 opacity-100' : 'translate-x-full opacity-0'
    }`}>
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
  );
};

export default PreviewSection;
