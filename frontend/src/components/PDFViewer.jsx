import React, { useState, useRef, useEffect } from 'react';

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

export default PDFViewer;