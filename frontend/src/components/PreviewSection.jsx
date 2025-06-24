import React from 'react';
import PDFViewer from './PDFViewer';

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

export default PreviewSection;