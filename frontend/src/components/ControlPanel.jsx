import React, { useState } from 'react';
import RedactionSelector from './RedactionSelector';
import { MAX_FILE_SIZE_MB } from '../utils/constants';

// --- FIX: Replaced local image import with a stable URL to resolve the build error ---
const logoUrl = '/Redaction-NO-BG.png';

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

export default ControlPanel;