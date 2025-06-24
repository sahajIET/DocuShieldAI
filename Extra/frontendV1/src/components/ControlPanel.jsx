import React, { useState } from 'react';

// Constant previously in utils/constants.js, now inlined to resolve import error.
const MAX_FILE_SIZE_MB = 5;

// Component previously in components/RedactionSelector.jsx, now inlined to resolve import error.
const RedactionSelector = ({ allTypes, selectedTypes, setSelectedTypes }) => {
  
  const handleTypeChange = (type) => {
    setSelectedTypes(prevTypes => {
      if (prevTypes.includes(type)) {
        return prevTypes.filter(t => t !== type);
      } else {
        return [...prevTypes, type];
      }
    });
  };

  const handleSelectAll = () => {
    if (selectedTypes.length === allTypes.length) {
      setSelectedTypes([]);
    } else {
      setSelectedTypes(allTypes);
    }
  };

  const formatTypeName = (type) => {
    return type.replace(/_/g, ' ').replace('PRESIDIO', 'Presidio:').replace('REGEX', 'Regex:').replace('SPACY', 'SpaCy:');
  };

  return (
    <div className="border rounded-lg p-4 mb-6 text-left max-h-60 overflow-y-auto">
      <div className="flex justify-between items-center mb-3 pb-2 border-b">
        <h3 className="text-lg font-semibold text-gray-800">Choose Types</h3>
        <button 
          onClick={handleSelectAll} 
          className="text-blue-600 hover:text-blue-800 text-sm font-medium"
        >
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


// The main control panel for file selection, options, and actions
const ControlPanel = ({
  selectedFile,
  fileInputRef,
  handleFileChange,
  allRedactionTypes,
  selectedRedactionTypes,
  setSelectedRedactionTypes,
  handleUpload,
  isLoading,
  backendStatus,
  progressMessage,
  message,
  redactionSummary,
  fetchRedactionTypes,
}) => {
  const [showTypeSelector, setShowTypeSelector] = useState(false);

  return (
    <div className={`bg-white p-8 rounded-xl shadow-lg text-center transition-all duration-500 ease-out ${
      selectedFile 
        ? 'w-full lg:w-96 lg:min-w-96 transform translate-x-0' 
        : 'w-full max-w-md'
    }`}>
      <h1 className={`font-bold text-gray-800 mb-6 transition-all duration-300 ${
        selectedFile ? 'text-3xl' : 'text-4xl'
      }`}>
        DocuShield AI
      </h1>
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

      {/* Redaction Type Selector Toggle */}
      <button 
        onClick={() => setShowTypeSelector(!showTypeSelector)}
        className="w-full mb-4 px-4 py-2 rounded-lg font-semibold text-sm bg-gray-200 text-gray-700 hover:bg-gray-300 transition-colors"
      >
        {showTypeSelector ? 'Hide Redaction Options' : 'Show Redaction Options'} 
        ({selectedRedactionTypes.length} / {allRedactionTypes.length} selected)
      </button>

      {showTypeSelector && (
        <RedactionSelector
          allTypes={allRedactionTypes}
          selectedTypes={selectedRedactionTypes}
          setSelectedTypes={setSelectedRedactionTypes}
        />
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

      {/* Retry Backend Connection */}
      {backendStatus === 'disconnected' && (
        <div className="mt-6 bg-white p-4 rounded-lg shadow-md text-center">
          <button 
            onClick={fetchRedactionTypes}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            Retry Backend Connection
          </button>
        </div>
      )}
    </div>
  );
};

export default ControlPanel;
