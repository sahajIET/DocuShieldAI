import React from 'react';

const RedactionSelector = ({ allTypes, selectedTypes, setSelectedTypes }) => {
  const handleTypeChange = (type) => { setSelectedTypes(prev => prev.includes(type) ? prev.filter(t => t !== type) : [...prev, type]); };
  const handleSelectAll = () => { setSelectedTypes(selectedTypes.length === allTypes.length ? [] : allTypes); };
  const formatTypeName = (type) => type.replace(/_/g, ' ').replace('PRESIDIO', 'Presidio:').replace('REGEX', 'Regex:').replace('SPACY', 'SpaCy:');
  
  return (
    <div className="bg-white/80 backdrop-blur-md border border-white/20 rounded-2xl p-6 mb-6 text-left max-h-60 overflow-y-auto shadow-xl">
      <div className="flex justify-between items-center mb-4 pb-3 border-b border-white/30">
        <h3 className="text-lg font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">Choose types</h3>
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

export default RedactionSelector;