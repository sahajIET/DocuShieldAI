import React from 'react';

// Component to select which types of information to redact
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

  // Helper to format display names
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

export default RedactionSelector;
