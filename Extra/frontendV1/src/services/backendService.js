import { POSSIBLE_BACKEND_URLS } from '../utils/constants';

/**
 * Fetches the available redaction types from the backend.
 * It tries a list of possible URLs until a connection is successful.
 * @returns {Promise<object>} An object with success status, and data or error info.
 */
export const fetchRedactionTypes = async () => {
  let connectedUrl = null;

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
        connectedUrl = url;
        console.log(`Successfully connected to backend at: ${url}`);
        return { success: true, types, connectedUrl };
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
export const uploadAndRedactFile = async (file, selectedRedactionTypes, backendUrl) => {
  const formData = new FormData();
  formData.append('file', file);

  const queryParams = new URLSearchParams();
  selectedRedactionTypes.forEach(type => queryParams.append('redaction_types', type));
  
  const requestUrl = `${backendUrl}/redact-pdf/?${queryParams.toString()}`;

  try {
    console.log('Making request to:', requestUrl);
    console.log('Selected redaction types:', selectedRedactionTypes);
    
    const response = await fetch(requestUrl, {
      method: 'POST',
      body: formData,
    });

    console.log('Response status:', response.status);
    console.log('Response headers:', Object.fromEntries(response.headers.entries()));

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
