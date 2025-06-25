import { POSSIBLE_BACKEND_URLS } from '../utils/constants.js';

// --- SERVICE: BACKEND & AI ---

/**
 * Calls the Gemini API with a specific prompt.
 * @param {string} prompt The prompt to send to the Gemini API.
 * @returns {Promise<string|null>} The generated text from the API, or null on error.
 */
export const callGeminiAPI = async (prompt) => {
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

// export const fetchTypesService = async () => {
//   for (const url of POSSIBLE_BACKEND_URLS) {
//     try {
//       console.log(`Trying to connect to backend at: ${url}`);
//       const response = await fetch(`${url}/redaction-types`, {
//         method: 'GET',
//         headers: { 'Accept': 'application/json' },
//       });
      
//       if (response.ok) {
//         const data = await response.json();
//         const types = data.redaction_types || [];
//         console.log(`Successfully connected to backend at: ${url}`);
//         return { success: true, types, connectedUrl: url };
//       }
//     } catch (error) {
//       console.log(`Failed to connect to ${url}:`, error.message);
//       continue;
//     }
//   }
//   return { success: false, triedUrls: POSSIBLE_BACKEND_URLS };
// };
export const fetchTypesService = async () => {
  // This function now uses the single, correct backend URL.
  try {
    console.log(`Trying to connect to backend at: ${POSSIBLE_BACKEND_URLS}`);
    const response = await fetch(`${POSSIBLE_BACKEND_URLS}/redaction-types`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
    });
    
    if (response.ok) {
      const data = await response.json();
      const types = data.redaction_types || [];
      console.log(`Successfully connected to backend at: ${POSSIBLE_BACKEND_URLS}`);
      return { success: true, types, connectedUrl: POSSIBLE_BACKEND_URLS };
    } else {
        // Handle non-OK responses from the server
        console.error(`Backend responded with status: ${response.status}`);
        return { success: false, error: `Server error (status: ${response.status})` };
    }
  } catch (error) {
    console.error(`Failed to connect to ${POSSIBLE_BACKEND_URLS}:`, error.message);
    // This will catch network errors like ERR_CONNECTION_REFUSED
    return { success: false, error: error.message };
  }
};

export const uploadAndRedactFile = async (file, selectedRedactionTypes, backendUrl) => {
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