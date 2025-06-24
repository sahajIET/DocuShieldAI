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