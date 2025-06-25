// --- UTILITY: CONSTANTS (Unchanged) ---
export const MAX_FILE_SIZE_MB = 5;
export const MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024;
export const ALLOWED_FILE_TYPES = ['application/pdf'];
export const POSSIBLE_BACKEND_URLS = [
  process.env.REACT_APP_BACKEND_URL,
  'http://localhost:8000'
];
export const FALLBACK_REDACTION_TYPES = [
    'PHONE_NUMBER', 'EMAIL_ADDRESS', 'PERSON', 'CREDIT_CARD', 
    'SSN', 'LOCATION', 'ORGANIZATION', 'IP_ADDRESS', 'URL'
];

// --- FIX: Replaced local image import with a stable URL to resolve the build error ---
export const logoUrl = '/Redaction-NO-BG.png';