// Frontend validation constants
export const MAX_FILE_SIZE_MB = 5;
export const MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024;
export const ALLOWED_FILE_TYPES = ['application/pdf'];

// Backend URL configuration
export const POSSIBLE_BACKEND_URLS = [
  'http://localhost:8000',
  'http://127.0.0.1:8000',
  'http://0.0.0.0:8000'
];

// Fallback redaction types for offline/demo mode
export const FALLBACK_REDACTION_TYPES = [
    'PHONE_NUMBER', 'EMAIL_ADDRESS', 'PERSON', 'CREDIT_CARD', 
    'SSN', 'LOCATION', 'ORGANIZATION', 'IP_ADDRESS', 'URL'
];
