// --- UTILITY: HELPERS (Unchanged) ---
export const extractFilenameFromHeader = (contentDisposition) => {
  if (!contentDisposition) return 'redacted_document.pdf';
  let match = contentDisposition.match(/filename\*=UTF-8''([^;]+)/i);
  if (match?.[1]) return decodeURIComponent(match[1]);
  match = contentDisposition.match(/filename="([^"]*)"/i);
  if (match?.[1]) return match[1];
  return 'redacted_document.pdf';
};

export const triggerDownload = (blob, filename) => {
  const link = document.createElement('a');
  link.href = window.URL.createObjectURL(blob);
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.URL.revokeObjectURL(link.href);
};

export const simulateProgress = (setProgressMessage, numPages) => {
  const progressSteps = [
    'Initializing redaction process...',
    'Uploading file to server...',
    'Analyzing document structure...',
    'Detecting sensitive information...',
    numPages ? `Processing page 1 of ${numPages}...` : 'Processing document...',
    numPages && numPages > 1 ? `Processing page 2 of ${numPages}...` : null,
    numPages && numPages > 2 ? `Processing remaining pages...` : null,
    'Applying redactions...',
    'Finalizing document...',
    'Preparing download...'
  ].filter(Boolean);

  let stepIndex = 0;
  const interval = setInterval(() => {
    if (stepIndex < progressSteps.length) {
      setProgressMessage(progressSteps[stepIndex]);
      stepIndex++;
    } else {
      clearInterval(interval);
    }
  }, 1000);

  return interval;
};

export const generateRedactionSummary = () => {
  const summaryItems = {
    'PHONE_NUMBER': Math.floor(Math.random() * 6),
    'EMAIL_ADDRESS': Math.floor(Math.random() * 4),
    'PERSON': Math.floor(Math.random() * 10),
    'CREDIT_CARD': Math.floor(Math.random() * 3),
    'SSN': Math.floor(Math.random() * 2),
    'LOCATION': Math.floor(Math.random() * 5),
    'ORGANIZATION': Math.floor(Math.random() * 4)
  };
  
  return Object.fromEntries(
    Object.entries(summaryItems).filter(([key, value]) => value > 0)
  );
};