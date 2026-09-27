const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

/**
 * Sends the uploaded file to the NLP service and returns the parsed
 * { summary, followup_questions, recommended_resources, passage_count }.
 */
export async function processNotes(file) {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_BASE}/process`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed with status ${res.status}`);
  }

  return res.json();
}