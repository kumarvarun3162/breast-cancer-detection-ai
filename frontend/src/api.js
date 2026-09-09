// Base URL for the FastAPI backend.
// In dev, defaults to the backend's local port (see main.py: uvicorn.run(..., port=8080)).
// Override by creating a .env file with VITE_API_BASE_URL=https://your-deployed-api
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8080";

class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

async function handleResponse(res) {
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
      // response wasn't JSON — keep the generic message
    }
    throw new ApiError(detail, res.status);
  }
  return res.json();
}

export async function getHealth() {
  const res = await fetch(`${API_BASE_URL}/health`);
  return handleResponse(res);
}

export async function getModelInfo() {
  const res = await fetch(`${API_BASE_URL}/model-info`);
  return handleResponse(res);
}

export async function predictMammogram(file) {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${API_BASE_URL}/predict`, {
    method: "POST",
    body: formData,
  });
  return handleResponse(res);
}

export { ApiError, API_BASE_URL };
