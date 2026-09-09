import { useRef, useState } from "react";
import { predictMammogram } from "../api";
import ResultPanel from "../components/ResultPanel";

const ACCEPTED_TYPES = ["image/png", "image/jpeg", "image/jpg"];
const MAX_SIZE_BYTES = 20 * 1024 * 1024;

function formatFileSize(bytes) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function Detect() {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("idle"); // idle | loading | done | error
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [isDragOver, setIsDragOver] = useState(false);
  const inputRef = useRef(null);

  function validateAndSetFile(candidate) {
    if (!candidate) return;
    if (!ACCEPTED_TYPES.includes(candidate.type)) {
      setError("Only PNG and JPG images are accepted.");
      setStatus("error");
      return;
    }
    if (candidate.size > MAX_SIZE_BYTES) {
      setError(`File is ${(candidate.size / (1024 * 1024)).toFixed(1)}MB. Maximum size is 20MB.`);
      setStatus("error");
      return;
    }
    setFile(candidate);
    setError("");
    setStatus("idle");
    setResult(null);
  }

  function handleDrop(e) {
    e.preventDefault();
    setIsDragOver(false);
    validateAndSetFile(e.dataTransfer.files?.[0]);
  }

  async function handleAnalyze() {
    if (!file) return;
    setStatus("loading");
    setError("");
    try {
      const data = await predictMammogram(file);
      setResult(data);
      setStatus("done");
    } catch (err) {
      setError(err.message || "Prediction failed. Please try again.");
      setStatus("error");
    }
  }

  function handleReset() {
    setFile(null);
    setResult(null);
    setError("");
    setStatus("idle");
    if (inputRef.current) inputRef.current.value = "";
  }

  return (
    <div className="container" style={{ paddingBottom: 40 }}>
      <div className="section-band" style={{ margin: "24px -24px 0" }}>
        <div className="container">Detection &amp; Result</div>
      </div>

      <div className="card">
        <label
          className={`dropzone ${isDragOver ? "dragover" : ""}`}
          onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
          onDragLeave={() => setIsDragOver(false)}
          onDrop={handleDrop}
        >
          <input
            ref={inputRef}
            type="file"
            accept="image/png, image/jpeg"
            onChange={(e) => validateAndSetFile(e.target.files?.[0])}
          />

          {file ? (
            <div className="dropzone-file-preview">
              <div className="file-icon">
                <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                  <path d="M4 2h8l4 4v12a1 1 0 01-1 1H4a1 1 0 01-1-1V3a1 1 0 011-1z" stroke="#1E824C" strokeWidth="1.4" strokeLinejoin="round" />
                  <path d="M6.5 11.5l2 2 5-5" stroke="#1E824C" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </div>
              <div className="file-info">
                <div className="file-name">{file.name}</div>
                <div className="file-size">{formatFileSize(file.size)} · ready to analyze</div>
              </div>
            </div>
          ) : (
            <div className="dropzone-empty">
              <div className="dropzone-icon">
                <svg width="22" height="22" viewBox="0 0 22 22" fill="none">
                  <path d="M11 14V4M11 4L7 8M11 4l4 4" stroke="#0E8A7D" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
                  <path d="M4 14v2.5A1.5 1.5 0 005.5 18h11a1.5 1.5 0 001.5-1.5V14" stroke="#0E8A7D" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </div>
              <div className="dropzone-title">Drag &amp; drop your mammogram here</div>
              <div className="dropzone-sub">
                or <span className="browse-link">click to browse</span> · PNG or JPG · up to 20MB
              </div>
            </div>
          )}
        </label>

        {file && status !== "loading" && status !== "done" && (
          <div className="dropzone-actions">
            <button className="btn btn-primary" onClick={handleAnalyze}>Analyze Image</button>
            <button className="btn btn-outline" onClick={handleReset}>Remove</button>
          </div>
        )}

        {status === "loading" && (
          <>
            <div className="progress-track"><div className="progress-fill" /></div>
            <div className="progress-label">Analyzing… EfficientNet-B0 · 224×224</div>
          </>
        )}

        {status === "error" && (
          <div className="error-banner" role="alert">{error}</div>
        )}

        {status === "done" && result && (
          <>
            <ResultPanel result={result} />
            <div className="dropzone-actions">
              <button className="btn btn-outline" onClick={handleReset}>Analyze Another Image</button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}