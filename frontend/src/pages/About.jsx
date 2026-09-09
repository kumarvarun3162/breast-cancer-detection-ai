import { useEffect, useState } from "react";
import { getModelInfo } from "../api";

function formatValue(v) {
  if (Array.isArray(v)) return v.join(", ");
  if (typeof v === "number") return Number.isInteger(v) ? v.toString() : v.toFixed(4);
  return v ?? "—";
}

export default function About() {
  const [info, setInfo] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    getModelInfo()
      .then(setInfo)
      .catch(() => setError("Could not load model info from the backend. Is the API running?"));
  }, []);

  const rows = info
    ? [
        ["Architecture", info.architecture],
        ["Image size", `${info.image_size} × ${info.image_size}`],
        ["Test AUC", info.test_auc],
        ["Validation AUC", info.val_auc],
        ["Sensitivity", info.sensitivity],
        ["Specificity", info.specificity],
        ["Decision threshold", info.threshold],
        ["Epochs trained", info.epochs_trained],
        ["Dataset", info.dataset],
        ["Classes", info.classes],
      ]
    : [];

  return (
    <div className="container" style={{ paddingBottom: 40 }}>
      <div className="section-band" style={{ margin: "24px -24px 0" }}>
        <div className="container">About / Model Info</div>
      </div>

      <div className="card">
        {error && <div className="error-banner" role="alert">{error}</div>}

        {!info && !error && (
          <>
            <div className="progress-track"><div className="progress-fill" /></div>
            <div className="progress-label">Loading model info…</div>
          </>
        )}

        {info && (
          <table className="spec-table">
            <tbody>
              {rows.map(([label, value]) => (
                <tr key={label}>
                  <th scope="row">{label}</th>
                  <td>{formatValue(value)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}