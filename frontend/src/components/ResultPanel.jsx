export default function ResultPanel({ result }) {
  const isMalignant = result.prediction === "Malignant";
  const isLowConfidence = result.confidence_label?.toLowerCase().includes("low");

  const formatNum = (v) => (typeof v === "number" ? v.toFixed(4) : v);

  return (
    <div className="result-row">
      <div className="result-box">
        <span className={`pred-badge ${isMalignant ? "malignant" : "benign"}`}>
          {result.prediction}
        </span>

        <div className="prob-label">Probability — Malignant</div>
        <div className="bar-track">
          <div
            className="bar-fill malignant"
            style={{ width: `${result.probability_cancer * 100}%` }}
          />
        </div>

        <div className="prob-label">Probability — Benign</div>
        <div className="bar-track">
          <div
            className="bar-fill"
            style={{ width: `${result.probability_benign * 100}%` }}
          />
        </div>
      </div>

      <div className="result-box">
        <div className="mono-row"><span>Confidence</span><b>{result.confidence_label}</b></div>
        <div className="mono-row"><span>Threshold used</span><b>{formatNum(result.threshold_used)}</b></div>
        <div className="mono-row"><span>Model AUC</span><b>{formatNum(result.model_auc)}</b></div>
        <div className="mono-row"><span>Filename</span><b>{result.filename}</b></div>

        <div
          className={`recommendation ${isLowConfidence ? "low-confidence" : isMalignant ? "malignant" : ""}`}
        >
          {result.recommendation}
        </div>
      </div>
    </div>
  );
}