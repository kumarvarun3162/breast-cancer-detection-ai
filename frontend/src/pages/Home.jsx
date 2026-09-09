import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { getModelInfo } from "../api";
import HeroIllustration from "../components/HeroIllustration";

const FALLBACK_STATS = { test_auc: "0.76", sensitivity: "0.70", specificity: "0.73" };

function formatPct(value) {
  if (typeof value !== "number") return value ?? "—";
  return `${Math.round(value * 100)}%`;
}

function formatAuc(value) {
  if (typeof value !== "number") return value ?? "—";
  return value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
}

export default function Home() {
  const [stats, setStats] = useState(FALLBACK_STATS);
  const [statsLoaded, setStatsLoaded] = useState(false);

  useEffect(() => {
    let cancelled = false;
    getModelInfo()
      .then((data) => {
        if (!cancelled) {
          setStats(data);
          setStatsLoaded(true);
        }
      })
      .catch(() => {
        // Backend unreachable — keep the fallback numbers, fail quietly on the homepage.
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <>
      <section className="hero container">
        <div className="hero-copy">
          <h1>Mammogram Analysis, Decision-Support Grade</h1>
          <p>
            An AI-assisted screening tool trained on 20,000 mammograms. This is a
            decision-support aid, not a diagnostic verdict — always confirm with a radiologist.
          </p>
          <Link to="/detect" className="btn btn-primary">Start Detection</Link>

          <div className="stat-strip">
            <div className="stat">
              <div className="stat-value">
                {statsLoaded ? formatAuc(stats.test_auc) : FALLBACK_STATS.test_auc}
              </div>
              <div className="stat-label">Test AUC</div>
            </div>
            <div className="stat">
              <div className="stat-value">
                {statsLoaded ? formatPct(stats.sensitivity) : formatPct(0.70)}
              </div>
              <div className="stat-label">Sensitivity</div>
            </div>
            <div className="stat">
              <div className="stat-value">
                {statsLoaded ? formatPct(stats.specificity) : formatPct(0.73)}
              </div>
              <div className="stat-label">Specificity</div>
            </div>
          </div>
        </div>
        <div className="hero-visual" aria-hidden="true">
          <HeroIllustration />
        </div>
      </section>

      <div className="section-band">
        <div className="container">How it works</div>
      </div>
      <div className="container">
        <div className="steps-row">
          <div className="step-card">
            <div className="step-num">01</div>
            <h3>Upload</h3>
            <p>Choose a mammogram image in PNG or JPG format, up to 20MB.</p>
          </div>
          <div className="step-card">
            <div className="step-num">02</div>
            <h3>Analyze</h3>
            <p>The EfficientNet-B0 model scores the image against a tuned decision threshold.</p>
          </div>
          <div className="step-card">
            <div className="step-num">03</div>
            <h3>Review</h3>
            <p>See the prediction, confidence level, and a recommended next step.</p>
          </div>
        </div>
      </div>
    </>
  );
}