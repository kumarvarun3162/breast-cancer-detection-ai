export default function HeroIllustration() {
  return (
    <svg
      viewBox="0 0 420 320"
      width="100%"
      height="100%"
      role="img"
      aria-label="Abstract illustration of AI-assisted scan analysis"
    >
      <defs>
        <linearGradient id="scanGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#0E8A7D" stopOpacity="0.16" />
          <stop offset="100%" stopColor="#0E8A7D" stopOpacity="0" />
        </linearGradient>
      </defs>

      {/* background grid */}
      <g stroke="#D7DEE5" strokeWidth="1">
        {Array.from({ length: 7 }).map((_, i) => (
          <line key={`v${i}`} x1={40 + i * 56} y1="30" x2={40 + i * 56} y2="290" />
        ))}
        {Array.from({ length: 5 }).map((_, i) => (
          <line key={`h${i}`} x1="40" y1={30 + i * 65} x2="380" y2={30 + i * 65} />
        ))}
      </g>

      {/* focus target */}
      <circle cx="210" cy="160" r="92" fill="url(#scanGrad)" />
      <circle cx="210" cy="160" r="92" stroke="#0F2942" strokeWidth="1.5" fill="none" />
      <circle cx="210" cy="160" r="62" stroke="#0E8A7D" strokeWidth="1.5" fill="none" strokeDasharray="6 5" />
      <circle cx="210" cy="160" r="6" fill="#0E8A7D" />

      {/* corner crosshairs */}
      {[
        [118, 68],
        [302, 68],
        [118, 252],
        [302, 252],
      ].map(([x, y], i) => (
        <g key={i} stroke="#42536B" strokeWidth="1.5">
          <line x1={x - 8} y1={y} x2={x + 8} y2={y} />
          <line x1={x} y1={y - 8} x2={x} y2={y + 8} />
        </g>
      ))}

      {/* scan line */}
      <rect x="118" y="158" width="184" height="3" fill="#0E8A7D" opacity="0.7" />

      {/* result chip */}
      <g transform="translate(272, 108)">
        <rect width="86" height="30" rx="4" fill="#FFFFFF" stroke="#D7DEE5" />
        <circle cx="14" cy="15" r="4.5" fill="#1E824C" />
        <text x="26" y="19" fontFamily="IBM Plex Mono, monospace" fontSize="10" fill="#0F2942">
          0.88 conf.
        </text>
      </g>
    </svg>
  );
}