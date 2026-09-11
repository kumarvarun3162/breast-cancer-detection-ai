import { NavLink } from "react-router-dom";

export default function Header() {
  return (
    <header className="site-header">
      <div className="container" style={{ display: "flex", justifyContent: "space-between", alignItems: "center", width: "100%" }}>
        <NavLink to="/" className="brand">
          <svg width="26" height="26" viewBox="0 0 26 26" fill="none" aria-hidden="true">
            <circle cx="13" cy="13" r="12" stroke="#0E8A7D" strokeWidth="1.6" />
            <path d="M8 13.5L11.2 16.7L18 9.3" stroke="#0E8A7D" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <span>BreastScan&nbsp;AI</span>
        </NavLink>
        <nav className="main-nav" aria-label="Main navigation">
          <NavLink to="/" end>Home</NavLink>
          <NavLink to="/detect">Detect</NavLink>
          <NavLink to="/about">About</NavLink>
        </nav>
      </div>
    </header>
  );
}