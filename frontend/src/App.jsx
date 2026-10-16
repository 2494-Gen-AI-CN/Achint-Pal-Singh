import { useEffect, useRef, useState } from "react";
import "./App.css";

const API_URL = (import.meta.env.VITE_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
const BRAND = import.meta.env.VITE_GUIDE_NAME || "California Restaurant Guide";
const label = (value) => value.replaceAll("_", " ");

function restaurantInitials(name) {
  return name.trim().split(/\s+/).slice(0, 2).map((word) => Array.from(word)[0]).join("").toLocaleUpperCase();
}

function mapsSearchUrl(restaurant) {
  const query = [restaurant.name, restaurant.city, restaurant.state, restaurant.country]
    .filter(Boolean).join(", ");
  return `https://www.google.com/maps/search/?${new URLSearchParams({ api: "1", query })}`;
}

async function request(path, options = {}) {
  const response = await fetch(`${API_URL}${path}`, { ...options, signal: AbortSignal.timeout(20000) });
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Please check your preferences and try again.");
  return data;
}

export default function App() {
  const [options, setOptions] = useState(null);
  const [optionsError, setOptionsError] = useState("");
  const [retry, setRetry] = useState(0);
  const [preferences, setPreferences] = useState({ budget: "any", cuisine: "", area: "", top_k: 6 });
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const generation = useRef(0);

  useEffect(() => {
    document.title = BRAND;
    let active = true;
    setOptionsError("");
    request("/discovery/options")
      .then((data) => { if (active) setOptions(data); })
      .catch(() => { if (active) setOptionsError("We couldn’t load the catalog. Check that the API is running, then retry."); });
    return () => { active = false; };
  }, [retry]);

  function updatePreference(event) {
    const { name, value } = event.target;
    generation.current += 1;
    setPreferences((current) => ({ ...current, [name]: name === "top_k" ? Number(value) : value }));
    setResults(null);
    setError("");
    setLoading(false);
  }

  async function discover(event) {
    event.preventDefault();
    const current = ++generation.current;
    setLoading(true);
    setError("");
    setResults(null);
    try {
      const data = await request("/recommend/preferences", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...preferences, cuisine: preferences.cuisine || null, area: preferences.area || null }),
      });
      if (current === generation.current) setResults(data);
    } catch (err) {
      if (current === generation.current) setError(
        err.name === "TypeError" || err.name === "TimeoutError"
          ? "We couldn’t reach the restaurant API. Check your connection and try again." : err.message,
      );
    } finally {
      if (current === generation.current) setLoading(false);
    }
  }

  return (
    <div className="app">
      <a className="skip-link" href="#discover">Skip to discovery</a>
      <header className="header">
        <a href="#" className="brand"><span className="brand-mark" aria-hidden="true">✳</span><span>{BRAND}</span></a>
        <nav aria-label="Main navigation"><a href="#discover">Discover</a><a href="#about">How it works</a></nav>
        <span className="demo-badge">Demo edition</span>
      </header>
      <main className="container">
        <section className="hero">
          <div>
            <p className="eyebrow">GOOD FOOD. YOUR WAY.</p>
            <h1>A little discovery.<br /><em>A great next meal.</em></h1>
            <p className="intro">Start with what matters to you. Explore restaurants by budget, cuisine, and dining space, with a clear reason behind every suggestion.</p>
            <a className="text-link" href="#discover">Find your next table <span aria-hidden="true">↗</span></a>
          </div>
          <aside className="hero-note" aria-label="About this edition">
            <span className="note-symbol" aria-hidden="true">✳</span><p className="eyebrow">THE DISCOVERY EDITION</p>
            <h2>Real records.<br />An honest starting point.</h2>
            <p>Explore a historical restaurant catalog. No invented reviews, photos, or promises.</p>
            <span className="note-footer">A preference-led demo</span>
          </aside>
        </section>
        <div className="data-notice"><strong>Demo data · Mexico</strong><p>{options?.demo_notice || "This prototype uses historical UCI restaurant records from Mexico, plus records with unknown country. It does not contain verified California restaurants."}</p></div>
        <section id="discover" className="search-panel" aria-labelledby="discover-heading">
          <div className="section-title"><div><p className="eyebrow">MAKE IT YOURS</p><h2 id="discover-heading">What sounds good?</h2></div><span>No account needed</span></div>
          {optionsError ? <div role="alert" className="error">{optionsError} <button onClick={() => setRetry((value) => value + 1)}>Retry catalog</button></div> : !options ? <p role="status">Loading available preferences…</p> : (
            <form onSubmit={discover}>
              <div className="form-grid">
                <label>Budget<select name="budget" value={preferences.budget} onChange={updatePreference}><option value="any">Any budget</option>{options.budgets.map((budget) => <option key={budget} value={budget}>{budget === "low" ? "Low · budget friendly" : budget === "medium" ? "Medium · everyday dining" : "High · a little extra"}</option>)}</select></label>
                <label>Cuisine<select name="cuisine" value={preferences.cuisine} onChange={updatePreference}><option value="">Any cuisine</option>{options.cuisines.map((cuisine) => <option key={cuisine} value={cuisine}>{label(cuisine)}</option>)}</select></label>
                <label>Dining space<select name="area" value={preferences.area} onChange={updatePreference}><option value="">Any dining space</option>{options.areas.map((area) => <option key={area} value={area}>{area === "open" ? "Outdoor · open area" : "Indoor · closed area"}</option>)}</select></label>
                <label>Suggestions<select name="top_k" value={preferences.top_k} onChange={updatePreference}>{[3, 6, 12].map((count) => <option key={count} value={count}>{count} restaurants</option>)}</select></label>
              </div>
              <div className="form-bottom"><p>Exact preference matches. Ranked using historical ratings.</p><button className="primary-button" disabled={loading} type="submit">{loading ? "Finding restaurants…" : "Find Restaurants"} <span aria-hidden="true">↗</span></button></div>
            </form>
          )}
        </section>
        <div aria-live="polite" aria-atomic="true">{loading && <p className="status">Matching your preferences and ranking the catalog…</p>}{error && <p className="error" role="alert">{error}</p>}{results && <p className="sr-only">{results.total_recommendations} restaurants found.</p>}</div>
        {results && <section className="results" aria-labelledby="results-heading">
          <div className="section-title"><div><p className="eyebrow">A PLACE TO START</p><h2 id="results-heading">Your restaurant shortlist</h2></div><span>{results.total_recommendations} results</span></div>
          <p className="results-note">Preference matches · Bayesian ranking · Scores are ranking signals on a 0–2 scale, not customer ratings or ANN predictions.</p>
          {results.recommendations.length === 0 ? <div className="empty"><h3>No exact matches this time.</h3><p>Try any cuisine, a different budget, or any dining space to explore more of the catalog.</p></div> : <div className="restaurant-grid">
            {results.recommendations.map((restaurant, index) => <article className="restaurant-card" key={restaurant.restaurant_id}>
              <div className="card-top"><span className="rank">{String(index + 1).padStart(2, "0")}</span><span className="price">{restaurant.price ? `${restaurant.price} budget` : "Budget unrecorded"}</span></div>
              <div className="restaurant-identity">
                <span className="restaurant-monogram" aria-hidden="true">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><path d="M4 3v5a3 3 0 0 0 6 0V3M7 3v18M17 3c-2 3-3 6-3 10h5M19 3v18" /></svg>
                  <span>{restaurantInitials(restaurant.name)}</span>
                </span>
                <h3>{restaurant.name}</h3>
              </div>
              <p className="location">{[restaurant.city, restaurant.state, restaurant.country].filter(Boolean).join(", ") || "Location unrecorded"}{!restaurant.country && " · country unrecorded"}</p>
              <p className="cuisines">{restaurant.cuisines.length ? restaurant.cuisines.map(label).join(" · ") : "Cuisine unrecorded"}</p>
              <div className="score"><strong>{restaurant.score.toFixed(2)}<small> / 2</small></strong><span>Bayesian ranking score</span></div>
              <p className="explanation">{restaurant.explanation}</p>
              <div className="card-footer"><span>{restaurant.rating_count} historical ratings</span><span>Demo record</span></div>
              <a className="maps-link" href={mapsSearchUrl(restaurant)} target="_blank" rel="noopener noreferrer" aria-label={`Search for ${restaurant.name} on Google Maps (opens in a new tab)`}>
                <span><svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d="M20 10c0 6-8 12-8 12S4 16 4 10a8 8 0 1 1 16 0Z" /><circle cx="12" cy="10" r="2.5" /></svg>Search Google Maps</span>
                <span className="maps-arrow" aria-hidden="true">↗</span>
              </a>
              <p className="maps-note">Uses recorded name and location · Listing unverified</p>
            </article>)}
          </div>}
        </section>}
        <section id="about" className="about" aria-labelledby="about-heading"><div><p className="eyebrow">A CLEARER KIND OF DISCOVERY</p><h2 id="about-heading">Good suggestions.<br />No mystery.</h2></div><div><h3>Your preferences come first</h3><p>Budget, recorded cuisine, and dining space narrow the catalog. Missing metadata never counts as a match.</p><h3>History helps order the list</h3><p>We balance each restaurant’s average rating with the overall average, using the weight of 10 ratings. This softens high scores from very small samples. New visitors use this approach; the existing-user API separately uses the trained ANN.</p></div></section>
      </main>
      <footer><span>{BRAND}</span><span>UCI demo catalog · Historical data · Current details unverified</span></footer>
    </div>
  );
}
