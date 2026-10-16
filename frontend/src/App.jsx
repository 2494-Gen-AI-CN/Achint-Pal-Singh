import { useEffect, useRef, useState } from "react";
import "./App.css";

const API_URL = (import.meta.env.VITE_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
const BRAND = import.meta.env.VITE_GUIDE_NAME || "Mexico Restaurant Guide";
const label = (value) => value.replaceAll("_", " ");
const paymentLabel = (value) => ({ visa: "Visa", "mastercard-eurocard": "Mastercard / Eurocard", cash: "Cash", american_express: "American Express", bank_debit_cards: "Debit card" }[value] || label(value));
const parkingLabel = (value) => ({ none: "No parking", yes: "Parking (type unspecified)", free: "Free parking", public: "Public parking", street: "Street parking", valet_parking: "Valet parking", validated_parking: "Validated parking" }[value] || label(value));

// Decorative initials for each card; these are not official restaurant logos.
function restaurantInitials(name) {
  return name.trim().split(/\s+/).slice(0, 2).map((word) => Array.from(word)[0]).join("").toLocaleUpperCase();
}

// Search using the recorded location; the link is not a verified business listing.
function mapsSearchUrl(restaurant) {
  const query = [restaurant.name, restaurant.city, restaurant.state, restaurant.country]
    .filter(Boolean).join(", ");
  return `https://www.google.com/maps/search/?${new URLSearchParams({ api: "1", query })}`;
}

// Shared JSON API helper: stop waiting after 20 seconds and surface server errors.
async function request(path, options = {}) {
  const response = await fetch(`${API_URL}${path}`, { ...options, signal: AbortSignal.timeout(20000) });
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Please check your preferences and try again.");
  return data;
}

// Share the same real catalog filters between visitor and existing-user forms.
function PreferenceFields({ options, preferences, onChange }) {
  return <>
                <label>Budget<select name="budget" value={preferences.budget} onChange={onChange}><option value="any">Any budget</option>{options.budgets.map((budget) => <option key={budget} value={budget}>{budget === "low" ? "Low · budget friendly" : budget === "medium" ? "Medium · everyday dining" : "High · a little extra"}</option>)}</select></label>
                <label>Cuisine<select name="cuisine" value={preferences.cuisine} onChange={onChange}><option value="">Any cuisine</option>{options.cuisines.map((cuisine) => <option key={cuisine} value={cuisine}>{label(cuisine)}</option>)}</select></label>
                <label>Dining space<select name="area" value={preferences.area} onChange={onChange}><option value="">Any dining space</option>{options.areas.map((area) => <option key={area} value={area}>{area === "open" ? "Outdoor · open area" : "Indoor · closed area"}</option>)}</select></label>
                <label>Parking<select name="parking" value={preferences.parking} onChange={onChange}><option value="">Any parking arrangement</option>{options.parking.map((value) => <option key={value} value={value}>{parkingLabel(value)}</option>)}</select></label>
                <label>Payment method<select name="payment" value={preferences.payment} onChange={onChange}><option value="">Any payment method</option>{options.payments.map((value) => <option key={value} value={value}>{paymentLabel(value)}</option>)}</select></label>
  </>;
}

export default function App() {
  const [options, setOptions] = useState(null);
  const [optionsError, setOptionsError] = useState("");
  const [retry, setRetry] = useState(0);
  const [preferences, setPreferences] = useState({ budget: "any", cuisine: "", area: "", parking: "", payment: "", top_k: 6 });
  const [mode, setMode] = useState("visitor");
  const [users, setUsers] = useState(null);
  const [usersError, setUsersError] = useState("");
  const [usersRetry, setUsersRetry] = useState(0);
  const [selectedUser, setSelectedUser] = useState("");
  const [userTopK, setUserTopK] = useState(5);
  const [savedLoading, setSavedLoading] = useState(false);
  const [savedNote, setSavedNote] = useState("");
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  // A request version prevents an older response from overwriting newer preferences.
  const generation = useRef(0);

  // Load supported preferences on startup and when the visitor retries.
  useEffect(() => {
    document.title = BRAND;
    let active = true;
    setOptionsError("");
    request("/discovery/options")
      .then((data) => { if (active) setOptions(data); })
      .catch(() => { if (active) setOptionsError("We couldn’t load the catalog. Check that the API is running, then retry."); });
    return () => { active = false; };
  }, [retry]);

  // The existing /users endpoint lists IDs with embeddings in the saved ANN.
  // Load these only for the existing-user demo; this is a dataset selector, not login.
  useEffect(() => {
    if (mode !== "existing") return;
    let active = true;
    setUsers(null);
    setUsersError("");
    request("/users")
      .then((data) => {
        if (!active) return;
        setUsers(data.users);
        setSelectedUser((current) => data.users.includes(current) ? current : (data.users[0] || ""));
      })
      .catch(() => { if (active) setUsersError("We couldn’t load saved users. Check that the API is running, then retry."); });
    return () => { active = false; };
  }, [mode, usersRetry]);

  // Changing mode or inputs invalidates any request still in flight.
  function clearShortlist() {
    generation.current += 1;
    setResults(null);
    setError("");
    setLoading(false);
    setSavedLoading(false);
    setSavedNote("");
  }

  function switchMode(nextMode) {
    clearShortlist();
    setMode(nextMode);
  }

  // Clear the old shortlist whenever a form value changes.
  function updatePreference(event) {
    const { name, value } = event.target;
    clearShortlist();
    setPreferences((current) => ({ ...current, [name]: name === "top_k" ? Number(value) : value }));
  }

  // Saved tastes are optional form defaults, not new ANN inputs or a model retrain.
  async function applySavedPreferences() {
    clearShortlist();
    const current = generation.current;
    setSavedLoading(true);
    try {
      const saved = await request(`/users/${encodeURIComponent(selectedUser)}/preferences`);
      if (current !== generation.current) return;
      const cuisine = saved.cuisines.find((value) => options.cuisines.includes(value)) || "";
      const payment = saved.payments.find((value) => options.payments.includes(value)) || "";
      setPreferences((previous) => ({ ...previous, budget: saved.budget, cuisine, payment }));
      setSavedNote(`Loaded ${selectedUser}’s saved budget and first supported cuisine/payment choice where available. Review or change them before submitting. Parking and dining space stay as selected.`);
    } catch (err) {
      if (current === generation.current) setError(err.message);
    } finally {
      if (current === generation.current) setSavedLoading(false);
    }
  }

  // Each mode uses its own real backend endpoint and its own score label.
  async function discover(event) {
    event.preventDefault();
    const current = ++generation.current;
    setLoading(true);
    setError("");
    setResults(null);
    try {
      // Existing users use saved embeddings; visitors use exact preference filters.
      // Attach the method to ANN responses because the legacy API omits that field.
      const query = new URLSearchParams({ top_k: String(userTopK), budget: preferences.budget });
      if (preferences.cuisine) query.set("cuisine", preferences.cuisine);
      if (preferences.area) query.set("area", preferences.area);
      if (preferences.parking) query.set("parking", preferences.parking);
      if (preferences.payment) query.set("payment", preferences.payment);
      const data = mode === "existing" ? {
        ...await request(`/recommend/${encodeURIComponent(selectedUser)}?${query}`),
        method: "ann",
      } : await request("/recommend/preferences", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...preferences, cuisine: preferences.cuisine || null, area: preferences.area || null, parking: preferences.parking || null, payment: preferences.payment || null }),
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

  const annResults = results?.method === "ann";

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
            <p className="intro">Explore by budget, cuisine, and dining space—or discover your next meal using a saved user’s rating history.</p>
            <a className="text-link" href="#discover">Find your next table <span aria-hidden="true">↗</span></a>
          </div>
          <aside className="hero-note" aria-label="About this edition">
            <span className="note-symbol" aria-hidden="true">✳</span><p className="eyebrow">THE DISCOVERY EDITION</p>
            <h2>Real records.<br />An honest starting point.</h2>
            <p>Explore a historical restaurant catalog. No invented reviews, photos, or promises.</p>
            <span className="note-footer">Two ways to find your next table</span>
          </aside>
        </section>
        <div className="data-notice"><strong>Demo data · Mexico</strong><p>{options?.demo_notice || "Explore historical UCI restaurant records from Mexico, plus records with unknown country. Current details and availability are unverified."}</p></div>
        <section id="discover" className="search-panel" aria-labelledby="discover-heading">
          <div className="section-title"><div><p className="eyebrow">MAKE IT YOURS</p><h2 id="discover-heading">{mode === "existing" ? "A new pick for a familiar taste." : "What sounds good?"}</h2></div><span>{mode === "existing" ? "Saved dataset users" : "No account needed"}</span></div>
          <div className="discovery-modes" role="group" aria-label="Recommendation mode">
            <button type="button" aria-pressed={mode === "visitor"} onClick={() => switchMode("visitor")}>New visitor <span>Choose your preferences</span></button>
            <button type="button" aria-pressed={mode === "existing"} onClick={() => switchMode("existing")}>Existing user <span>Use past ratings</span></button>
          </div>
          {mode === "existing" ? <>
            <p className="mode-description">Choose a saved user and what they feel like today. We match their preferences, exclude already-rated restaurants, then rank the remaining options using the ANN.</p>
            {usersError ? <div role="alert" className="error">{usersError} <button type="button" onClick={() => setUsersRetry((value) => value + 1)}>Retry users</button></div> : !users ? <p role="status">Loading saved users…</p> : users.length === 0 ? <p role="status">No saved users are available in this model.</p> : <form onSubmit={discover}>
              <div className="form-grid user-form-grid">
                <label>Saved user ID<select value={selectedUser} onChange={(event) => { clearShortlist(); setSelectedUser(event.target.value); }}>{users.map((user) => <option key={user} value={user}>{user}</option>)}</select></label>
                <label>Top recommendations<select value={userTopK} onChange={(event) => { clearShortlist(); setUserTopK(Number(event.target.value)); }}>{[3, 5, 10].map((count) => <option key={count} value={count}>Top {count} restaurants</option>)}</select></label>
              </div>
              <div className="saved-preferences"><button className="secondary-button" type="button" disabled={savedLoading || loading || !selectedUser || !options || Boolean(optionsError)} onClick={applySavedPreferences}>{savedLoading ? "Loading saved preferences…" : "Use saved preferences"}</button><p>Optional: use this dataset user’s recorded budget, cuisine, and payment choices.</p></div>
              {savedNote && <p className="mode-description" role="status">{savedNote}</p>}
              {optionsError ? <div role="alert" className="error">{optionsError} <button type="button" onClick={() => setRetry((value) => value + 1)}>Retry catalog</button></div> : !options ? <p role="status">Loading available preferences…</p> : <div className="form-grid preference-grid"><PreferenceFields options={options} preferences={preferences} onChange={updatePreference} /></div>}
              <div className="form-bottom"><p>Your preferences → Unrated matches → ANN ranking → Up to {userTopK} picks<br />Demo IDs from the historical dataset; no sign-in required.</p><button className="primary-button" disabled={loading || savedLoading || !selectedUser || !options || Boolean(optionsError)} type="submit">{loading ? "Finding your picks…" : "Get My Recommendations"} <span aria-hidden="true">↗</span></button></div>
            </form>}
          </> : optionsError ? <div role="alert" className="error">{optionsError} <button onClick={() => setRetry((value) => value + 1)}>Retry catalog</button></div> : !options ? <p role="status">Loading available preferences…</p> : (
            <form onSubmit={discover}>
              <div className="form-grid">
                <PreferenceFields options={options} preferences={preferences} onChange={updatePreference} />
                <label>Suggestions<select name="top_k" value={preferences.top_k} onChange={updatePreference}>{[3, 6, 12].map((count) => <option key={count} value={count}>{count} restaurants</option>)}</select></label>
              </div>
              <div className="form-bottom"><p>Exact preference matches. Ranked using historical ratings.</p><button className="primary-button" disabled={loading} type="submit">{loading ? "Finding restaurants…" : "Find Restaurants"} <span aria-hidden="true">↗</span></button></div>
            </form>
          )}
        </section>
        <div aria-live="polite" aria-atomic="true">{loading && <p className="status">{mode === "existing" ? "Predicting ratings with the saved ANN and ranking unseen restaurants…" : "Matching your preferences and ranking the catalog…"}</p>}{error && <p className="error" role="alert">{error}</p>}{results && <p className="sr-only">{results.total_recommendations} restaurants found.</p>}</div>
        {results && <section className="results" aria-labelledby="results-heading">
          <div className="section-title"><div><p className="eyebrow">{annResults ? "BASED ON PAST RATINGS" : "A PLACE TO START"}</p><h2 id="results-heading">{annResults ? `Picks for ${results.user_id}` : "Your restaurant shortlist"}</h2></div><span>{results.total_recommendations} results</span></div>
          <p className="results-note">{annResults ? "Your preference matches · Ranked by the ANN using saved rating history · Predicted ratings on the original 0–2 scale · Already-rated restaurants excluded. Fewer matches may be returned than requested." : "Preference matches · Bayesian ranking · Scores are ranking signals on a 0–2 scale, not customer ratings or ANN predictions."}</p>
          {results.recommendations.length === 0 ? <div className="empty"><h3>{annResults ? "No unrated restaurants match these preferences." : "No exact matches this time."}</h3><p>{annResults ? "Try any budget, any cuisine, or any dining space—or select another saved user." : "Try any cuisine, a different budget, or any dining space to explore more of the catalog."}</p></div> : <div className="restaurant-grid">
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
              <div className="score"><strong>{(annResults ? restaurant.predicted_rating : restaurant.score).toFixed(2)}<small> / 2</small></strong><span>{annResults ? "ANN predicted rating" : "Bayesian ranking score"}</span></div>
              <p className="explanation">{annResults ? `Estimated for ${results.user_id} using learned user and restaurant embeddings after applying your preference filters. This restaurant has not been rated by this user in the dataset.` : restaurant.explanation}</p>
              <details className="recorded-details"><summary>Recorded amenities &amp; hours</summary>
                <p><strong>Parking:</strong> {restaurant.parking.length ? restaurant.parking.map(parkingLabel).join(", ") : "Unrecorded"}</p>
                <p><strong>Payments:</strong> {restaurant.payments.length ? restaurant.payments.map(paymentLabel).join(", ") : "Unrecorded"}</p>
                <p><strong>Historical hours · not live opening status</strong></p>
                {restaurant.recorded_hours.length ? <ul>{restaurant.recorded_hours.map((entry, i) => <li key={i}>{entry.days.replaceAll(";", ", ")}: {entry.hours.replaceAll(";", "; ")}</li>)}</ul> : <p>Hours unrecorded</p>}
              </details>
              <div className="card-footer"><span>{annResults ? "Saved ANN model" : `${restaurant.rating_count} historical ratings`}</span><span>Demo record</span></div>
              <a className="maps-link" href={mapsSearchUrl(restaurant)} target="_blank" rel="noopener noreferrer" aria-label={`Search for ${restaurant.name} on Google Maps (opens in a new tab)`}>
                <span><svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d="M20 10c0 6-8 12-8 12S4 16 4 10a8 8 0 1 1 16 0Z" /><circle cx="12" cy="10" r="2.5" /></svg>Search Google Maps</span>
                <span className="maps-arrow" aria-hidden="true">↗</span>
              </a>
              <p className="maps-note">Uses recorded name and location · Listing unverified</p>
            </article>)}
          </div>}
        </section>}
        <section id="about" className="about" aria-labelledby="about-heading"><div><p className="eyebrow">A CLEARER KIND OF DISCOVERY</p><h2 id="about-heading">Good suggestions.<br />No mystery.</h2></div><div><h3>New here? Start with preferences.</h3><p>Budget, recorded cuisine, and dining space narrow the catalog. Bayesian ranking balances each restaurant’s average with 10 global-average ratings to soften scores from small samples.</p><h3>A saved user? Let history help.</h3><p>Select an existing dataset user and optional budget, cuisine, and dining-space preferences. We filter matching restaurants, exclude already-rated places, then use the trained ANN to predict ratings and return the highest-ranked matches.</p></div></section>
      </main>
      <footer><span>{BRAND}</span><span>UCI demo catalog · Historical data · Current details unverified</span></footer>
    </div>
  );
}
