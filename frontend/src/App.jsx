
import { useEffect, useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

export default function App() {
  const [users, setUsers] = useState([]);
  const [selectedUser, setSelectedUser] = useState("");
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [searched, setSearched] = useState(false);

  useEffect(() => {
    async function fetchUsers() {
      try {
        const response = await fetch(`${API_URL}/users`);

        if (!response.ok) {
          throw new Error("Unable to fetch users");
        }

        const data = await response.json();
        setUsers(data.users);

        if (data.users.length > 0) {
          setSelectedUser(data.users[0]);
        }
      } catch {
        setError("Cannot connect to FastAPI backend.");
      }
    }

    fetchUsers();
  }, []);

  async function getRecommendations() {
    if (!selectedUser) return;

    setLoading(true);
    setError("");
    setSearched(false);
    setRecommendations([]);

    try {
      const response = await fetch(
        `${API_URL}/recommend/${encodeURIComponent(selectedUser)}?top_k=5`
      );

      if (!response.ok) {
        throw new Error("Failed to fetch recommendations");
      }

      const data = await response.json();
      setRecommendations(data.recommendations);
      setSearched(true);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div className="logo">🍽️ Restaurant AI</div>
        <span className="header-tag">Powered by PyTorch ANN</span>
      </header>

      <main className="container">
        <section className="hero">
          <span className="eyebrow">PERSONALIZED DINING</span>
          <h1>Discover your next favorite restaurant.</h1>
          <p>
            Get personalized restaurant recommendations using
            artificial neural networks and collaborative filtering.
          </p>
        </section>

        <section className="search-panel">
          <div className="search-content">
            <label htmlFor="user">Select a user</label>

            <select
              id="user"
              value={selectedUser}
              onChange={(e) => setSelectedUser(e.target.value)}
            >
              {users.length === 0 && (
                <option value="">No users available</option>
              )}

              {users.map((user) => (
                <option key={user} value={user}>
                  {user}
                </option>
              ))}
            </select>

            <button
              onClick={getRecommendations}
              disabled={loading || !selectedUser}
            >
              {loading ? "Finding restaurants..." : "Get Recommendations →"}
            </button>
          </div>
        </section>

        {error && <div className="error">{error}</div>}

        {searched && (
          <section className="results">
            <div className="results-header">
              <div>
                <span className="eyebrow">CURATED FOR YOU</span>
                <h2>Recommended Restaurants</h2>
                <p>Personalized suggestions for {selectedUser}</p>
              </div>
              <span className="count">
                {recommendations.length} Results
              </span>
            </div>

            {recommendations.length === 0 ? (
              <p>No recommendations available for this user.</p>
            ) : (
              <div className="restaurant-grid">
                {recommendations.map((restaurant, index) => (
                  <article
                    className="restaurant-card"
                    key={restaurant.restaurant_id}
                  >
                    <div className="card-banner">
                      <span className="rank">#{index + 1} Recommended</span>
                      <span className="food-icon">🍴</span>
                    </div>

                    <div className="card-body">
                      <h3>{restaurant.name}</h3>

                      <div className="card-details">
                        <span className="price">
                          {restaurant.price} price
                        </span>
                        <span className="rating">
                          ★ {restaurant.predicted_rating.toFixed(2)} / 2
                        </span>
                      </div>

                      <div className="card-footer">
                        <span>AI predicted rating</span>
                        <span>ID: {restaurant.restaurant_id}</span>
                      </div>
                    </div>
                  </article>
                ))}
              </div>
            )}
          </section>
        )}
      </main>

      <footer>
        Restaurant AI · Built with React, FastAPI & PyTorch
      </footer>
    </div>
  );
}
