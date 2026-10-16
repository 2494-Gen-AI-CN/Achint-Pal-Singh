
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .train import RecommenderANN
from .data import load_data
from .recommendations import (
    Preferences, available_cuisines, available_values, filter_catalog, load_catalog,
    recommend_preferences,
)
from .metadata import extra_details, load_user_preferences


# -----------------------------
# 1. Configuration
# -----------------------------

MODEL_PATH = Path(__file__).resolve().parent / "models" / "recommender.pt"

model = None
user_map = {}
restaurant_map = {}
ratings = None
restaurants = None
catalog = None
saved_preferences = {}


# -----------------------------
# 2. Load model on startup
# -----------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load saved ANN weights and the discovery catalog once when the API starts; do not retrain."""
    global model, user_map, restaurant_map, ratings, restaurants, catalog, saved_preferences

    if not MODEL_PATH.exists():
        raise RuntimeError(
            "Trained model not found. Run python -m backend.train from the project root first."
        )

    # Only load checkpoints created by your own trusted training code.
    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False
    )

    user_map = checkpoint["user_map"]
    restaurant_map = checkpoint["restaurant_map"]

    model = RecommenderANN(
        num_users=checkpoint["num_users"],
        num_restaurants=checkpoint["num_restaurants"]
    )

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    ratings, restaurants = load_data()
    catalog = load_catalog(ratings)
    saved_preferences = load_user_preferences()

    print("ANN recommendation model loaded successfully.")

    yield


# -----------------------------
# 3. Initialize FastAPI
# -----------------------------

app = FastAPI(
    title="Mexico Restaurant Guide",
    description="PyTorch ANN Collaborative Filtering API",
    version="1.0.0",
    lifespan=lifespan
)


# -----------------------------
# 4. Enable React CORS
# -----------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)


# -----------------------------
# 5. Home endpoint
# -----------------------------

@app.get("/")
def home():
    """Return a basic API welcome message and the interactive documentation address."""
    return {
        "message": "Restaurant Recommendation API is running",
        "docs": "/docs"
    }


# -----------------------------
# 6. Health check
# -----------------------------

@app.get("/health")
def health():
    """Report whether the saved ANN has been loaded into this server process."""
    return {
        "status": "healthy" if model is not None else "unavailable",
        "model_loaded": model is not None
    }


# -----------------------------
# 7. Get available users
# -----------------------------

@app.get("/users")
def get_users():
    """List known user IDs that have learned embeddings in the saved model."""
    return {
        "total_users": len(user_map),
        "users": [str(user_id) for user_id in user_map]
    }


# -----------------------------
# 8. Get restaurants
# -----------------------------

@app.get("/restaurants")
def get_restaurants():
    """Return restaurant metadata from the historical catalog."""
    return restaurants.to_dict(orient="records")


@app.get("/users/{user_id}/preferences")
def user_preferences(user_id: str):
    """Expose only a known demo user's saved dining choices, for optional form autofill."""
    if user_id not in user_map:
        raise HTTPException(status_code=404, detail="User not found in training data")
    return {"user_id": user_id, **saved_preferences.get(user_id, {
        "budget": "any", "cuisines": [], "payments": [],
    })}


def validate_choices(preferences):
    """Reject unrecorded categories instead of silently ignoring an explicit filter."""
    for value, choices, field in (
        (preferences.cuisine, available_cuisines(catalog), "cuisine"),
        (preferences.parking, available_values(catalog, "parking"), "parking"),
        (preferences.payment, available_values(catalog, "payments"), "payment"),
    ):
        if value and value not in choices:
            raise HTTPException(status_code=422, detail=f"Choose a {field} from /discovery/options")


# -----------------------------
# 9. New-visitor discovery and known-user ANN recommendations
# -----------------------------

@app.get("/discovery/options")
def discovery_options():
    """Provide valid form choices and dataset coverage information to React."""
    if catalog is None:
        raise HTTPException(status_code=503, detail="Discovery catalog is unavailable")
    return {
        "budgets": ["low", "medium", "high"],
        "cuisines": available_cuisines(catalog),
        "areas": sorted(catalog["area"].dropna().unique().tolist()),
        "parking": available_values(catalog, "parking"),
        "payments": available_values(catalog, "payments"),
        "restaurant_count": len(catalog),
        "dataset": "UCI Restaurant & Consumer Ratings",
        "countries": sorted(catalog["country"].dropna().unique().tolist()),
        "unknown_country_count": int(catalog["country"].isna().sum()),
        "demo_notice": "Demo catalog: Mexican restaurant records and records with unknown country. Historical data; current availability is unverified.",
    }


@app.post("/recommend/preferences")
def discover(preferences: Preferences):
    """Validate visitor preferences and call the metadata/Bayesian ranker, not the ANN."""
    if catalog is None:
        raise HTTPException(status_code=503, detail="Discovery catalog is unavailable")
    validate_choices(preferences)
    return recommend_preferences(catalog, preferences)


@app.get("/recommend/{user_id}")
def recommend(
    user_id: str,
    top_k: int = Query(default=5, ge=1, le=50),
    budget: Literal["any", "low", "medium", "high"] = "any",
    cuisine: str | None = Query(default=None, min_length=1, max_length=80),
    area: Literal["open", "closed"] | None = None,
    parking: str | None = Query(default=None, min_length=1, max_length=80),
    payment: str | None = Query(default=None, min_length=1, max_length=80),
):
    # Validate user
    """For a known user, exclude already-rated restaurants and rank candidates with the ANN."""
    if user_id not in user_map:
        raise HTTPException(
            status_code=404,
            detail="User not found in training data"
        )

    if catalog is None:
        raise HTTPException(status_code=503, detail="Discovery catalog is unavailable")
    preferences = Preferences(budget=budget, cuisine=cuisine, area=area,
                              parking=parking, payment=payment, top_k=top_k)
    validate_choices(preferences)

    # Preferences restrict which restaurants are eligible. The saved ANN still
    # scores and orders the matches using this known user's learned embedding.
    matches = filter_catalog(catalog, preferences)
    eligible = set(matches["placeID"])

    # Find restaurants already rated by this user
    visited = set(
        ratings.loc[
            ratings["userID"] == user_id,
            "placeID"
        ].tolist()
    )

    # Select unseen restaurants known to the model
    candidates = [
        place_id
        for place_id in restaurant_map
        if place_id not in visited and place_id in eligible
    ]

    if not candidates:
        return {
            "user_id": user_id,
            "total_recommendations": 0,
            "recommendations": []
        }

    # Convert IDs into model input tensors
    user_tensor = torch.full(
        (len(candidates),),
        user_map[user_id],
        dtype=torch.long
    )

    restaurant_tensor = torch.tensor(
        [restaurant_map[place_id] for place_id in candidates],
        dtype=torch.long
    )

    # ANN inference: use saved weights without training or recording gradients.
    with torch.inference_mode():
        scores = model(
            user_tensor,
            restaurant_tensor
        ).tolist()

    # Sort by highest predicted score
    ranked = sorted(
        zip(candidates, scores),
        key=lambda item: item[1],
        reverse=True
    )[:top_k]

    recommendations = []

    for place_id, score in ranked:
        match = restaurants.loc[
            restaurants["placeID"] == place_id
        ]

        if match.empty:
            continue

        restaurant = match.iloc[0]
        # Attach recorded metadata so the frontend can show why the filters match.
        details = catalog.loc[catalog["placeID"] == place_id].iloc[0]

        recommendations.append({
            "restaurant_id": int(place_id),
            "name": str(restaurant["name"]),
            "price": str(restaurant["price"]),
            # Restore the original 0-2 rating scale from the ANN sigmoid output.
            "predicted_rating": round(float(score) * 2, 2),
            "cuisines": details["cuisines"],
            "area": details["area"],
            "city": details["city"],
            "state": details["state"],
            "country": details["country"],
            **extra_details(details),
        })

    return {
        "user_id": user_id,
        "total_recommendations": len(recommendations),
        "recommendations": recommendations
    }
