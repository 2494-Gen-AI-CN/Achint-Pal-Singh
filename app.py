
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from train import RecommenderANN
from data import load_data
from recommendations import Preferences, available_cuisines, load_catalog, recommend_preferences


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


# -----------------------------
# 2. Load model on startup
# -----------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, user_map, restaurant_map, ratings, restaurants, catalog

    if not MODEL_PATH.exists():
        raise RuntimeError(
            "Trained model not found. Run python train.py first."
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

    print("ANN recommendation model loaded successfully.")

    yield


# -----------------------------
# 3. Initialize FastAPI
# -----------------------------

app = FastAPI(
    title="Restaurant Recommendation System",
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
    return {
        "message": "Restaurant Recommendation API is running",
        "docs": "/docs"
    }


# -----------------------------
# 6. Health check
# -----------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "unavailable",
        "model_loaded": model is not None
    }


# -----------------------------
# 7. Get available users
# -----------------------------

@app.get("/users")
def get_users():
    return {
        "total_users": len(user_map),
        "users": [str(user_id) for user_id in user_map]
    }


# -----------------------------
# 8. Get restaurants
# -----------------------------

@app.get("/restaurants")
def get_restaurants():
    return restaurants.to_dict(orient="records")


# -----------------------------
# 9. ANN recommendations
# -----------------------------

@app.get("/discovery/options")
def discovery_options():
    if catalog is None:
        raise HTTPException(status_code=503, detail="Discovery catalog is unavailable")
    return {
        "budgets": ["low", "medium", "high"],
        "cuisines": available_cuisines(catalog),
        "areas": sorted(catalog["area"].dropna().unique().tolist()),
        "restaurant_count": len(catalog),
        "dataset": "UCI Restaurant & Consumer Ratings",
        "countries": sorted(catalog["country"].dropna().unique().tolist()),
        "unknown_country_count": int(catalog["country"].isna().sum()),
        "demo_notice": "Demo catalog: Mexican restaurant records and records with unknown country. No verified California coverage. Historical data; current availability is unverified.",
    }


@app.post("/recommend/preferences")
def discover(preferences: Preferences):
    if catalog is None:
        raise HTTPException(status_code=503, detail="Discovery catalog is unavailable")
    if preferences.cuisine and preferences.cuisine not in available_cuisines(catalog):
        raise HTTPException(status_code=422, detail="Choose a cuisine from /discovery/options")
    return recommend_preferences(catalog, preferences)


@app.get("/recommend/{user_id}")
def recommend(
    user_id: str,
    top_k: int = Query(default=5, ge=1, le=50)
):
    # Validate user
    if user_id not in user_map:
        raise HTTPException(
            status_code=404,
            detail="User not found in training data"
        )

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
        if place_id not in visited
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

    # ANN inference
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

        recommendations.append({
            "restaurant_id": int(place_id),
            "name": str(restaurant["name"]),
            "price": str(restaurant["price"]),
            "predicted_rating": round(float(score) * 2, 2)
        })

    return {
        "user_id": user_id,
        "total_recommendations": len(recommendations),
        "recommendations": recommendations
    }
