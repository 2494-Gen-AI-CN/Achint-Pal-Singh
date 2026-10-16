
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Query

from train import RecommenderANN
from data import load_data


MODEL_PATH = Path(__file__).resolve().parent / "models" / "recommender.pt"

model = None
user_map = {}
restaurant_map = {}
ratings = None
restaurants = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, user_map, restaurant_map, ratings, restaurants

    # Load the model saved by train.py
    if not MODEL_PATH.exists():
        raise RuntimeError("Model not found. Run python train.py first.")

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

    # Load restaurant details and historical ratings
    ratings, restaurants = load_data()

    print("ANN recommendation model loaded successfully.")

    yield


app = FastAPI(
    title="Restaurant Recommendation System",
    description="PyTorch ANN Collaborative Filtering API",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
def home():
    return {
        "message": "Restaurant Recommendation API is running",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.get("/restaurants")
def get_restaurants():
    return restaurants.to_dict(orient="records")


@app.get("/users")
def get_users():
    return {
        "total_users": len(user_map),
        "users": [str(user_id) for user_id in user_map]
    }


@app.get("/recommend/{user_id}")
def recommend(
    user_id: str,
    top_k: int = Query(default=5, ge=1, le=50)
):
    if user_id not in user_map:
        raise HTTPException(
            status_code=404,
            detail="User not found in training data"
        )

    # Restaurants the user has already rated
    visited = set(
        ratings.loc[
            ratings["userID"] == user_id,
            "placeID"
        ].tolist()
    )

    # Recommend only known, unrated restaurants
    candidates = [
        place_id
        for place_id in restaurant_map
        if place_id not in visited
    ]

    if not candidates:
        return {
            "user_id": user_id,
            "recommendations": []
        }

    # Prepare input tensors
    user_tensor = torch.full(
        (len(candidates),),
        user_map[user_id],
        dtype=torch.long
    )

    restaurant_tensor = torch.tensor(
        [restaurant_map[place_id] for place_id in candidates],
        dtype=torch.long
    )

    # Predict ratings without training
    with torch.inference_mode():
        scores = model(
            user_tensor,
            restaurant_tensor
        ).tolist()

    # Rank restaurants by predicted rating
    ranked = sorted(
        zip(candidates, scores),
        key=lambda item: item[1],
        reverse=True
    )[:top_k]

    recommendations = []

    for place_id, score in ranked:
        match = restaurants[
            restaurants["placeID"] == place_id
        ]

        if match.empty:
            continue

        restaurant = match.iloc[0]

        recommendations.append({
            "restaurant_id": int(place_id),
            "name": restaurant["name"],
            "price": str(restaurant["price"]),
            "predicted_rating": round(float(score) * 2, 2)
        })

    return {
        "user_id": user_id,
        "total_recommendations": len(recommendations),
        "recommendations": recommendations
    }
