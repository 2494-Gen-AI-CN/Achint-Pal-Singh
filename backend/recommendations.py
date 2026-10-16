"""Cold-start discovery using metadata and historical UCI ratings, not ANN inference."""
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .data import DATA_DIR
from .metadata import enrich_catalog, extra_details


# FastAPI uses this schema to reject invalid form values before ranking restaurants.
class Preferences(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    budget: Literal["any", "low", "medium", "high"] = "any"
    cuisine: str | None = Field(default=None, min_length=1, max_length=80)
    area: Literal["open", "closed"] | None = None
    parking: str | None = Field(default=None, min_length=1, max_length=80)
    payment: str | None = Field(default=None, min_length=1, max_length=80)
    top_k: int = Field(default=5, ge=1, le=50)

# Combine real restaurant metadata with rating counts and Bayesian ranking scores.

def load_catalog(ratings):
    """Combine real restaurant metadata with rating counts and Bayesian ranking scores."""
    catalog = pd.read_csv(DATA_DIR / "geoplaces2.csv", encoding="latin-1")
    catalog = catalog[["placeID", "name", "price", "city", "state", "country", "area"]]
    catalog = catalog.drop_duplicates("placeID").replace("?", None)
    catalog["country"] = catalog["country"].str.strip().str.title()
    cuisines = pd.read_csv(DATA_DIR / "chefmozcuisine.csv")
    cuisines = cuisines.dropna().loc[lambda frame: frame["Rcuisine"] != "?"]
    cuisine_map = cuisines.groupby("placeID")["Rcuisine"].agg(lambda values: sorted(set(values)))
    catalog["cuisines"] = catalog["placeID"].map(cuisine_map).apply(
        lambda value: value if isinstance(value, list) else []
    )
    # load_data normalizes the original 0–2 ratings to 0–1 for the ANN.
    stats = ratings.groupby("placeID")["rating"].agg(["count", "mean"])
    catalog = catalog.merge(stats, left_on="placeID", right_index=True, how="left")
    catalog["count"] = catalog["count"].fillna(0).astype(int)
    prior_mean = float(ratings["rating"].mean()) * 2
    # Ten global-average pseudo-ratings soften extreme scores from tiny samples.
    # This is a ranking heuristic on the original 0-2 scale, separate from the ANN.
    prior_weight = 10
    catalog["historical_average"] = catalog["mean"] * 2
    catalog["score"] = (
        catalog["count"] * catalog["historical_average"].fillna(prior_mean)
        + prior_weight * prior_mean
    ) / (catalog["count"] + prior_weight)
    # Pandas 3 string columns can export missing text as NaN, which JSON rejects.
    return enrich_catalog(catalog.astype(object).where(catalog.notna(), None))


def available_cuisines(catalog):
    """Return the unique recorded cuisines for form options and API input validation."""
    return sorted({cuisine for values in catalog["cuisines"] for cuisine in values})


def available_values(catalog, column):
    """Offer only supplementary choices recorded for restaurants in this catalog."""
    return sorted({value for values in catalog[column] for value in values})


def filter_catalog(catalog, preferences):
    """Apply the same exact metadata constraints before either ranking method."""
    matches = catalog
    if preferences.budget != "any":
        matches = matches.loc[matches["price"] == preferences.budget]
    if preferences.cuisine:
        matches = matches.loc[matches["cuisines"].apply(lambda values: preferences.cuisine in values)]
    if preferences.area:
        matches = matches.loc[matches["area"] == preferences.area]
    for field, column in (("parking", "parking"), ("payment", "payments")):
        value = getattr(preferences, field)
        if value:
            matches = matches.loc[matches[column].apply(lambda values: value in values)]
    return matches


def recommend_preferences(catalog, preferences):
    """Filter exact visitor preferences, then return the highest Bayesian-ranked matches."""
    matches = filter_catalog(catalog.loc[catalog["count"] > 0], preferences)
    # Break score ties by evidence count, then ID, so repeated requests are stable.
    matches = matches.sort_values(["score", "count", "placeID"], ascending=[False, False, True])
    results = []
    for row in matches.head(preferences.top_k).to_dict("records"):
        reasons = []
        if preferences.budget != "any":
            reasons.append(f"Matches your {preferences.budget} budget")
        if preferences.cuisine:
            reasons.append(f"Listed cuisine: {preferences.cuisine.replace('_', ' ')}")
        if preferences.area:
            reasons.append(f"Listed dining area: {'outdoor' if preferences.area == 'open' else 'indoor'}")
        if preferences.parking:
            reasons.append(f"Recorded parking: {preferences.parking.replace('_', ' ')}")
        if preferences.payment:
            reasons.append(f"Recorded payment: {preferences.payment.replace('_', ' ')}")
        reasons.append(f"Ranked using {row['count']} historical ratings, smoothed with 10 global-average ratings")
        results.append({
            "restaurant_id": int(row["placeID"]), "name": row["name"],
            "price": row["price"], "score": round(row["score"], 4),
            "method": "cold_start_bayesian", "explanation": ". ".join(reasons) + ".",
            "rating_count": row["count"],
            "historical_average": round(row["historical_average"], 4),
            "cuisines": row["cuisines"], "area": row["area"],
            "city": row["city"], "state": row["state"], "country": row["country"],
            **extra_details(row),
        })
    return {"method": "cold_start_bayesian", "score_scale": [0, 2],
            "total_recommendations": len(results), "recommendations": results}
