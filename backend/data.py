
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent / "data"


def load_data():
    """Clean duplicate/missing ratings, join restaurant details, and scale ratings from 0-2 to 0-1."""
    ratings = pd.read_csv(DATA_DIR / "rating_final.csv")
    # This source CSV uses Latin-1 text encoding for restaurant names.
    restaurants = pd.read_csv(DATA_DIR / "geoplaces2.csv", encoding="latin-1")

    ratings = ratings[
        ["userID", "placeID", "rating"]
    ].dropna().copy()

    restaurants = restaurants[
        ["placeID", "name", "price"]
    ].drop_duplicates("placeID")

    ratings = ratings.drop_duplicates(
        subset=["userID", "placeID"]
    )

    ratings = ratings.merge(
        restaurants,
        on="placeID",
        how="inner"
    )

    ratings["rating"] = ratings["rating"].astype(float) / 2.0

    return ratings, restaurants


def prepare_data():
    """Create train/validation/test ratings and map known IDs to embedding-table indices."""
    ratings, restaurants = load_data()

    # Use only users and restaurants with sufficient interactions.
    # This also makes evaluation more reliable.
    for _ in range(5):
        user_counts = ratings.groupby("userID")["placeID"].transform("count")
        item_counts = ratings.groupby("placeID")["userID"].transform("count")

        filtered = ratings[
            (user_counts >= 3) & (item_counts >= 3)
        ].copy()

        if len(filtered) == len(ratings):
            break

        ratings = filtered

    if len(ratings) < 10:
        raise ValueError("Not enough ratings to train the model.")

    # Hold out one rating per user for the final test, then split the remaining
    # ratings for training/validation. These are held-out ratings, not new users.
    held_out = (
        ratings.groupby("userID", group_keys=False)
        .sample(n=1, random_state=42)
    )

    remaining = ratings.drop(index=held_out.index)

    train, validation = train_test_split(
        remaining,
        test_size=0.15,
        random_state=42
    )

    test = held_out.copy()

    # Ensure all IDs in validation/test were seen during training.
    known_users = set(train["userID"])
    known_places = set(train["placeID"])

    def move_cold_rows(source, training):
        """Move held-out ratings with unknown IDs into training so embedding lookup is possible."""
        cold = source[
            ~source["userID"].isin(known_users)
            | ~source["placeID"].isin(known_places)
        ]

        source = source.drop(index=cold.index)
        training = pd.concat([training, cold], ignore_index=True)

        return source, training

    validation, train = move_cold_rows(validation, train)

    known_users = set(train["userID"])
    known_places = set(train["placeID"])

    test, train = move_cold_rows(test, train)

    user_ids = sorted(train["userID"].unique())
    place_ids = sorted(train["placeID"].unique())

    # Encoding creates lookup indices only; embeddings are learned later in train.py.
    user_map = {user: idx for idx, user in enumerate(user_ids)}
    restaurant_map = {
        place: idx for idx, place in enumerate(place_ids)
    }

    for frame in (train, validation, test):
        frame["user_idx"] = frame["userID"].map(user_map)
        frame["restaurant_idx"] = frame["placeID"].map(
            restaurant_map
        )

    return (
        train,
        validation,
        test,
        restaurants,
        user_map,
        restaurant_map
    )


if __name__ == "__main__":
    train, val, test, restaurants, users, places = prepare_data()

    print("Training interactions:", len(train))
    print("Validation interactions:", len(val))
    print("Testing interactions:", len(test))
    print("Users:", len(users))
    print("Restaurants:", len(places))
    print("\nSample training data:")
    print(train.head())
