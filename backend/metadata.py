"""Supplementary UCI metadata: payments, parking, historic hours, and saved tastes."""

import pandas as pd

from .data import DATA_DIR


def grouped_values(filename, key, value, normalize=False):
    """Collapse repeated CSV rows to one sorted list per ID without duplicating ratings."""
    rows = pd.read_csv(DATA_DIR / filename, usecols=[key, value]).dropna()
    rows[value] = rows[value].str.strip()
    rows = rows.loc[~rows[value].isin(["", "?"])]
    if normalize:
        # VISA and Visa must match the same payment method across restaurant/user files.
        rows[value] = rows[value].str.casefold()
    return rows.groupby(key)[value].agg(lambda values: sorted(set(values))).to_dict()


def enrich_catalog(catalog):
    """Join supplementary details only to the app's existing restaurant IDs."""
    for column, filename, source, normalize in (
        ("payments", "chefmozaccepts.csv", "Rpayment", True),
        ("parking", "chefmozparking.csv", "parking_lot", False),
    ):
        mapping = grouped_values(filename, "placeID", source, normalize)
        catalog[column] = catalog["placeID"].apply(lambda place: mapping.get(place, []))

    # Hours are historical strings, not live opening status. Keep days paired with hours.
    rows = pd.read_csv(DATA_DIR / "chefmozhours4.csv").dropna().drop_duplicates()
    hours = {}
    for row in rows.itertuples(index=False):
        if row.hours == "?" or row.days == "?":
            continue
        hours.setdefault(row.placeID, []).append({
            "days": row.days.strip("; "), "hours": row.hours.strip("; "),
        })
    catalog["recorded_hours"] = catalog["placeID"].apply(lambda place: hours.get(place, []))
    return catalog


def load_user_preferences():
    """Read saved cuisine/payment choices and budget; omit unrelated personal fields."""
    cuisines = grouped_values("usercuisine.csv", "userID", "Rcuisine")
    payments = grouped_values("userpayment.csv", "userID", "Upayment", normalize=True)
    profiles = pd.read_csv(DATA_DIR / "userprofile.csv", usecols=["userID", "budget"])
    profiles = profiles.drop_duplicates("userID").set_index("userID")
    result = {}
    for user in set(cuisines) | set(payments) | set(profiles.index):
        budget = profiles.at[user, "budget"] if user in profiles.index else None
        result[user] = {
            "budget": budget if budget in ("low", "medium", "high") else "any",
            "cuisines": cuisines.get(user, []), "payments": payments.get(user, []),
        }
    return result


def extra_details(row):
    """Return recorded amenities for cards in both recommendation paths."""
    return {key: row[key] for key in ("parking", "payments", "recorded_hours")}
