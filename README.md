# California Restaurant Guide — demo edition

An upgrade of the existing FastAPI / PyTorch ANN / React application. No model retraining or architecture changes are required.

## Data and scope

The existing UCI catalog contains 130 restaurants: 102 explicitly list Mexico, and 28 have unknown country. Cities include San Luis Potosi, Cuernavaca, Jiutepec, and Ciudad Victoria, with inconsistent spelling in source records. This is historical demo data, not a California restaurant directory. Current opening status, prices, addresses, and availability have not been verified. No restaurant images or reviews are fabricated.

## Run locally (PowerShell)

From the existing project root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

In a second terminal:

```powershell
cd frontend
npm run dev -- --host 127.0.0.1 --port 5173
```

Use the existing installed Python and npm dependencies. Required backend packages are torch, pandas, scikit-learn, fastapi, and uvicorn. Existing files in `data/` and the trusted `models/recommender.pt` checkpoint are required. Startup deliberately fails if these assets cannot load. The model is loaded on CPU as before.

Optional frontend configuration uses Vite's existing environment support (set before running/building):

```powershell
$env:VITE_GUIDE_NAME = "Restaurant Discovery Demo"
$env:VITE_API_URL = "http://127.0.0.1:8000"
```

The default brand is California Restaurant Guide. Changing branding does not change data coverage or remove demo disclosures. The API CORS policy permits GET and POST from localhost / 127.0.0.1 on port 5173; use that port for local development.

## Recommendation methods

- **Known users:** `GET /recommend/{user_id}` retains the existing checkpoint, user/item mappings, ANN inference, visited-item exclusion, and `predicted_rating` response field on the original 0–2 scale.
- **New visitors:** `POST /recommend/preferences` applies exact metadata filters and sorts by Bayesian average: `(n × restaurant_mean + 10 × global_mean) / (n + 10)`. Both means use historical ratings on the original 0–2 scale. Ten global-average pseudo-ratings reduce the effect of small samples; this is a transparent heuristic, not a tuned model or evaluation claim. Ties use rating count descending, then restaurant ID ascending. Restaurants without historical ratings are excluded.

Budget categories are dataset labels, not dollar ranges or a maximum-spend guarantee. Cuisine comes from `chefmozcuisine.csv`. Dining space maps the dataset's `open` / `closed` area field to outdoor / indoor; it says nothing about current opening hours or seat availability. Unknown fields never satisfy an explicit filter. Filters are not silently relaxed when there are no matches. New visitor scores are labeled ranking signals, not individual reviews, ANN predictions, or unseen-user personalization.

The metadata endpoint `GET /discovery/options` provides supported options and the coverage notice. The legacy `/`, `/health`, `/users`, and `/restaurants` endpoints remain available.

## API examples

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/discovery/options
Invoke-RestMethod 'http://127.0.0.1:8000/recommend/U1014?top_k=5'
Invoke-RestMethod http://127.0.0.1:8000/recommend/preferences -Method Post -ContentType 'application/json' -Body '{"budget":"medium","top_k":5}'
Invoke-RestMethod http://127.0.0.1:8000/recommend/preferences -Method Post -ContentType 'application/json' -Body '{"budget":"low","cuisine":"Mexican","area":"closed","top_k":6}'
```

Budget accepts `any`, `low`, `medium`, `high`; area accepts `null`, `open`, `closed`; cuisine accepts `null` or an exact option from the metadata endpoint. `top_k` must be an integer from 1 to 50. Unsupported values and unknown fields return 422. No matches returns 200 with an empty recommendations array. Each recommendation includes ID, name, price, score, method, explanation, historical mean/count, cuisines, area, and recorded location.

## Validation

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
cd frontend
npm run build
npm run lint
```

Backend tests use the standard library and a temporary local Uvicorn server with the real checkpoint and dataset. They verify legacy endpoints, ANN results, geographic disclosures, all budget filters, combined cuisine/area filtering, deterministic Bayesian scores against source ratings, invalid inputs, empty results, and browser POST preflight. No additional test dependency is needed.

The frontend handles metadata loading/retry, request timeouts/errors, results, and empty states. Editing preferences clears previous results; submit again to refresh the list. Responses from older submissions cannot overwrite a changed selection. Layout adapts to desktop, tablet, and mobile, with keyboard focus indicators and live status messages.

Datasets, checkpoints, virtual environments, npm dependencies, and build output are ignored. Existing tracked assets, if any, must be managed manually by the project owner. No Git write operations or deployment are part of this upgrade.

## Upgrade verification (October 15, 2026)

All 8 backend regression tests passed with the actual checkpoint and dataset. The checkpoint SHA-256 and all 15 sampled recommendations across U1014, U1001, and U1100 exactly matched a baseline captured before the upgrade. The production build and oxlint passed using existing dependencies. No model training or evaluation metrics were added.

A temporary headless Chrome check passed against actual local API/Vite servers: low/high budget results, preference-change clearing, demo/score labels, desktop and 375px/320px mobile layouts without horizontal overflow, empty results, API failure messages, loading, obsolete-response suppression, and metadata retry. No browser JavaScript exceptions occurred. Desktop/mobile screenshots were visually reviewed. The temporary harness required no new project dependencies and its servers were stopped after testing.
