# Mexico Restaurant Guide — demo edition

An upgrade of the existing FastAPI / PyTorch ANN / React application. No model retraining or architecture changes are required.

## Data and scope

The existing UCI catalog contains 130 restaurants: 102 explicitly list Mexico, and 28 have unknown country. Cities include San Luis Potosi, Cuernavaca, Jiutepec, and Ciudad Victoria, with inconsistent spelling in source records. This is a historical Mexico restaurant discovery demo; records with unknown country remain clearly identified. Current opening status, prices, addresses, and availability have not been verified. No restaurant images or reviews are fabricated.

## Folder layout

```text
backend/                  Python API and machine learning
  app.py                  FastAPI startup and endpoints
  data.py                 CSV cleaning, split, and ID encoding
  dataset.py              UCI CSV downloader
  train.py                ANN definition, training, evaluation, checkpoint save
  recommendations.py      Shared preference filters and Bayesian ranking
  metadata.py             Parking, payments, historical hours, saved user preferences
  requirements.txt        Backend dependencies (current environment versions)
  README.md               Function map and presentation walkthrough
  data/                   Nine active UCI CSV inputs (local, ignored)
  models/                 Trusted trained checkpoint (local, ignored)
  tests/                  Backend API regression tests
frontend/                 React/Vite interface, styles, and npm configuration
archive/                  Original extra dataset and old installers/local files (ignored)
.venv/                    Existing local Python environment (ignored)
```

Run Python commands from the project root so `backend` resolves as a package.
The former root-level `app.py` command is now `backend.app:app`.
An empty old `data/` folder may remain while this app holds it as its working directory;
the application reads CSVs only from `backend/data/`.

## Run locally (PowerShell)

From the existing project root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```

In a second terminal:

```powershell
cd frontend
npm run dev -- --host 127.0.0.1 --port 5173
```

Use the existing installed Python and npm dependencies. Required backend packages are torch, pandas, scikit-learn, fastapi, and uvicorn. Existing files in `backend/data/` and the trusted `backend/models/recommender.pt` checkpoint are required. Startup deliberately fails if these assets cannot load. The model is loaded on CPU as before.

Optional frontend configuration uses Vite's existing environment support (set before running/building):

```powershell
$env:VITE_GUIDE_NAME = "Restaurant Discovery Demo"
$env:VITE_API_URL = "http://127.0.0.1:8000"
```

The default brand is Mexico Restaurant Guide. Changing branding does not change data coverage or remove demo disclosures. The API CORS policy permits GET and POST from localhost / 127.0.0.1 on port 5173; use that port for local development.

## Recommendation methods

- **Known users:** `GET /recommend/{user_id}` uses the existing checkpoint and user/item mappings. Optional `budget`, `cuisine`, and `area` query parameters filter the restaurant catalog first. Already-rated restaurants are excluded, and ANN predictions rank the remaining matches. `predicted_rating` stays on the original 0–2 scale. Unfiltered requests retain their original ANN ordering. Returned records include recorded cuisine, dining area, and location metadata.
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
.\.venv\Scripts\python.exe -m unittest discover -s backend/tests -t . -v
cd frontend
npm run build
npm run lint
```

Backend tests use the standard library and a temporary local Uvicorn server with the real checkpoint and dataset. They verify legacy endpoints, ANN results, geographic disclosures, all budget filters, combined cuisine/area filtering, deterministic Bayesian scores against source ratings, invalid inputs, empty results, and browser POST preflight. No additional test dependency is needed.

The frontend offers two modes: **New visitor** uses the preference form and Bayesian ranking; **Existing user** loads saved IDs from `/users` and calls `/recommend/{user_id}` for ANN predictions. Choose top 3, 5 (default), or 10 for an existing user, with optional budget, cuisine, and dining-space filters. Filters apply before ANN ranking; they do not retrain the model or become embedding inputs. Fewer than K results can be returned when fewer candidates match. Already-rated restaurants are excluded by the backend. Dataset user selection is a demonstration, not authentication. ANN cards show predicted ratings on the original 0-2 scale and recorded metadata for the preference matches. Their Maps links search the recorded name/location; listings remain unverified.

The frontend handles metadata/user loading and retry, request timeouts/errors, results, and empty states. Switching mode, changing a saved user, or changing the result count clears old results and ignores obsolete responses. Editing preferences clears previous results; submit again to refresh the list. Responses from older submissions cannot overwrite a changed selection. Layout adapts to desktop, tablet, and mobile, with keyboard focus indicators and live status messages.

Datasets, checkpoints, virtual environments, npm dependencies, and build output are ignored. Existing tracked assets, if any, must be managed manually by the project owner. No Git write operations or deployment are part of this upgrade.

## Upgrade verification (October 15, 2026)

All 8 backend regression tests passed with the actual checkpoint and dataset. The checkpoint SHA-256 and all 15 sampled recommendations across U1014, U1001, and U1100 exactly matched a baseline captured before the upgrade. The production build and oxlint passed using existing dependencies. No model training or evaluation metrics were added.

A temporary headless Chrome check passed against actual local API/Vite servers: low/high budget results, preference-change clearing, demo/score labels, desktop and 375px/320px mobile layouts without horizontal overflow, empty results, API failure messages, loading, obsolete-response suppression, and metadata retry. No browser JavaScript exceptions occurred. Desktop/mobile screenshots were visually reviewed. The temporary harness required no new project dependencies and its servers were stopped after testing.

## Data preparation and training commands

These commands are optional; serving the existing checkpoint does not retrain it.
Downloading replaces the active CSV inputs, and training writes a new checkpoint.

```powershell
.\.venv\Scripts\python.exe -m backend.dataset
.\.venv\Scripts\python.exe -m backend.data
.\.venv\Scripts\python.exe -m backend.train
```

For a fresh Python environment, install with `python -m pip install -r backend/requirements.txt`.
For a fresh frontend installation, run `npm ci` inside `frontend/`.
See `backend/README.md` for the function map and what code to show for each slide.
The previously exported PDF/PPT materials retain their old file paths and line numbers;
use this updated source map when navigating the reorganized project.

## Existing-user preference demo

1. Select **Existing user**, then saved user **U1014**.
2. Keep **Top 5 restaurants** and select a budget, cuisine, and dining space.
3. Click **Get My Recommendations**. These are actual ANN predictions for matching, previously unrated restaurants.
4. Change a filter and submit again to demonstrate a different eligible candidate set. With no matches, the interface shows an empty state rather than relaxing preferences.

```powershell
Invoke-RestMethod 'http://127.0.0.1:8000/recommend/U1014?top_k=5&budget=medium&cuisine=Mexican&area=closed'
```

Thirteen backend tests pass, including filtered ANN ordering against direct model predictions, exact metadata constraints, already-rated exclusion, invalid filters, and empty matches. Browser checks cover both recommendation modes, changing preferences, actual API scores, stale responses, user-list retry, and mobile layouts. No checkpoint retraining is part of this UI change.

## Supplementary datasets now used

The app uses all nine CSV files locally in `backend/data/`. `python -m backend.dataset` downloads the full set for a fresh setup. Existing rating/model files are preserved; joining supplementary rows does not duplicate restaurants or ratings.

| File | How it is used |
| --- | --- |
| `rating_final.csv` | User/restaurant ratings for ANN training and historical ranking statistics. |
| `geoplaces2.csv` | Names, budget categories, dining space, and recorded location. |
| `chefmozcuisine.csv` | Restaurant cuisine choices and filtering. |
| `chefmozparking.csv` | Exact parking-type filter and recorded parking details. `none` means no parking; unknown parking does not satisfy a filter. |
| `chefmozaccepts.csv` | Accepted-payment filter and details; VISA/Visa are normalized to the same option. |
| `chefmozhours4.csv` | Day/hour pairs shown under **Recorded amenities & hours**. These are historical strings, never a live open-now claim. |
| `usercuisine.csv` | Saved cuisine choices for an existing user. |
| `userpayment.csv` | Saved payment choices for an existing user. |
| `userprofile.csv` | Only the saved budget column is used; unrelated personal profile fields are not exposed. |

Both modes support optional `parking` and `payment` filters in addition to budget, cuisine, and dining space. Supported values come from `/discovery/options`, scoped to the 130 restaurants actually in this catalog. Missing values never count as a match; no matches returns an empty result.

Existing users can click **Use saved preferences**. `GET /users/{user_id}/preferences` returns their recorded budget, cuisines, and payments. The form applies the saved budget and first supported cuisine/payment choice when available, explains that choice, and lets the user edit it. Parking/dining space remain as selected. This action is optional and never silently changes ANN embeddings.

```powershell
Invoke-RestMethod 'http://127.0.0.1:8000/users/U1014/preferences'
Invoke-RestMethod 'http://127.0.0.1:8000/recommend/U1014?top_k=5&parking=free&payment=cash'
```

Supplementary metadata improves the detail and control available for matching; no measured improvement in user satisfaction or recommendation quality is claimed. The saved ANN still takes only user and restaurant IDs. New visitors still use Bayesian ranking after filtering.

The current 13-test suite also checks supplementary values against the source CSVs, preserves 130 unique restaurants and 1,161 rating counts, validates parking/payment filters in both paths, and verifies saved-user choices without exposing unrelated profile fields.
