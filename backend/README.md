# Backend code walkthrough

All commands below run from the project root. Imports use the `backend` package.
Inline comments and function docstrings explain the main calculations and decisions.

## File and function map

| File | Functions | Responsibility |
| --- | --- | --- |
| `dataset.py` | `download_dataset` | Download the UCI ZIP and copy the three required CSVs into `backend/data/`. |
| `data.py` | `load_data` | Read ratings and restaurant details, remove invalid/duplicate rows, and normalize ratings from 0-2 to 0-1. |
| `data.py` | `prepare_data`, nested `move_cold_rows` | Hold out ratings, create train/validation/test sets, ensure evaluation IDs exist in training, and encode IDs as indices. |
| `train.py` | `RecommenderANN.__init__`, `forward` | Define the embedding tables and rating network; combine user and restaurant vectors to predict a rating. |
| `train.py` | `make_dataset` | Convert encoded IDs and targets into tensors. |
| `train.py` | `evaluate` | Calculate MAE/RMSE without updating weights. |
| `train.py` | `train_model` | Train with MSE/Adam, retain lowest validation RMSE weights, evaluate test ratings, and save the checkpoint. |
| `recommendations.py` | `Preferences` | Validate the new-visitor request with a Pydantic schema. |
| `recommendations.py` | `load_catalog`, `available_cuisines` | Build metadata/rating statistics and the supported cuisine list. |
| `recommendations.py` | `recommend_preferences` | Filter metadata exactly, rank by Bayesian score, and format explanations. |
| `app.py` | `lifespan` | Load saved weights, maps, ratings, and catalog at server startup. |
| `app.py` | `home`, `health`, `get_users`, `get_restaurants` | Basic API status and catalog endpoints. |
| `app.py` | `discovery_options`, `discover` | Supply frontend options and handle preference recommendations. |
| `app.py` | `recommend` | Apply optional metadata preferences, exclude already-rated restaurants, predict ratings for known users, and return top K matches. |
| `tests/test_app.py` | setup/teardown, request helper, ten test methods | Start a real local API and check both recommendation paths, scoring, filters, errors, and CORS. |

## Which code to show for the six-slide presentation

1. **Problem and product:** show `frontend/src/App.jsx` and describe restaurant discovery.
2. **Dataset and preparation:** open `data.py`: `load_data`, then `prepare_data`. ID encoding is a lookup mapping; embeddings are learned later. Test rows are held-out ratings, not necessarily unseen users.
3. **ANN and training:** open `train.py`: model constructor, `forward`, then the batch loop and best-validation selection in `train_model`. Sixteen user values plus sixteen restaurant values give 32 inputs; layers output 64, 32, and one normalized rating. MSE measures error; Adam updates weights. Thirty epochs are thirty full passes, with batches of up to 32.
4. **Two recommendation paths:** show `app.py:recommend` for known users and `recommendations.py:recommend_preferences` for new visitors. The preference form does not invoke the ANN. Bayesian scores stay on the original 0-2 scale; sigmoid belongs to the ANN.
5. **Frontend demo:** show `frontend/src/App.jsx`: select **New visitor** to submit preferences and see Bayesian ranking scores. Select **Existing user**, choose a saved ID such as U1014, keep **Top 5 restaurants**, choose optional budget/cuisine/dining-space filters, and click **Get My Recommendations** to show ANN predicted ratings. The user-loading effect fetches `/users`; `discover` calls the endpoint for the selected mode; `switchMode` and `clearShortlist` invalidate stale responses. `mapsSearchUrl` builds a search link using the recorded name and location.
6. **Validation and next steps:** show `tests/test_app.py` and `train.py:evaluate`. Passing API tests establishes functional behavior. MAE/RMSE measure rating error; precision/recall require a separate ranking evaluation and are not implemented here.

## Request flow

Preference form -> `POST /recommend/preferences` -> `Preferences` validation -> metadata filters -> Bayesian ordering -> JSON -> React cards.

Existing user selector -> `GET /users` -> select ID/top K/preferences -> `GET /recommend/{user_id}` -> validate filters -> matching catalog candidates -> saved ID maps -> exclude rated restaurants -> ANN inference -> top K -> JSON.

## Useful commands

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
.\.venv\Scripts\python.exe -m unittest discover -s backend/tests -t . -v
.\.venv\Scripts\python.exe -m backend.data
```

`python -m backend.train` trains and overwrites the saved checkpoint; it is not needed to serve recommendations.
All asset paths resolve relative to their backend source files, independent of the terminal working directory.

## Additional CSV metadata

`metadata.py` adds the extra datasets without changing the ANN architecture:

- `grouped_values`: clean and group repeated cuisine/payment/parking rows into unique lists keyed by ID; payment matching is case-normalized.
- `enrich_catalog`: join parking/payments and historical day/hour pairs to the existing catalog, without creating duplicate restaurants.
- `load_user_preferences`: read saved cuisine/payment lists and only the budget from userprofile.csv.
- `extra_details`: include recorded amenities and hours in both kinds of response.

`recommendations.py:filter_catalog` applies the same five metadata filters before either ranking path. `available_values` limits filter choices to actual catalog records. `app.py:validate_choices` rejects unsupported cuisine, parking, and payment values. `app.py:user_preferences` serves the optional saved-preference form action.

For slide 4, explain: extra context narrows candidates, then the ANN predicts ratings for known users or Bayesian ranking orders new-visitor matches. For slide 5, demonstrate parking/payment choices and **Use saved preferences**, then expand a card's **Recorded amenities & hours**. These hours are historical, not live status. Additional data use has not yet been shown to improve recommendation-quality metrics.

The frontend and API are named **Mexico Restaurant Guide**. Coverage remains 102 records explicitly listing Mexico and 28 with unknown country.
