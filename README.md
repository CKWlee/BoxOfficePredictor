# box office predictor

predict how much a movie will make at the box office using pre-release info. budget, cast, genre, release timing, franchise status, marketing signals — 103 features total.

trained on 1,919 films from TMDB. XGBoost model, R² = 0.71.

**[try the live demo →](https://letsseeyourfakemovie.streamlit.app/)**

---

## how it works

```
TMDB API → collect 1,919 films (hits + flops)
     ↓
feature engineering → 103 features
     ↓
model comparison → Linear Reg (0.62) vs Random Forest (0.62) vs XGBoost (0.71)
     ↓
serve via FastAPI + Streamlit dashboard
```

the pipeline collects data using 4 strategies (by popularity, by year, low revenue flops, high revenue blockbusters) to avoid training bias. feature engineering covers temporal patterns, genre encoding, cast/crew popularity, studio signals, marketing indicators, and historical track records.

---

## results

| model | R² |
|---|---|
| linear regression | 0.62 |
| random forest | 0.62 |
| **xgboost** | **0.71** |

top features by importance: log budget, lead actor historical revenue, director historical revenue, total star power, release timing.

---

## stack

| | |
|---|---|
| ml | XGBoost, scikit-learn |
| tracking | MLflow |
| api | FastAPI |
| dashboard | Streamlit |
| data | TMDB API, pandas |
| infra | Docker Compose |

---

## run it

**quick start**
```bash
docker-compose up
```
- api: http://localhost:8000
- dashboard: http://localhost:8501
- mlflow: http://localhost:5000

**api example**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"budget": 150000000, "genres": ["Action", "Adventure"], "runtime": 140, "release_month": 7, "is_franchise": 1}'
```

**train from scratch**
```bash
pip install -r requirements.txt
python src/collect_data.py    # grab data from tmdb
python src/feature_engineering.py  # build features
python src/train_with_mlflow.py    # train + log to mlflow
```

add your TMDB key to `.env`:
```
TMDB_API_KEY=your_key_here
```

---

## project structure

```
├── src/
│   ├── collect_data.py          # tmdb data collection (4 strategies)
│   ├── feature_engineering.py   # 103 features
│   ├── train_model.py           # model comparison
│   └── train_with_mlflow.py     # training with experiment tracking
├── server/
│   └── main.py                  # fastapi prediction endpoint
├── app.py                       # streamlit dashboard
├── docker-compose.yml
├── data/
│   ├── raw/                     # from tmdb
│   └── processed/               # engineered features
└── models/
    └── best_model.joblib        # trained xgboost
```
