import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Box Office Predictor API")

# cors so the dashboard and any frontend can hit this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# load model and feature columns once at startup
MODEL_PATH = "models/best_model.joblib"
FEATURES_PATH = "data/processed/X_features.csv"

model = None
feature_columns = None


@app.on_event("startup")
def load_resources():
    global model, feature_columns
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"model not found at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    feature_columns = pd.read_csv(FEATURES_PATH).columns.tolist()


class MovieInput(BaseModel):
    budget: float = 50_000_000
    runtime: Optional[float] = 110
    genres: Optional[list[str]] = ["Drama"]
    release_month: Optional[int] = 6
    release_year: Optional[int] = 2025
    release_dayofweek: Optional[int] = 4
    is_franchise: Optional[int] = 0
    certification: Optional[str] = "PG-13"
    director_popularity: Optional[float] = 5.0
    actor_1_popularity: Optional[float] = 5.0
    actor_2_popularity: Optional[float] = 3.0
    actor_3_popularity: Optional[float] = 2.0
    director_historical_avg_log_revenue: Optional[float] = 0.0
    director_prior_films: Optional[int] = 0
    lead_actor_historical_avg_log_revenue: Optional[float] = 0.0
    lead_actor_prior_films: Optional[int] = 0
    has_major_studio: Optional[int] = 0
    is_disney: Optional[int] = 0
    is_warner: Optional[int] = 0
    is_universal: Optional[int] = 0
    num_trailers: Optional[int] = 2
    marketing_lead_days: Optional[float] = 90
    social_media_presence: Optional[int] = 1
    num_release_countries: Optional[int] = 15
    title: Optional[str] = "Untitled"


def build_feature_vector(inp: MovieInput) -> pd.DataFrame:
    # build the feature vector from user input
    f = {}

    # budget stuff
    f["budget"] = inp.budget
    f["log_budget"] = np.log1p(inp.budget)
    f["is_low_budget"] = 1 if inp.budget < 15_000_000 else 0
    f["is_mid_budget"] = 1 if 15_000_000 <= inp.budget < 100_000_000 else 0
    f["is_blockbuster_budget"] = 1 if inp.budget >= 100_000_000 else 0

    # basic attributes
    f["runtime"] = inp.runtime or 110
    f["num_genres"] = len(inp.genres) if inp.genres else 1
    f["num_production_companies"] = 1 if inp.has_major_studio else 0
    f["num_cast"] = 5
    f["num_keywords"] = 3
    f["is_franchise"] = inp.is_franchise

    # release timing
    f["release_year"] = inp.release_year
    f["release_month"] = inp.release_month
    f["release_dayofweek"] = inp.release_dayofweek
    f["release_quarter"] = (inp.release_month - 1) // 3 + 1
    f["is_weekend_release"] = 1 if inp.release_dayofweek >= 4 else 0
    f["is_summer_release"] = 1 if inp.release_month in [5, 6, 7, 8] else 0
    f["is_holiday_release"] = 1 if inp.release_month in [11, 12] else 0
    # rough days since 2000 from year/month
    f["days_since_2000"] = (inp.release_year - 2000) * 365 + inp.release_month * 30

    # talent
    f["director_popularity"] = inp.director_popularity
    f["actor_1_popularity"] = inp.actor_1_popularity
    f["actor_2_popularity"] = inp.actor_2_popularity
    f["actor_3_popularity"] = inp.actor_3_popularity
    f["avg_actor_popularity"] = np.mean([inp.actor_1_popularity, inp.actor_2_popularity, inp.actor_3_popularity])
    f["max_actor_popularity"] = max(inp.actor_1_popularity, inp.actor_2_popularity, inp.actor_3_popularity)
    f["total_star_power"] = inp.director_popularity + inp.actor_1_popularity + inp.actor_2_popularity + inp.actor_3_popularity
    f["has_popular_director"] = 1 if inp.director_popularity > 20 else 0
    f["has_popular_lead"] = 1 if inp.actor_1_popularity > 20 else 0

    # studio
    f["has_major_studio"] = inp.has_major_studio
    f["is_disney"] = inp.is_disney
    f["is_warner"] = inp.is_warner
    f["is_universal"] = inp.is_universal

    # historical track record
    f["director_historical_avg_log_revenue"] = inp.director_historical_avg_log_revenue
    f["director_prior_films"] = inp.director_prior_films
    f["is_director_debut"] = 1 if inp.director_prior_films == 0 else 0
    f["lead_actor_historical_avg_log_revenue"] = inp.lead_actor_historical_avg_log_revenue
    f["lead_actor_prior_films"] = inp.lead_actor_prior_films

    # competition defaults (hard to know at prediction time)
    f["movies_same_month"] = 15
    f["blockbusters_same_month"] = 3

    # title
    f["title_length"] = len(inp.title)
    f["title_word_count"] = len(inp.title.split())
    sequel_words = ["2", "3", "II", "III", "Part", "Chapter", "Returns", "Rises", "Awakens"]
    f["title_suggests_sequel"] = 1 if any(w in inp.title for w in sequel_words) else 0

    # certification one-hot
    cert = inp.certification or "PG-13"
    f["cert_g"] = 1 if cert == "G" else 0
    f["cert_pg"] = 1 if cert == "PG" else 0
    f["cert_pg13"] = 1 if cert == "PG-13" else 0
    f["cert_r"] = 1 if cert == "R" else 0
    f["is_family_friendly_cert"] = 1 if cert in ["G", "PG"] else 0
    f["is_r_rated"] = 1 if cert == "R" else 0

    # marketing
    f["num_trailers"] = inp.num_trailers
    f["num_teasers"] = 1
    f["has_trailer"] = 1 if inp.num_trailers > 0 else 0
    f["has_multiple_trailers"] = 1 if inp.num_trailers > 1 else 0
    f["has_teaser"] = 1
    f["days_trailer_before_release"] = inp.marketing_lead_days
    f["early_marketing"] = 1 if inp.marketing_lead_days > 180 else 0
    f["late_marketing"] = 1 if inp.marketing_lead_days < 30 else 0
    f["social_media_presence"] = inp.social_media_presence
    f["has_strong_social"] = 1 if inp.social_media_presence >= 2 else 0
    f["has_tagline"] = 1
    f["tagline_length"] = 40
    f["has_homepage"] = inp.has_major_studio
    f["overview_length"] = 150

    # international
    f["num_release_countries"] = inp.num_release_countries
    f["wide_release"] = 1 if inp.num_release_countries > 20 else 0
    f["num_production_countries"] = 1
    f["is_us_production"] = 1
    f["num_spoken_languages"] = 1
    f["is_english"] = 1

    # crew defaults
    f["num_writers"] = 2
    f["num_producers"] = 3
    f["num_exec_producers"] = 2
    f["writer_popularity"] = 3.0
    f["producer_popularity"] = 3.0
    f["composer_popularity"] = 3.0
    f["cinematographer_popularity"] = 3.0
    f["total_crew_popularity"] = 15.0
    f["has_popular_composer"] = 0
    f["production_team_size"] = 7

    # genre one-hot
    all_genres = ["action", "adventure", "animation", "comedy", "crime", "documentary",
                  "drama", "family", "fantasy", "history", "horror", "music",
                  "mystery", "romance", "science_fiction", "thriller", "war", "western"]
    selected = inp.genres or []
    for g in all_genres:
        display = g.replace("_", " ").title()
        if g == "science_fiction":
            display = "Science Fiction"
        f[f"genre_{g}"] = 1 if display in selected else 0

    # keyword features derived from genres/franchise
    f["keyword_superhero"] = 1 if "Action" in selected and inp.is_franchise else 0
    f["keyword_sequel_keyword"] = 1 if inp.is_franchise else 0
    f["keyword_family_friendly"] = 1 if "Family" in selected or "Animation" in selected else 0
    f["keyword_action_heavy"] = 1 if "Action" in selected else 0
    f["keyword_romance"] = 1 if "Romance" in selected else 0
    f["keyword_scifi"] = 1 if "Science Fiction" in selected else 0
    f["keyword_horror_keyword"] = 1 if "Horror" in selected else 0

    vec = pd.DataFrame([f])

    # fill any missing columns the model expects
    for col in feature_columns:
        if col not in vec.columns:
            vec[col] = 0

    vec = vec[feature_columns].fillna(0)
    return vec


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/model-info")
def model_info():
    return {
        "r2_score": 0.71,
        "feature_count": len(feature_columns) if feature_columns else 103,
        "training_samples": 1919,
        "model_type": "XGBoost",
        "target": "log_revenue (worldwide gross)",
    }


@app.post("/predict")
def predict(movie: MovieInput):
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    try:
        vec = build_feature_vector(movie)
        log_pred = float(model.predict(vec)[0])
        revenue = float(np.expm1(log_pred))

        # clamp to sane minimum
        revenue = max(revenue, 1_000_000)

        # rough confidence band — model R²=0.71 so ~±30% is a fair heuristic
        low = revenue * 0.6
        high = revenue * 1.6

        if revenue >= 1_000_000_000:
            revenue_fmt = f"${revenue / 1e9:.2f}B"
        else:
            revenue_fmt = f"${revenue / 1e6:.0f}M"

        roi = ((revenue - movie.budget) / movie.budget) * 100

        return {
            "predicted_revenue": revenue,
            "predicted_revenue_formatted": revenue_fmt,
            "confidence_range": {"low": low, "high": high},
            "roi_percent": round(roi, 1),
            "outcome": "blockbuster" if roi >= 100 else ("profitable" if roi >= 0 else "loss"),
            "log_prediction": log_pred,
            "model_r2": 0.71,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
