# Box Office Revenue Predictor

**[Try the live demo →](https://letsseeyourfakemovie.streamlit.app/)**

Predicting how much money a movie will make at the box office using machine learning. Built this to explore what actually drives box office success, still movies are a piece of the times so not everything can be explained.

## What It Does

Takes pre-release info about a movie (budget, cast, release date, genre, etc.) and predicts worldwide box office revenue. Also includes a Streamlit web app where you can build a hypothetical movie and see what it might earn.

## The Data

Pulled ~2,600 movies from TMDB API and ended up with 1,919 that had complete budget/revenue data.

**Sampling approach**: Unfortunately, this isn't a completely random sample as a statistics course would like. I deliberately pulled movies across different revenue tiers (flops, mid-performers, blockbusters) to make sure the model could learn from failures, not just hits. The dataset is biased toward "notable" films (ones with 50+ votes on TMDB), but that's fine since the model is meant for theatrical releases, not micro-budget films that never hit theaters.

**What's in the data:**
- Revenue range: $7 to $2.9B
- 350 movies that lost money (18%)
- Budget range: $2K to $584M
- Years: 1939-2025 (mostly 2010s-2020s)
- English-language theatrical releases only

## Features Used (103 total)

- **Budget**: Raw budget, log-transformed, budget tier flags
- **Cast/Crew**: Director + actor popularity scores, their historical avg revenue, prior film counts
- **Timing**: Release month, day of week, summer/holiday flags, competition that month
- **Production**: Major studio flags (Disney, Warner, Universal), franchise indicator
- **Marketing signals**: Trailer count, social media presence, tagline length
- **Genre**: One-hot encoded, plus keyword flags (superhero, sequel, etc.)

**Excluded to avoid data leakage:**
- Vote average/count (comes after release)
- TMDB popularity score (changes over time)
- Translation count (accumulates post-release)

## Results

| Model | RMSE (log) | R² Score |
|-------|------------|----------|
| Linear Regression | 1.19 | 0.62 |
| Ridge Regression | 1.19 | 0.62 |
| Random Forest | 1.18 | 0.62 |
| **XGBoost** | **1.03** | **0.71** |

XGBoost explains 71% of the variance in box office revenue using only pre-release information. Not bad considering how unpredictable movies can be.

## Run It Yourself

1. Clone and install:
```bash
git clone https://github.com/yourusername/box-office-predictor.git
cd box-office-predictor
pip install -r requirements.txt
```

2. Get a TMDB API key from https://www.themoviedb.org/ (free, takes 2 min)

3. Create `.env` file:
```
TMDB_API_KEY=your_key_here
```

4. Run the pipeline:
```bash
python src/collect_data.py         # ~35 min (API rate limits)
python src/feature_engineering.py  # ~10 sec
python src/train_model.py          # ~30 sec
```

5. Launch the web app:
```bash
streamlit run app.py
```

## Project Structure
```
box-office-predictor/
├── data/
│   ├── raw/              # Raw TMDB data
│   └── processed/        # Engineered features
├── models/               # Saved model (.joblib)
├── src/
│   ├── collect_data.py
│   ├── feature_engineering.py
│   └── train_model.py
├── app.py                # Streamlit dashboard
├── requirements.txt
└── README.md
```

## Limitations

- English-language films only
- Biased toward movies notable enough to have TMDB data
- No marketing spend data (studios don't publish this)
- Pre-COVID and post-COVID box office dynamics may differ

## License
MIT
