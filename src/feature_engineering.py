"""
feature_engineering.py
Transform raw movie data into features for modeling.
"""

import pandas as pd
import numpy as np
from datetime import datetime


def load_raw_data(filepath: str = "data/raw/movies_raw.csv") -> pd.DataFrame:
    """Load the raw movie dataset."""
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} movies")
    return df


def engineer_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal features from release date."""
    
    df = df.copy()
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    
    # Drop rows with invalid dates
    df = df.dropna(subset=["release_date"])
    
    # Basic date features
    df["release_year"] = df["release_date"].dt.year
    df["release_month"] = df["release_date"].dt.month
    df["release_day"] = df["release_date"].dt.day
    df["release_dayofweek"] = df["release_date"].dt.dayofweek  # 0=Monday, 6=Sunday
    df["release_quarter"] = df["release_date"].dt.quarter
    
    # Is it a weekend release? (Friday=4, Saturday=5, Sunday=6)
    df["is_weekend_release"] = df["release_dayofweek"].isin([4, 5, 6]).astype(int)
    
    # Summer release (May-August)
    df["is_summer_release"] = df["release_month"].isin([5, 6, 7, 8]).astype(int)
    
    # Holiday season (November-December)
    df["is_holiday_release"] = df["release_month"].isin([11, 12]).astype(int)
    
    # Days since a reference date (for capturing trends)
    reference_date = datetime(2000, 1, 1)
    df["days_since_2000"] = (df["release_date"] - reference_date).dt.days
    
    return df


def engineer_genre_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode genres and create genre-based features."""
    
    df = df.copy()
    
    # Define main genres to track
    main_genres = [
        "Action", "Adventure", "Animation", "Comedy", "Crime", 
        "Documentary", "Drama", "Family", "Fantasy", "History",
        "Horror", "Music", "Mystery", "Romance", "Science Fiction",
        "Thriller", "War", "Western"
    ]
    
    # One-hot encode each genre
    for genre in main_genres:
        col_name = f"genre_{genre.lower().replace(' ', '_')}"
        df[col_name] = df["genres"].str.contains(genre, case=False, na=False).astype(int)
    
    return df


def engineer_cast_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create aggregate cast/crew features."""
    
    df = df.copy()
    
    # Average popularity of top 3 actors
    actor_cols = ["actor_1_popularity", "actor_2_popularity", "actor_3_popularity"]
    df["avg_actor_popularity"] = df[actor_cols].mean(axis=1)
    df["max_actor_popularity"] = df[actor_cols].max(axis=1)
    
    # Total star power (director + actors)
    df["total_star_power"] = (
        df["director_popularity"] + 
        df["actor_1_popularity"] + 
        df["actor_2_popularity"] + 
        df["actor_3_popularity"]
    )
    
    # Has A-list talent (popularity > 20 is roughly A-list)
    df["has_popular_director"] = (df["director_popularity"] > 20).astype(int)
    df["has_popular_lead"] = (df["actor_1_popularity"] > 20).astype(int)
    
    return df


def engineer_budget_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create budget-related features."""
    
    df = df.copy()
    
    # Log transform budget (reduces skew)
    df["log_budget"] = np.log1p(df["budget"])
    
    # Budget tiers
    df["is_low_budget"] = (df["budget"] < 15_000_000).astype(int)
    df["is_mid_budget"] = ((df["budget"] >= 15_000_000) & (df["budget"] < 100_000_000)).astype(int)
    df["is_blockbuster_budget"] = (df["budget"] >= 100_000_000).astype(int)
    
    return df


def engineer_studio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features based on production studio (known before release)."""
    
    df = df.copy()
    
    # Major studios that historically produce high-grossing films
    major_studios = [
        "Warner Bros", "Universal Pictures", "Walt Disney", "Disney",
        "Paramount", "20th Century", "Sony Pictures", "Columbia Pictures",
        "Marvel Studios", "Lucasfilm", "Pixar", "DreamWorks",
        "Lionsgate", "New Line Cinema", "Legendary"
    ]
    
    # Check if movie has a major studio
    def has_major_studio(companies_str):
        if pd.isna(companies_str):
            return 0
        for studio in major_studios:
            if studio.lower() in companies_str.lower():
                return 1
        return 0
    
    df["has_major_studio"] = df["production_companies"].apply(has_major_studio)
    
    # Specific studio indicators for top performers
    df["is_disney"] = df["production_companies"].str.contains(
        "Disney|Pixar|Marvel Studios|Lucasfilm", case=False, na=False
    ).astype(int)
    df["is_warner"] = df["production_companies"].str.contains(
        "Warner Bros|DC|New Line", case=False, na=False
    ).astype(int)
    df["is_universal"] = df["production_companies"].str.contains(
        "Universal Pictures|Illumination|Amblin", case=False, na=False
    ).astype(int)
    
    return df


def engineer_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create historical performance features for directors and actors.
    
    Uses only past performance data to avoid leakage - for each movie,
    we calculate the historical average revenue of the director/actors
    based only on their films released BEFORE this one.
    """
    
    df = df.copy()
    df = df.sort_values("release_date").reset_index(drop=True)
    
    # Calculate expanding historical averages for directors
    director_history = {}
    director_avg_revenue = []
    director_film_count = []
    
    for _, row in df.iterrows():
        director = row["director_name"]
        if pd.notna(director) and director in director_history:
            past_revenues = director_history[director]
            director_avg_revenue.append(np.mean(past_revenues))
            director_film_count.append(len(past_revenues))
        else:
            director_avg_revenue.append(0)
            director_film_count.append(0)
        
        # Update history with this movie's revenue (for future movies)
        if pd.notna(director):
            if director not in director_history:
                director_history[director] = []
            director_history[director].append(row["revenue"])
    
    df["director_historical_avg_revenue"] = director_avg_revenue
    df["director_historical_avg_log_revenue"] = np.log1p(df["director_historical_avg_revenue"])
    df["director_prior_films"] = director_film_count
    df["is_director_debut"] = (df["director_prior_films"] == 0).astype(int)
    
    # Calculate for lead actor
    actor_history = {}
    actor_avg_revenue = []
    actor_film_count = []
    
    for _, row in df.iterrows():
        actor = row["actor_1_name"]
        if pd.notna(actor) and actor in actor_history:
            past_revenues = actor_history[actor]
            actor_avg_revenue.append(np.mean(past_revenues))
            actor_film_count.append(len(past_revenues))
        else:
            actor_avg_revenue.append(0)
            actor_film_count.append(0)
        
        if pd.notna(actor):
            if actor not in actor_history:
                actor_history[actor] = []
            actor_history[actor].append(row["revenue"])
    
    df["lead_actor_historical_avg_revenue"] = actor_avg_revenue
    df["lead_actor_historical_avg_log_revenue"] = np.log1p(df["lead_actor_historical_avg_revenue"])
    df["lead_actor_prior_films"] = actor_film_count
    
    return df


def engineer_competition_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features based on release window competition."""
    
    df = df.copy()
    df["release_date"] = pd.to_datetime(df["release_date"])
    
    # Count movies released in the same month/year
    df["year_month"] = df["release_date"].dt.to_period("M")
    monthly_counts = df.groupby("year_month").size().to_dict()
    df["movies_same_month"] = df["year_month"].map(monthly_counts)
    
    # High-budget competition in same month
    df["is_high_budget"] = (df["budget"] >= 100_000_000).astype(int)
    monthly_blockbusters = df.groupby("year_month")["is_high_budget"].sum().to_dict()
    df["blockbusters_same_month"] = df["year_month"].map(monthly_blockbusters)
    
    # Drop temporary columns
    df = df.drop(columns=["year_month", "is_high_budget"])
    
    return df


def engineer_keyword_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features from movie keywords (known before release from marketing)."""
    
    df = df.copy()
    
    # High-performing keyword categories
    keyword_categories = {
        "superhero": ["superhero", "super hero", "marvel", "dc comics", "comic book"],
        "sequel_keyword": ["sequel", "based on novel", "based on comic"],
        "family_friendly": ["animation", "family", "children", "pixar", "animated"],
        "action_heavy": ["explosion", "car chase", "fight", "battle", "war"],
        "romance": ["love", "romance", "relationship", "wedding"],
        "scifi": ["alien", "space", "robot", "future", "dystopia", "time travel"],
        "horror_keyword": ["horror", "ghost", "haunted", "monster", "zombie", "vampire"],
    }
    
    for category, keywords in keyword_categories.items():
        pattern = "|".join(keywords)
        df[f"keyword_{category}"] = df["keywords"].str.contains(
            pattern, case=False, na=False
        ).astype(int)
    
    return df


def engineer_title_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features from movie title."""
    
    df = df.copy()
    
    # Title length (shorter titles often more memorable)
    df["title_length"] = df["title"].str.len()
    df["title_word_count"] = df["title"].str.split().str.len()
    
    # Sequel indicators from title
    sequel_patterns = r"\b(2|3|4|5|II|III|IV|V|Part|Chapter|Episode|Returns|Reloaded|Rises|Awakens)\b"
    df["title_suggests_sequel"] = df["title"].str.contains(
        sequel_patterns, case=False, na=False
    ).astype(int)
    
    return df


def engineer_certification_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features from MPAA certification/rating."""
    
    df = df.copy()
    
    # Standardize certification values
    cert_mapping = {
        "G": "G",
        "PG": "PG", 
        "PG-13": "PG-13",
        "R": "R",
        "NC-17": "NC-17",
        "NR": "NR",
        "": "Unknown",
    }
    
    df["certification"] = df["certification"].fillna("").str.strip()
    df["certification_clean"] = df["certification"].map(lambda x: cert_mapping.get(x, "Other"))
    
    # One-hot encode certifications
    for cert in ["G", "PG", "PG-13", "R"]:
        df[f"cert_{cert.lower().replace('-', '')}"] = (df["certification_clean"] == cert).astype(int)
    
    # Family-friendly indicator (G or PG)
    df["is_family_friendly_cert"] = df["certification_clean"].isin(["G", "PG"]).astype(int)
    
    # Adult content indicator
    df["is_r_rated"] = (df["certification_clean"] == "R").astype(int)
    
    return df


def engineer_marketing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features related to marketing and promotional reach."""
    
    df = df.copy()
    
    # Video/trailer features (already in raw data)
    if "num_trailers" in df.columns:
        df["has_trailer"] = (df["num_trailers"] > 0).astype(int)
        df["has_multiple_trailers"] = (df["num_trailers"] > 1).astype(int)
    
    if "num_teasers" in df.columns:
        df["has_teaser"] = (df["num_teasers"] > 0).astype(int)
    
    # Marketing lead time - how early was first trailer?
    if "days_trailer_before_release" in df.columns:
        df["early_marketing"] = (df["days_trailer_before_release"] > 180).astype(int)  # 6+ months
        df["late_marketing"] = (df["days_trailer_before_release"] < 30).astype(int)  # Less than a month
    
    # Social media presence
    if "social_media_presence" in df.columns:
        df["has_strong_social"] = (df["social_media_presence"] >= 2).astype(int)
    
    # International reach
    if "num_translations" in df.columns:
        df["log_translations"] = np.log1p(df["num_translations"])
        df["high_international_reach"] = (df["num_translations"] > 30).astype(int)
    
    if "num_release_countries" in df.columns:
        df["wide_release"] = (df["num_release_countries"] > 20).astype(int)
    
    return df


def engineer_crew_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional crew-related features."""
    
    df = df.copy()
    
    # Total crew power
    crew_cols = ["writer_popularity", "producer_popularity", "composer_popularity", "cinematographer_popularity"]
    existing_cols = [c for c in crew_cols if c in df.columns]
    if existing_cols:
        df["total_crew_popularity"] = df[existing_cols].sum(axis=1)
    
    # Has famous composer (Hans Zimmer, John Williams effect)
    if "composer_popularity" in df.columns:
        df["has_popular_composer"] = (df["composer_popularity"] > 10).astype(int)
    
    # Production team size
    prod_cols = ["num_writers", "num_producers", "num_exec_producers"]
    existing_prod = [c for c in prod_cols if c in df.columns]
    if existing_prod:
        df["production_team_size"] = df[existing_prod].sum(axis=1)
    
    return df


def engineer_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """Create and transform the target variable."""
    
    df = df.copy()
    
    # Log transform revenue (our target)
    df["log_revenue"] = np.log1p(df["revenue"])
    
    # ROI (return on investment) - alternative target
    df["roi"] = (df["revenue"] - df["budget"]) / df["budget"]
    
    # Profit
    df["profit"] = df["revenue"] - df["budget"]
    df["is_profitable"] = (df["profit"] > 0).astype(int)
    
    return df


def select_model_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select features for modeling and return X, y.
    
    IMPORTANT: We exclude post-release features to prevent data leakage:
    - vote_average, vote_count, popularity: These are only available after release
    - These would not be known when predicting box office before a movie opens
    """
    
    # Define feature columns - ONLY pre-release information
    numeric_features = [
        # Budget features (known before release)
        "budget", "log_budget", 
        "is_low_budget", "is_mid_budget", "is_blockbuster_budget",
        
        # Movie attributes (known before release)
        "runtime", "num_genres", "num_production_companies", "num_cast",
        "num_keywords", "is_franchise",
        
        # Release timing (known before release)
        "release_year", "release_month", "release_dayofweek", "release_quarter",
        "is_weekend_release", "is_summer_release", "is_holiday_release",
        "days_since_2000",
        
        # Cast/crew popularity (known before release - based on prior work)
        "director_popularity", "actor_1_popularity", "actor_2_popularity", 
        "actor_3_popularity", "avg_actor_popularity", "max_actor_popularity",
        "total_star_power", "has_popular_director", "has_popular_lead",
        
        # Studio features (known before release)
        "has_major_studio", "is_disney", "is_warner", "is_universal",
        
        # Historical performance (no leakage - uses only past data)
        "director_historical_avg_log_revenue", "director_prior_films", "is_director_debut",
        "lead_actor_historical_avg_log_revenue", "lead_actor_prior_films",
        
        # Competition features
        "movies_same_month", "blockbusters_same_month",
        
        # Title features
        "title_length", "title_word_count", "title_suggests_sequel",
        
        # Certification features (known before release)
        "cert_g", "cert_pg", "cert_pg13", "cert_r",
        "is_family_friendly_cert", "is_r_rated",
        
        # Marketing features (known before release)
        # Note: num_videos excluded - additional clips added post-release
        "num_trailers", "num_teasers",
        "has_trailer", "has_multiple_trailers", "has_teaser",
        "days_trailer_before_release", "early_marketing", "late_marketing",
        "social_media_presence", "has_strong_social",
        "has_tagline", "tagline_length", "has_homepage", "overview_length",
        
        # EXCLUDED (potential leakage - accumulates post-release):
        # - num_translations: translations added after release as movie becomes popular
        # - num_videos: additional videos/clips added post-release
        
        # International reach (known before release - initial release plan)
        # Note: num_translations excluded due to leakage
        "num_release_countries", "wide_release",
        "num_production_countries", "is_us_production",
        "num_spoken_languages", "is_english",
        
        # Additional crew features
        "num_writers", "num_producers", "num_exec_producers",
        "writer_popularity", "producer_popularity", 
        "composer_popularity", "cinematographer_popularity",
        "total_crew_popularity", "has_popular_composer", "production_team_size",
        
        # EXCLUDED (data leakage - only available after release):
        # - vote_average: audience ratings come after release
        # - vote_count: vote counts accumulate post-release  
        # - popularity: TMDB popularity based on current user activity
    ]
    
    # Add genre columns
    genre_cols = [col for col in df.columns if col.startswith("genre_")]
    
    # Add keyword category columns
    keyword_cols = [col for col in df.columns if col.startswith("keyword_")]
    feature_cols = numeric_features + genre_cols + keyword_cols
    
    # Filter to only existing columns
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    X = df[feature_cols].copy()
    y = df["log_revenue"].copy()
    
    # Handle any remaining NaN values
    X = X.fillna(0)
    
    return X, y


def main():
    """Main feature engineering pipeline."""
    
    print("Loading raw data...")
    df = load_raw_data()
    
    print("Engineering date features...")
    df = engineer_date_features(df)
    
    print("Engineering genre features...")
    df = engineer_genre_features(df)
    
    print("Engineering cast features...")
    df = engineer_cast_features(df)
    
    print("Engineering budget features...")
    df = engineer_budget_features(df)
    
    print("Engineering studio features...")
    df = engineer_studio_features(df)
    
    print("Engineering historical performance features...")
    df = engineer_historical_features(df)
    
    print("Engineering competition features...")
    df = engineer_competition_features(df)
    
    print("Engineering keyword features...")
    df = engineer_keyword_features(df)
    
    print("Engineering title features...")
    df = engineer_title_features(df)
    
    print("Engineering certification features...")
    df = engineer_certification_features(df)
    
    print("Engineering marketing features...")
    df = engineer_marketing_features(df)
    
    print("Engineering additional crew features...")
    df = engineer_crew_features(df)
    
    print("Engineering target variable...")
    df = engineer_target_variable(df)
    
    # Save full engineered dataset
    df.to_csv("data/processed/movies_engineered.csv", index=False)
    print(f"Saved engineered data: {len(df)} movies, {len(df.columns)} columns")
    
    # Prepare modeling dataset
    print("\nPreparing modeling dataset...")
    X, y = select_model_features(df)
    
    # Save modeling data
    X.to_csv("data/processed/X_features.csv", index=False)
    y.to_csv("data/processed/y_target.csv", index=False)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Features: {list(X.columns)}")
    
    return df, X, y


if __name__ == "__main__":
    main()
