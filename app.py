"""
Box Office Predictor - Interactive Dashboard
Predict how a hypothetical movie would perform at the box office.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, date

# Page config
st.set_page_config(
    page_title="Box Office Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Clean Light Theme
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    
    /* Headers */
    h1 {
        color: #1a73e8 !important;
        font-weight: 700 !important;
    }
    
    h2, h3 {
        color: #202124 !important;
        font-weight: 600 !important;
    }
    
    /* Metric cards */
    [data-testid="stMetric"] {
        background: #ffffff;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    [data-testid="stMetricLabel"] {
        color: #5f6368 !important;
        font-size: 0.9rem !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #1a73e8 !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #1a73e8;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 500;
        transition: background-color 0.2s, box-shadow 0.2s;
    }
    
    .stButton > button:hover {
        background-color: #1557b0;
        box-shadow: 0 2px 8px rgba(26, 115, 232, 0.3);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #f1f3f4;
        padding: 4px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px;
        padding: 10px 20px;
        color: #5f6368;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        color: #1a73e8;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Cards */
    .movie-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 24px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    
    /* Prediction context */
    .prediction-context {
        background: #e8f0fe;
        border-radius: 10px;
        padding: 16px;
        margin: 10px 0;
    }
    
    /* Info boxes */
    .info-box {
        background: #e8f0fe;
        border-left: 4px solid #1a73e8;
        padding: 16px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
        color: #202124;
    }
    
    /* Warning box */
    .warning-box {
        background: #fef7e0;
        border-left: 4px solid #f9ab00;
        padding: 16px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
        color: #202124;
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        background: #1a73e8 !important;
    }
    
    /* Divider */
    hr {
        border-color: #e0e0e0;
    }
    
    /* General text */
    p, span, label {
        color: #202124;
    }
</style>
""", unsafe_allow_html=True)


# Load model and data
@st.cache_resource
def load_model():
    return joblib.load("models/best_model.joblib")

@st.cache_data
def load_reference_data():
    """Load reference data for dropdowns and historical averages."""
    df = pd.read_csv("data/raw/movies_raw.csv")
    engineered = pd.read_csv("data/processed/movies_engineered.csv")
    return df, engineered

@st.cache_data
def get_unique_values(df):
    """Extract unique actors, directors, studios for dropdowns."""
    # Get unique directors with their popularity
    director_pop = df.groupby("director_name")["director_popularity"].mean().to_dict()
    directors = sorted(director_pop.keys(), key=lambda x: director_pop.get(x, 0), reverse=True)
    directors = [d for d in directors if isinstance(d, str)][:200]  # Top 200
    
    # Get unique actors
    actors = set()
    for col in ["actor_1_name", "actor_2_name", "actor_3_name"]:
        actors.update(df[col].dropna().unique().tolist())
    actors = sorted([a for a in actors if isinstance(a, str)])[:300]  # Top 300
    
    # Get production companies
    all_companies = df["production_companies"].dropna().str.split(",").explode().str.strip()
    company_counts = all_companies.value_counts()
    companies = company_counts.head(100).index.tolist()  # Top 100
    
    return directors, actors, companies

@st.cache_data
def get_person_popularity(df, name, role="director"):
    """Get average popularity for a person based on historical data."""
    if not name or name in ["(Select)", "(Unknown Director)", "(Unknown)"]:
        return 2.0  # Low default for unknown
    
    if role == "director":
        matches = df[df["director_name"] == name]["director_popularity"]
    else:
        matches = pd.concat([
            df[df["actor_1_name"] == name]["actor_1_popularity"],
            df[df["actor_2_name"] == name]["actor_2_popularity"],
            df[df["actor_3_name"] == name]["actor_3_popularity"],
        ])
    
    if len(matches) > 0:
        return matches.mean()
    return 2.0

@st.cache_data
def get_historical_revenue(df, name, role="director"):
    """Get historical average revenue for a person."""
    if not name or name in ["(Select)", "(Unknown Director)", "(Unknown)"]:
        return 0
        
    if role == "director":
        matches = df[df["director_name"] == name]["revenue"]
    else:
        matches = pd.concat([
            df[df["actor_1_name"] == name]["revenue"],
            df[df["actor_2_name"] == name]["revenue"],
            df[df["actor_3_name"] == name]["revenue"],
        ])
    
    if len(matches) > 0:
        return matches.mean()
    return 0

def build_feature_vector(inputs, df, feature_columns):
    """Build a feature vector from user inputs."""
    
    features = {}
    
    # Budget features
    features["budget"] = inputs["budget"]
    features["log_budget"] = np.log1p(inputs["budget"])
    features["is_low_budget"] = 1 if inputs["budget"] < 15_000_000 else 0
    features["is_mid_budget"] = 1 if 15_000_000 <= inputs["budget"] < 100_000_000 else 0
    features["is_blockbuster_budget"] = 1 if inputs["budget"] >= 100_000_000 else 0
    
    # Movie attributes
    features["runtime"] = inputs["runtime"]
    features["num_genres"] = len(inputs["genres"]) if inputs["genres"] else 1
    features["num_production_companies"] = len(inputs["studios"]) if inputs["studios"] else 0
    features["num_cast"] = 5 if inputs["actors"] else 0
    features["num_keywords"] = 3
    features["is_franchise"] = inputs["is_franchise"]
    
    # Release timing
    release_date = inputs["release_date"]
    features["release_year"] = release_date.year
    features["release_month"] = release_date.month
    features["release_dayofweek"] = release_date.weekday()
    features["release_quarter"] = (release_date.month - 1) // 3 + 1
    features["is_weekend_release"] = 1 if release_date.weekday() >= 4 else 0
    features["is_summer_release"] = 1 if release_date.month in [5, 6, 7, 8] else 0
    features["is_holiday_release"] = 1 if release_date.month in [11, 12] else 0
    features["days_since_2000"] = (release_date - date(2000, 1, 1)).days
    
    # Director features
    director = inputs["director"] if inputs["director"] not in ["(Select)", "(Unknown Director)", ""] else None
    features["director_popularity"] = get_person_popularity(df, director, "director") if director else 1.0
    director_hist_rev = get_historical_revenue(df, director, "director") if director else 0
    features["director_historical_avg_log_revenue"] = np.log1p(director_hist_rev)
    director_films = len(df[df["director_name"] == director]) if director else 0
    features["director_prior_films"] = director_films
    features["is_director_debut"] = 1 if director_films == 0 else 0
    features["has_popular_director"] = 1 if features["director_popularity"] > 20 else 0
    
    # Actor features
    actor_popularities = []
    valid_actors = [a for a in inputs["actors"] if a and a not in ["(Select)", "(Unknown)"]]
    
    for i in range(3):
        if i < len(valid_actors):
            pop = get_person_popularity(df, valid_actors[i], "actor")
            features[f"actor_{i+1}_popularity"] = pop
            actor_popularities.append(pop)
        else:
            features[f"actor_{i+1}_popularity"] = 0
            actor_popularities.append(0)
    
    features["avg_actor_popularity"] = np.mean(actor_popularities) if actor_popularities else 0
    features["max_actor_popularity"] = max(actor_popularities) if actor_popularities else 0
    features["total_star_power"] = features["director_popularity"] + sum(actor_popularities)
    features["has_popular_lead"] = 1 if actor_popularities and actor_popularities[0] > 20 else 0
    
    # Lead actor historical revenue
    if valid_actors:
        lead_hist_rev = get_historical_revenue(df, valid_actors[0], "actor")
        features["lead_actor_historical_avg_log_revenue"] = np.log1p(lead_hist_rev)
        features["lead_actor_prior_films"] = len(df[df["actor_1_name"] == valid_actors[0]])
    else:
        features["lead_actor_historical_avg_log_revenue"] = 0
        features["lead_actor_prior_films"] = 0
    
    # Studio features
    studios_str = str(inputs["studios"]) if inputs["studios"] else ""
    major_studios = ["Warner Bros", "Universal", "Disney", "Paramount", "Sony", "Marvel", "Pixar", "Lucasfilm", "20th Century"]
    features["has_major_studio"] = 1 if any(s in studios_str for s in major_studios) else 0
    features["is_disney"] = 1 if any(s in studios_str for s in ["Disney", "Pixar", "Marvel", "Lucasfilm"]) else 0
    features["is_warner"] = 1 if "Warner" in studios_str else 0
    features["is_universal"] = 1 if "Universal" in studios_str else 0
    
    # Competition (use averages)
    features["movies_same_month"] = 15
    features["blockbusters_same_month"] = 3
    
    # Title features
    features["title_length"] = len(inputs["title"])
    features["title_word_count"] = len(inputs["title"].split())
    sequel_indicators = ["2", "3", "II", "III", "Returns", "Part", "Chapter"]
    features["title_suggests_sequel"] = 1 if any(s in inputs["title"] for s in sequel_indicators) else 0
    
    # Certification features
    cert = inputs["certification"]
    features["cert_g"] = 1 if cert == "G" else 0
    features["cert_pg"] = 1 if cert == "PG" else 0
    features["cert_pg13"] = 1 if cert == "PG-13" else 0
    features["cert_r"] = 1 if cert == "R" else 0
    features["is_family_friendly_cert"] = 1 if cert in ["G", "PG"] else 0
    features["is_r_rated"] = 1 if cert == "R" else 0
    
    # Marketing features
    features["num_videos"] = inputs.get("num_trailers", 2) + 1
    features["num_trailers"] = inputs.get("num_trailers", 2)
    features["num_teasers"] = 1
    features["has_trailer"] = 1 if inputs.get("num_trailers", 2) > 0 else 0
    features["has_multiple_trailers"] = 1 if inputs.get("num_trailers", 2) > 1 else 0
    features["has_teaser"] = 1
    features["days_trailer_before_release"] = inputs.get("marketing_lead_days", 90)
    features["early_marketing"] = 1 if inputs.get("marketing_lead_days", 90) > 180 else 0
    features["late_marketing"] = 1 if inputs.get("marketing_lead_days", 90) < 30 else 0
    features["social_media_presence"] = inputs.get("social_media", 1)
    features["has_strong_social"] = 1 if inputs.get("social_media", 1) >= 2 else 0
    features["has_tagline"] = 1
    features["tagline_length"] = 40
    features["has_homepage"] = 1 if features["has_major_studio"] else 0
    features["overview_length"] = 150
    
    # International features
    features["num_translations"] = inputs.get("international_reach", 20)
    features["log_translations"] = np.log1p(inputs.get("international_reach", 20))
    features["high_international_reach"] = 1 if inputs.get("international_reach", 20) > 30 else 0
    features["num_release_countries"] = inputs.get("release_countries", 15)
    features["wide_release"] = 1 if inputs.get("release_countries", 15) > 20 else 0
    features["num_production_countries"] = 1
    features["is_us_production"] = 1
    features["num_spoken_languages"] = 1
    features["is_english"] = 1
    
    # Crew features
    features["num_writers"] = 2
    features["num_producers"] = 3
    features["num_exec_producers"] = 2
    features["writer_popularity"] = 3
    features["producer_popularity"] = 3
    features["composer_popularity"] = inputs.get("composer_popularity", 3)
    features["cinematographer_popularity"] = 3
    features["total_crew_popularity"] = 15
    features["has_popular_composer"] = 1 if inputs.get("composer_popularity", 3) > 10 else 0
    features["production_team_size"] = 7
    
    # Genre features
    all_genres = ["action", "adventure", "animation", "comedy", "crime", "documentary",
                  "drama", "family", "fantasy", "history", "horror", "music", 
                  "mystery", "romance", "science_fiction", "thriller", "war", "western"]
    
    selected_genres = inputs["genres"] if inputs["genres"] else []
    for genre in all_genres:
        genre_display = genre.replace("_", " ").title()
        if genre == "science_fiction":
            genre_display = "Science Fiction"
        features[f"genre_{genre}"] = 1 if genre_display in selected_genres else 0
    
    # Keyword features
    features["keyword_superhero"] = 1 if "Action" in selected_genres and inputs["is_franchise"] else 0
    features["keyword_sequel_keyword"] = 1 if inputs["is_franchise"] else 0
    features["keyword_family_friendly"] = 1 if "Family" in selected_genres or "Animation" in selected_genres else 0
    features["keyword_action_heavy"] = 1 if "Action" in selected_genres else 0
    features["keyword_romance"] = 1 if "Romance" in selected_genres else 0
    features["keyword_scifi"] = 1 if "Science Fiction" in selected_genres else 0
    features["keyword_horror_keyword"] = 1 if "Horror" in selected_genres else 0
    
    # Create dataframe with correct column order
    feature_vector = pd.DataFrame([features])
    
    # Ensure all columns exist
    for col in feature_columns:
        if col not in feature_vector.columns:
            feature_vector[col] = 0
    
    feature_vector = feature_vector[feature_columns]
    feature_vector = feature_vector.fillna(0)
    
    return feature_vector


def main():
    # Header with logo/title
    col_title, col_info = st.columns([3, 1])
    with col_title:
        st.markdown("""
        # 🎬 Box Office Predictor
        ### Build your dream movie and predict its box office potential
        """)
    
    with col_info:
        st.markdown("""
        <div class="info-box">
        <strong>Model Accuracy</strong><br>
        R² = 0.71 (71% variance explained)
        </div>
        """, unsafe_allow_html=True)
    
    # Check if model exists
    if not os.path.exists("models/best_model.joblib"):
        st.error("⚠️ Model not found! Run `python src/train_model.py` first.")
        st.stop()
    
    # Load resources
    try:
        model = load_model()
        df, engineered = load_reference_data()
        directors, actors, companies = get_unique_values(df)
        feature_columns = pd.read_csv("data/processed/X_features.csv").columns.tolist()
        
        # Data stats for context
        min_revenue = df['revenue'].min()
        max_revenue = df['revenue'].max()
        median_revenue = df['revenue'].median()
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
    
    # Sidebar inputs
    with st.sidebar:
        st.markdown("## 🎥 Build Your Movie")
        st.markdown("---")
        
        # Basic Info
        st.markdown("### 📋 Basic Info")
        title = st.text_input("Movie Title", value="Untitled Project", 
                             help="What's your movie called?")
        
        budget = st.slider(
            "💰 Production Budget",
            min_value=1_000_000,
            max_value=400_000_000,
            value=50_000_000,
            step=5_000_000,
            format="$%d",
            help="How much are you spending to make this?"
        )
        
        runtime = st.slider("⏱️ Runtime (minutes)", 80, 200, 110)
        
        st.markdown("---")
        
        # Genre
        st.markdown("### 🎭 Genre & Rating")
        genre_options = ["Action", "Adventure", "Animation", "Comedy", "Crime", 
                         "Documentary", "Drama", "Family", "Fantasy", "History",
                         "Horror", "Music", "Mystery", "Romance", "Science Fiction",
                         "Thriller", "War", "Western"]
        genres = st.multiselect("Genres", genre_options, default=["Drama"])
        
        certification = st.selectbox("MPAA Rating", ["G", "PG", "PG-13", "R"], index=2)
        
        is_franchise = st.checkbox("🔄 Part of a Franchise/Sequel", value=False)
        
        st.markdown("---")
        
        # Release
        st.markdown("### 📅 Release")
        release_date = st.date_input(
            "Release Date",
            value=date(2026, 6, 15),
            min_value=date(2025, 1, 1),
            max_value=date(2030, 12, 31)
        )
        
        st.markdown("---")
        
        # Talent
        st.markdown("### ⭐ Talent")
        director = st.selectbox(
            "Director",
            ["(Unknown Director)"] + directors,
            index=0,
            help="Directors with proven track records boost predictions"
        )
        
        actor1 = st.selectbox("Lead Actor/Actress", ["(Unknown)"] + actors, index=0)
        actor2 = st.selectbox("Supporting #1", ["(Unknown)"] + actors, index=0)
        actor3 = st.selectbox("Supporting #2", ["(Unknown)"] + actors, index=0)
        
        selected_actors = [actor1, actor2, actor3]
        
        st.markdown("---")
        
        # Studio
        st.markdown("### 🏢 Studio")
        studios = st.multiselect(
            "Production Studios",
            companies,
            default=[],
            help="Major studios = wider distribution"
        )
        
        # Advanced options
        st.markdown("---")
        with st.expander("🔧 Advanced Options"):
            num_trailers = st.slider("Trailers Released", 0, 5, 2)
            marketing_lead_days = st.slider("Marketing Lead (days)", 0, 365, 90)
            social_media = st.slider("Social Media Presence (0-3)", 0, 3, 1)
            international_reach = st.slider("International Markets", 0, 50, 20)
            release_countries = st.slider("Release Countries", 1, 50, 15)
    
    # Build inputs
    inputs = {
        "title": title,
        "budget": budget,
        "runtime": runtime,
        "genres": genres,
        "release_date": release_date,
        "certification": certification,
        "is_franchise": 1 if is_franchise else 0,
        "director": director,
        "actors": selected_actors,
        "studios": studios,
        "num_trailers": num_trailers,
        "marketing_lead_days": marketing_lead_days,
        "social_media": social_media,
        "international_reach": international_reach,
        "release_countries": release_countries,
        "composer_popularity": 3,
    }
    
    # Main content - 3 columns
    col1, col2, col3 = st.columns([1.5, 1, 1])
    
    with col1:
        st.markdown(f"""
        <div class="movie-card">
        <h2 style="color: #1a73e8; margin-top:0;">📽️ {title}</h2>
        <table style="width:100%; color: #202124;">
            <tr><td style="color: #5f6368;">Budget</td><td style="text-align:right;"><strong>${budget:,.0f}</strong></td></tr>
            <tr><td style="color: #5f6368;">Genres</td><td style="text-align:right;">{", ".join(genres) if genres else "Not set"}</td></tr>
            <tr><td style="color: #5f6368;">Rating</td><td style="text-align:right;">{certification}</td></tr>
            <tr><td style="color: #5f6368;">Runtime</td><td style="text-align:right;">{runtime} min</td></tr>
            <tr><td style="color: #5f6368;">Release</td><td style="text-align:right;">{release_date.strftime('%b %d, %Y')}</td></tr>
            <tr><td style="color: #5f6368;">Director</td><td style="text-align:right;">{director if director != "(Unknown Director)" else "TBD"}</td></tr>
            <tr><td style="color: #5f6368;">Lead Cast</td><td style="text-align:right;">{actor1 if actor1 != "(Unknown)" else "TBD"}</td></tr>
            <tr><td style="color: #5f6368;">Studio</td><td style="text-align:right;">{studios[0] if studios else "Independent"}</td></tr>
            <tr><td style="color: #5f6368;">Franchise</td><td style="text-align:right;">{"✅ Yes" if is_franchise else "❌ No"}</td></tr>
        </table>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        try:
            feature_vector = build_feature_vector(inputs, df, feature_columns)
            log_prediction = model.predict(feature_vector)[0]
            prediction = np.expm1(log_prediction)
            
            # Clamp to reasonable range
            prediction = max(prediction, 1_000_000)  # At least $1M
            
            # Format prediction
            if prediction >= 1_000_000_000:
                pred_str = f"${prediction/1_000_000_000:.2f}B"
            else:
                pred_str = f"${prediction/1_000_000:.0f}M"
            
            st.markdown("### 💰 Predicted Box Office")
            st.metric(label="Worldwide Gross", value=pred_str)
            
            # ROI
            roi = ((prediction - budget) / budget) * 100
            
            if roi >= 100:
                roi_emoji = "🚀"
                roi_text = "Blockbuster"
            elif roi >= 0:
                roi_emoji = "✅"
                roi_text = "Profitable"
            else:
                roi_emoji = "⚠️"
                roi_text = "Loss"
            
            st.metric(
                label="Estimated ROI",
                value=f"{roi:.0f}%",
                delta=f"{roi_emoji} {roi_text}"
            )
            
        except Exception as e:
            st.error(f"Error: {e}")
            prediction = 0
            roi = 0
    
    with col3:
        # Percentile context
        if prediction > 0:
            percentile = (df['revenue'] < prediction).mean() * 100
            
            st.markdown("### 📊 Context")
            
            st.markdown(f"""
            <div class="prediction-context">
            <p style="color: #5f6368; margin:0;">Compared to training data:</p>
            <p style="font-size: 1.5rem; color: #1a73e8; margin:5px 0;"><strong>Top {100-percentile:.0f}%</strong></p>
            <p style="color: #5f6368; font-size: 0.8rem; margin:0;">of {len(df)} movies in database</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Visual comparison
            st.markdown(f"""
            <div style="margin-top:15px;">
            <p style="color:#5f6368; font-size:0.8rem;">Database range:</p>
            <p style="color:#202124; font-size:0.75rem;">Min: ${min_revenue/1e6:.0f}M | Med: ${median_revenue/1e6:.0f}M | Max: ${max_revenue/1e9:.1f}B</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Dataset info
    st.markdown("""
    <div class="info-box">
    <strong>Dataset:</strong> Trained on <strong>1,919 movies</strong> with diverse outcomes — 
    from $7 flops to $2.9B blockbusters. Includes 350 movies that lost money (18%), 
    so predictions reflect real-world risk. Revenue range covers indie releases to mega-hits.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Analysis tabs
    tab1, tab2, tab3 = st.tabs(["💡 Key Factors", "🎬 Similar Movies", "📈 Recommendations"])
    
    with tab1:
        col_pos, col_neg = st.columns(2)
        
        positive_factors = []
        negative_factors = []
        
        # Analyze factors
        if budget >= 100_000_000:
            positive_factors.append("Blockbuster-level budget")
        elif budget < 20_000_000:
            negative_factors.append("Limited budget for marketing")
        
        if is_franchise:
            positive_factors.append("Franchise = built-in audience")
        
        if certification == "R":
            negative_factors.append("R-rating limits audience")
        elif certification in ["G", "PG"]:
            positive_factors.append("Family-friendly rating")
        
        if release_date.month in [5, 6, 7, 8]:
            positive_factors.append("Summer blockbuster season")
        elif release_date.month in [11, 12]:
            positive_factors.append("Holiday/awards season")
        elif release_date.month in [1, 2, 9]:
            negative_factors.append("Weak release window")
        
        if any(s in str(studios) for s in ["Disney", "Marvel", "Pixar", "Warner", "Universal"]):
            positive_factors.append("Major studio distribution")
        elif not studios:
            negative_factors.append("No major studio backing")
        
        if director and director != "(Unknown Director)":
            dir_films = len(df[df["director_name"] == director])
            if dir_films > 3:
                positive_factors.append(f"Proven director ({dir_films} hits)")
        else:
            negative_factors.append("Unknown director")
        
        if actor1 and actor1 != "(Unknown)":
            positive_factors.append("Star casting")
        else:
            negative_factors.append("No star power")
        
        if "Animation" in genres:
            positive_factors.append("Animation = global appeal")
        
        with col_pos:
            st.markdown("#### ✅ Working For You")
            if positive_factors:
                for f in positive_factors:
                    st.markdown(f"- {f}")
            else:
                st.markdown("*Add more elements to strengthen your project*")
        
        with col_neg:
            st.markdown("#### ⚠️ Potential Concerns")
            if negative_factors:
                for f in negative_factors:
                    st.markdown(f"- {f}")
            else:
                st.markdown("*Looking good!*")
    
    with tab2:
        # Find similar movies
        similar = df.copy()
        budget_low, budget_high = budget * 0.5, budget * 1.5
        similar = similar[(similar["budget"] >= budget_low) & (similar["budget"] <= budget_high)]
        
        if genres:
            genre_filter = similar["genres"].apply(lambda x: any(g in str(x) for g in genres))
            if genre_filter.sum() > 3:
                similar = similar[genre_filter]
        
        if len(similar) > 0:
            similar = similar.nlargest(8, "revenue")[["title", "budget", "revenue", "release_date", "director_name"]]
            
            for _, row in similar.iterrows():
                st.markdown(f"""
                **{row['title']}** ({row['release_date'][:4] if pd.notna(row['release_date']) else 'N/A'})  
                Budget: ${row['budget']/1e6:.0f}M → Box Office: ${row['revenue']/1e6:.0f}M  
                Director: {row['director_name'] if pd.notna(row['director_name']) else 'Unknown'}
                """)
                st.markdown("---")
        else:
            st.info("No similar movies found with these criteria")
    
    with tab3:
        st.markdown("### How to Boost Box Office Potential")
        
        recs = []
        
        if not is_franchise:
            recs.append("🎬 **Make it a franchise** — Sequels average 40% higher revenue")
        
        if certification == "R" and "Horror" not in genres:
            recs.append("🎭 **Consider PG-13** — Wider audience = more tickets")
        
        if release_date.month in [1, 2, 9]:
            recs.append("📅 **Move to summer or holiday** — Peak moviegoing seasons")
        
        if not studios or not any(s in str(studios) for s in ["Disney", "Marvel", "Warner", "Universal", "Paramount"]):
            recs.append("🏢 **Partner with major studio** — Critical for wide release")
        
        if director == "(Unknown Director)":
            recs.append("🎥 **Attach proven director** — Track record drives confidence")
        
        if actor1 == "(Unknown)":
            recs.append("⭐ **Cast star talent** — Name recognition opens strong")
        
        if budget < 30_000_000 and "Action" in genres:
            recs.append("💰 **Increase budget** — Action needs VFX investment")
        
        if recs:
            for rec in recs:
                st.markdown(rec)
        else:
            st.success("✨ Your project has strong commercial elements!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #5f6368; font-size: 0.8rem;">
    Built with XGBoost & Streamlit | Trained on 1,919 movies from TMDB | Model R² = 0.71<br>
    <em>For educational purposes — not financial advice</em>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
