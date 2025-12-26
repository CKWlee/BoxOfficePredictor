"""
collect_data.py
Fetches movie data from TMDB API for box office prediction.
"""

import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

API_KEY = os.getenv("TMDB_API_KEY")
BASE_URL = "https://api.themoviedb.org/3"

# Rate limiting: TMDB allows ~40 requests per 10 seconds
REQUEST_DELAY = 0.25


def get_popular_movies(num_pages: int = 20) -> list[dict]:
    """
    Fetch movies from TMDB using multiple strategies to get a balanced dataset
    that includes both hits AND flops.
    """
    movies = []
    seen_ids = set()
    
    # Strategy 1: Get movies by popularity (includes both hits and flops)
    print("Strategy 1: Fetching by popularity...")
    for page in tqdm(range(1, num_pages + 1), desc="By popularity"):
        url = f"{BASE_URL}/discover/movie"
        params = {
            "api_key": API_KEY,
            "language": "en-US",
            "sort_by": "popularity.desc",
            "include_adult": "false",
            "include_video": "false",
            "page": page,
            "vote_count.gte": 50,
            "with_original_language": "en",
        }
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            for movie in response.json().get("results", []):
                if movie["id"] not in seen_ids:
                    movies.append(movie)
                    seen_ids.add(movie["id"])
        time.sleep(REQUEST_DELAY)
    
    # Strategy 2: Get movies by year (2010-2024) to ensure temporal diversity
    print("\nStrategy 2: Fetching by year...")
    for year in tqdm(range(2010, 2025), desc="By year"):
        for page in range(1, 8):  # 7 pages per year
            url = f"{BASE_URL}/discover/movie"
            params = {
                "api_key": API_KEY,
                "language": "en-US",
                "sort_by": "vote_count.desc",  # Most voted, not highest revenue
                "include_adult": "false",
                "primary_release_year": year,
                "page": page,
                "vote_count.gte": 30,
                "with_original_language": "en",
            }
            
            response = requests.get(url, params=params)
            if response.status_code == 200:
                for movie in response.json().get("results", []):
                    if movie["id"] not in seen_ids:
                        movies.append(movie)
                        seen_ids.add(movie["id"])
            time.sleep(REQUEST_DELAY)
    
    # Strategy 3: Get some low-revenue movies explicitly (flops)
    print("\nStrategy 3: Fetching potential flops (low revenue)...")
    for page in tqdm(range(1, 15), desc="Low revenue"):
        url = f"{BASE_URL}/discover/movie"
        params = {
            "api_key": API_KEY,
            "language": "en-US",
            "sort_by": "revenue.asc",  # Ascending = low revenue first
            "include_adult": "false",
            "page": page,
            "vote_count.gte": 100,  # Still need enough votes for reliability
            "with_original_language": "en",
            "primary_release_date.gte": "2005-01-01",
            "with_runtime.gte": 70,  # Exclude shorts
        }
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            for movie in response.json().get("results", []):
                if movie["id"] not in seen_ids:
                    movies.append(movie)
                    seen_ids.add(movie["id"])
        time.sleep(REQUEST_DELAY)
    
    # Strategy 4: Also get some high-revenue movies to ensure blockbusters are included
    print("\nStrategy 4: Fetching blockbusters...")
    for page in tqdm(range(1, 20), desc="High revenue"):
        url = f"{BASE_URL}/discover/movie"
        params = {
            "api_key": API_KEY,
            "language": "en-US",
            "sort_by": "revenue.desc",
            "include_adult": "false",
            "page": page,
            "vote_count.gte": 100,
            "with_original_language": "en",
        }
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            for movie in response.json().get("results", []):
                if movie["id"] not in seen_ids:
                    movies.append(movie)
                    seen_ids.add(movie["id"])
        time.sleep(REQUEST_DELAY)
    
    print(f"\nTotal unique movies collected: {len(movies)}")
    return movies


def get_movie_details(movie_id: int, max_retries: int = 3) -> dict | None:
    """
    Fetch detailed movie information including budget, revenue, and credits.
    Includes retry logic for timeout errors.
    """
    url = f"{BASE_URL}/movie/{movie_id}"
    params = {
        "api_key": API_KEY,
        "language": "en-US",
        # Fetch all useful additional data in one request
        "append_to_response": "credits,keywords,release_dates,videos,external_ids,translations",
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            return None
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            return None
        except requests.exceptions.RequestException:
            return None


def extract_movie_features(movie_detail: dict) -> dict:
    """
    Extract relevant features from movie detail response.
    """
    # Basic info
    features = {
        "id": movie_detail.get("id"),
        "title": movie_detail.get("title"),
        "budget": movie_detail.get("budget", 0),
        "revenue": movie_detail.get("revenue", 0),
        "runtime": movie_detail.get("runtime"),
        "release_date": movie_detail.get("release_date"),
        "vote_average": movie_detail.get("vote_average"),
        "vote_count": movie_detail.get("vote_count"),
        "popularity": movie_detail.get("popularity"),
        "original_language": movie_detail.get("original_language"),
    }
    
    # Genres (as comma-separated string)
    genres = movie_detail.get("genres", [])
    features["genres"] = ",".join([g["name"] for g in genres])
    features["num_genres"] = len(genres)
    
    # Production companies
    companies = movie_detail.get("production_companies", [])
    features["num_production_companies"] = len(companies)
    features["production_companies"] = ",".join([c["name"] for c in companies[:3]])
    
    # Check if part of a collection (franchise)
    features["is_franchise"] = 1 if movie_detail.get("belongs_to_collection") else 0
    
    # Credits - Director
    credits = movie_detail.get("credits", {})
    crew = credits.get("crew", [])
    directors = [c for c in crew if c.get("job") == "Director"]
    if directors:
        features["director_name"] = directors[0].get("name")
        features["director_popularity"] = directors[0].get("popularity", 0)
    else:
        features["director_name"] = None
        features["director_popularity"] = 0
    
    # Credits - Top 3 cast members
    cast = credits.get("cast", [])[:3]
    features["num_cast"] = len(credits.get("cast", []))
    
    for i, actor in enumerate(cast):
        features[f"actor_{i+1}_name"] = actor.get("name")
        features[f"actor_{i+1}_popularity"] = actor.get("popularity", 0)
    
    # Fill missing actor slots
    for i in range(len(cast), 3):
        features[f"actor_{i+1}_name"] = None
        features[f"actor_{i+1}_popularity"] = 0
    
    # Keywords
    keywords = movie_detail.get("keywords", {}).get("keywords", [])
    features["keywords"] = ",".join([k["name"] for k in keywords[:10]])
    features["num_keywords"] = len(keywords)
    
    # === NEW FEATURES ===
    
    # Release dates - Extract US certification (MPAA rating)
    release_dates = movie_detail.get("release_dates", {}).get("results", [])
    us_release = next((r for r in release_dates if r.get("iso_3166_1") == "US"), None)
    if us_release:
        us_releases = us_release.get("release_dates", [])
        # Get theatrical release certification
        theatrical = next((r for r in us_releases if r.get("type") in [2, 3]), None)  # 2=Theatrical limited, 3=Theatrical
        if theatrical:
            features["certification"] = theatrical.get("certification", "")
        else:
            # Fall back to any certification
            any_cert = next((r for r in us_releases if r.get("certification")), None)
            features["certification"] = any_cert.get("certification", "") if any_cert else ""
    else:
        features["certification"] = ""
    
    # Count how many countries it's releasing in
    features["num_release_countries"] = len(release_dates)
    
    # Videos - trailers, teasers, etc.
    videos = movie_detail.get("videos", {}).get("results", [])
    features["num_videos"] = len(videos)
    features["num_trailers"] = len([v for v in videos if v.get("type") == "Trailer"])
    features["num_teasers"] = len([v for v in videos if v.get("type") == "Teaser"])
    
    # Calculate days between first trailer and release (marketing lead time)
    trailer_dates = []
    for v in videos:
        if v.get("type") in ["Trailer", "Teaser"] and v.get("published_at"):
            try:
                trailer_date = pd.to_datetime(v["published_at"]).date()
                trailer_dates.append(trailer_date)
            except:
                pass
    
    if trailer_dates and movie_detail.get("release_date"):
        try:
            release = pd.to_datetime(movie_detail["release_date"]).date()
            first_trailer = min(trailer_dates)
            features["days_trailer_before_release"] = (release - first_trailer).days
        except:
            features["days_trailer_before_release"] = None
    else:
        features["days_trailer_before_release"] = None
    
    # External IDs - for potential linking
    external_ids = movie_detail.get("external_ids", {})
    features["imdb_id"] = external_ids.get("imdb_id")
    features["has_imdb"] = 1 if external_ids.get("imdb_id") else 0
    features["has_facebook"] = 1 if external_ids.get("facebook_id") else 0
    features["has_instagram"] = 1 if external_ids.get("instagram_id") else 0
    features["has_twitter"] = 1 if external_ids.get("twitter_id") else 0
    features["social_media_presence"] = features["has_facebook"] + features["has_instagram"] + features["has_twitter"]
    
    # Translations - international reach
    translations = movie_detail.get("translations", {}).get("translations", [])
    features["num_translations"] = len(translations)
    
    # Production countries
    prod_countries = movie_detail.get("production_countries", [])
    features["num_production_countries"] = len(prod_countries)
    features["is_us_production"] = 1 if any(c.get("iso_3166_1") == "US" for c in prod_countries) else 0
    
    # Spoken languages
    spoken_languages = movie_detail.get("spoken_languages", [])
    features["num_spoken_languages"] = len(spoken_languages)
    features["is_english"] = 1 if any(l.get("iso_639_1") == "en" for l in spoken_languages) else 0
    
    # Additional crew - writers, producers, composers
    if crew:
        writers = [c for c in crew if c.get("department") == "Writing"]
        producers = [c for c in crew if c.get("job") == "Producer"]
        exec_producers = [c for c in crew if c.get("job") == "Executive Producer"]
        composers = [c for c in crew if c.get("job") in ["Original Music Composer", "Music"]]
        cinematographers = [c for c in crew if c.get("job") == "Director of Photography"]
        
        features["num_writers"] = len(writers)
        features["num_producers"] = len(producers)
        features["num_exec_producers"] = len(exec_producers)
        
        # Top writer popularity
        if writers:
            features["writer_popularity"] = max(w.get("popularity", 0) for w in writers)
        else:
            features["writer_popularity"] = 0
        
        # Top producer popularity
        if producers:
            features["producer_popularity"] = max(p.get("popularity", 0) for p in producers)
        else:
            features["producer_popularity"] = 0
        
        # Composer popularity (Hans Zimmer effect)
        if composers:
            features["composer_name"] = composers[0].get("name")
            features["composer_popularity"] = composers[0].get("popularity", 0)
        else:
            features["composer_name"] = None
            features["composer_popularity"] = 0
        
        # Cinematographer
        if cinematographers:
            features["cinematographer_popularity"] = cinematographers[0].get("popularity", 0)
        else:
            features["cinematographer_popularity"] = 0
    else:
        features["num_writers"] = 0
        features["num_producers"] = 0
        features["num_exec_producers"] = 0
        features["writer_popularity"] = 0
        features["producer_popularity"] = 0
        features["composer_name"] = None
        features["composer_popularity"] = 0
        features["cinematographer_popularity"] = 0
    
    # Tagline presence (movies with taglines often have bigger marketing)
    features["has_tagline"] = 1 if movie_detail.get("tagline") else 0
    features["tagline_length"] = len(movie_detail.get("tagline", ""))
    
    # Homepage (official website = bigger marketing)
    features["has_homepage"] = 1 if movie_detail.get("homepage") else 0
    
    # Overview/description length (more detailed = more marketing effort)
    features["overview_length"] = len(movie_detail.get("overview", ""))
    
    return features


def main():
    """Main data collection pipeline."""
    
    # Create data directories
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    print("Step 1: Fetching diverse movie list from TMDB...")
    print("(Using multiple strategies to include both hits AND flops)")
    movies = get_popular_movies(num_pages=20)  # num_pages only affects strategy 1
    print(f"Found {len(movies)} unique movies")
    
    print("\nStep 2: Fetching detailed info for each movie...")
    movie_features = []
    
    for movie in tqdm(movies, desc="Fetching details"):
        details = get_movie_details(movie["id"])
        
        if details:
            features = extract_movie_features(details)
            
            # Only keep movies with valid budget AND revenue
            # But now we include ALL revenue ranges (including flops!)
            if features["budget"] > 0 and features["revenue"] > 0:
                movie_features.append(features)
        
        time.sleep(REQUEST_DELAY)
    
    print(f"\nCollected {len(movie_features)} movies with complete data")
    
    # Save to CSV
    df = pd.DataFrame(movie_features)
    output_path = "data/raw/movies_raw.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved raw data to {output_path}")
    
    # Print summary statistics
    print("\n--- Dataset Summary ---")
    print(f"Total movies: {len(df)}")
    if len(df) > 0:
        print(f"Date range: {df['release_date'].min()} to {df['release_date'].max()}")
        print(f"Budget range: ${df['budget'].min():,.0f} - ${df['budget'].max():,.0f}")
        print(f"Revenue range: ${df['revenue'].min():,.0f} - ${df['revenue'].max():,.0f}")
        
        # Show revenue distribution
        print("\n--- Revenue Distribution ---")
        print(f"Movies under $10M: {len(df[df['revenue'] < 10_000_000])}")
        print(f"Movies $10M-$50M: {len(df[(df['revenue'] >= 10_000_000) & (df['revenue'] < 50_000_000)])}")
        print(f"Movies $50M-$100M: {len(df[(df['revenue'] >= 50_000_000) & (df['revenue'] < 100_000_000)])}")
        print(f"Movies $100M-$500M: {len(df[(df['revenue'] >= 100_000_000) & (df['revenue'] < 500_000_000)])}")
        print(f"Movies $500M+: {len(df[df['revenue'] >= 500_000_000])}")
        
        # Show ROI distribution (to see flops)
        df['roi'] = (df['revenue'] - df['budget']) / df['budget'] * 100
        print(f"\n--- ROI Distribution ---")
        print(f"Flops (ROI < 0%): {len(df[df['roi'] < 0])}")
        print(f"Break-even (0-100%): {len(df[(df['roi'] >= 0) & (df['roi'] < 100)])}")
        print(f"Profitable (100%+): {len(df[df['roi'] >= 100])}")
    else:
        print("No movies collected. Check your API key and network connection.")


if __name__ == "__main__":
    main()
