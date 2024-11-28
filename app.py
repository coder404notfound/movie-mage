# Import necessary libraries
from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
app = Flask(__name__)

# Fetch API key and base URL from environment variables
TMDB_API_KEY = os.getenv('TMDB_API_KEY')
TMDB_BASE_URL = os.getenv('TMDB_BASE_URL')

# Load the movie dataset (make sure 'tmdb_processed.csv' exists in the project directory)
df = pd.read_csv('tmdb_processed.csv')

# Vectorization of the cast, genres, and tags columns (turning text into feature vectors)
vectorizer = CountVectorizer(stop_words='english')
cast_matrix = vectorizer.fit_transform(df['cast'].fillna(''))  # Handle missing values by replacing with empty string
genres_matrix = vectorizer.fit_transform(df['genres'].fillna(''))
tags_matrix = vectorizer.fit_transform(df['tags'].fillna(''))

# Calculate cosine similarity between movies based on cast, genres, and tags
cast_similarity = cosine_similarity(cast_matrix)
genres_similarity = cosine_similarity(genres_matrix)
tags_similarity = cosine_similarity(tags_matrix)

# Function to fetch movie details from TMDB API using the movie ID
def get_movie_details(movie_id):
    url = f"{TMDB_BASE_URL}{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
    try:
        response = requests.get(url)  # Make a GET request to TMDB API
        response.raise_for_status()  # Raise an exception for HTTP errors
        movie_data = response.json()

        # Check if the movie is flagged as adult and return None if true
        if movie_data.get('adult', False):
            return None  # Return None for adult movies

        return movie_data
    except requests.RequestException as e:
        print(f"TMDB API Error: {e}")  # Log any request errors
        return {}

# Function to recommend movies based on a given title and number of recommendations (n)
def recommend_movies(title, n=5):  # Default value of n is 5
    # Get the index of the movie in the dataframe
    movie_idx = df[df['title'].str.lower() == title.lower()].index
    if not len(movie_idx):
        return f"Movie '{title}' not found in the dataset!"  # If movie not found
    movie_idx = movie_idx[0]

    # Prepare the recommendations dictionary based on different similarity measures
    recommendations = {
        f'Similar Cast (Top-{n})': cast_similarity[movie_idx],
        f'Similar Cast (Random-{n})': cast_similarity[movie_idx],
        f'Similar Genres (Top-{n})': genres_similarity[movie_idx],
        f'Similar Genres (Random-{n})': genres_similarity[movie_idx],
        f'Similar Tags (Top-{n})': tags_similarity[movie_idx],
        f'Similar Tags (Random-{n})': tags_similarity[movie_idx]
    }

    result = {}
    for basis, similarity_scores in recommendations.items():
        if f"Top-{n}" in basis:
            # Get top-n similar movies based on similarity score
            sim_indices = sorted(enumerate(similarity_scores), key=lambda x: x[1], reverse=True)[1:n + 1]
            sim_movies = df.iloc[[idx for idx, _ in sim_indices]].sort_values(by='popularity', ascending=False)
        else:  # For "Random-n" - Randomly sample n movies from top 30 most similar movies
            sim_indices = sorted(enumerate(similarity_scores), key=lambda x: x[1], reverse=True)[:30]
            sim_movies = df.iloc[[idx for idx, _ in sim_indices]]  # Get the top 30 most similar movies
            sim_movies = sim_movies.sample(n=n)  # Randomly sample n movies

        result[basis] = sim_movies[['id', 'title']].to_dict(orient='records')

    return result

# Define the main route for the web app
@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = None
    selected_movie = None  # For storing selected movie details
    movie_titles = df['title'].tolist()  # Dropdown options for movie titles
    n = 5  # Default number of recommendations

    if request.method == 'POST':
        # Get the movie title and number of recommendations (n) from the form
        movie_title = request.form.get('movie_title')
        n = int(request.form.get('n', 5))  # Get n from the form, default to 5 if not provided
        recommendations = recommend_movies(movie_title, n)  # Get movie recommendations

        # Fetch selected movie details from TMDB API
        selected_movie = None
        selected_movie_id = df[df['title'].str.lower() == movie_title.lower()]['id'].values

        if selected_movie_id:
            details = get_movie_details(selected_movie_id[0])
            if details:
                selected_movie = {
                    'title': details.get('title'),
                    'thumbnail': f"https://image.tmdb.org/t/p/w500{details.get('poster_path')}" if details.get('poster_path') else None
                }
            else:
                selected_movie = {"title": "Movie Not Found", "thumbnail": None}

    # Fetch movie details for recommendations
    movie_details = {}
    if recommendations and isinstance(recommendations, dict):
        for basis, movies in recommendations.items():
            movie_details[basis] = []
            for movie in movies:
                if len(movie_details[basis]) >= n:  # Ensure only n recommendations are added
                    break
                details = get_movie_details(movie['id'])
                if details:
                    movie_details[basis].append({
                        'title': details.get('title'),
                        'thumbnail': f"https://image.tmdb.org/t/p/w500{details.get('poster_path')}" if details.get('poster_path') else None,
                        'basis': basis  # Basis for recommendation (e.g., cast, genres, etc.)
                    })

    # Render the HTML page with the recommendations and selected movie details
    return render_template(
        'index.html',
        movie_titles=sorted(movie_titles),  # Sort movie titles alphabetically
        selected_movie=selected_movie,  # Pass the selected movie details
        movie_details=movie_details  # Pass the movie recommendations details
    )

# Run the app in debug mode if this is the main script
if __name__ == '__main__':
    app.run(debug=True)
