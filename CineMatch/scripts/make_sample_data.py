"""
make_sample_data.py
--------------------
Generates a small synthetic MovieLens-like dataset (movies.csv, ratings.csv)
with a genre column, so the project runs end-to-end without internet
access. Use download_movielens.py instead to get the REAL dataset.

Run:
    python scripts/make_sample_data.py
"""
import random
import csv
import os

random.seed(42)
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUT_DIR, exist_ok=True)

GENRES = ["Action", "Comedy", "Drama", "Sci-Fi", "Romance", "Horror", "Animation"]
N_MOVIES = 50
N_USERS = 150
N_RATINGS = 2500


def main():
    movies = []
    for movie_id in range(1, N_MOVIES + 1):
        movies.append({
            "movie_id": movie_id,
            "movie_title": f"Movie {movie_id}",
            "movie_genre": random.choice(GENRES),
        })

    ratings = []
    for _ in range(N_RATINGS):
        user_id = random.randint(1, N_USERS)
        movie_id = random.randint(1, N_MOVIES)
        rating = min(5, max(1, round(random.gauss(3.5, 1.1))))
        ratings.append({"user_id": user_id, "movie_id": movie_id, "rating": rating})

    with open(os.path.join(OUT_DIR, "movies.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["movie_id", "movie_title", "movie_genre"])
        w.writeheader()
        w.writerows(movies)

    with open(os.path.join(OUT_DIR, "ratings.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["user_id", "movie_id", "rating"])
        w.writeheader()
        w.writerows(ratings)

    print(f"Wrote {len(movies)} movies and {len(ratings)} ratings to {OUT_DIR}/")


if __name__ == "__main__":
    main()
