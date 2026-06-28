"""
download_movielens.py
----------------------
Downloads the REAL MovieLens 100k dataset from GroupLens and writes
movies.csv (movie_id, movie_title, movie_genre) and ratings.csv
(user_id, movie_id, rating) into data/.

NOTE: needs normal internet access to files.grouplens.org. Run this on
your own machine, not inside a network-restricted sandbox.

Run:
    python scripts/download_movielens.py
"""
import os
import io
import zipfile
import urllib.request
import pandas as pd

URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

GENRE_COLS = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def first_genre(row):
    for g in GENRE_COLS[1:]:
        if row[g] == 1:
            return g
    return "unknown"


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Downloading {URL} ...")
    with urllib.request.urlopen(URL, timeout=60) as resp:
        data = resp.read()
    zf = zipfile.ZipFile(io.BytesIO(data))

    with zf.open("ml-100k/u.item") as f:
        movies = pd.read_csv(
            f, sep="|", encoding="latin-1",
            names=["movie_id", "movie_title", "release_date", "video_release_date", "imdb_url"] + GENRE_COLS,
        )
    movies["movie_genre"] = movies.apply(first_genre, axis=1)
    movies = movies[["movie_id", "movie_title", "movie_genre"]]

    with zf.open("ml-100k/u.data") as f:
        ratings = pd.read_csv(f, sep="\t", names=["user_id", "movie_id", "rating", "timestamp"])
    ratings = ratings[["user_id", "movie_id", "rating"]]

    movies.to_csv(os.path.join(DATA_DIR, "movies.csv"), index=False)
    ratings.to_csv(os.path.join(DATA_DIR, "ratings.csv"), index=False)
    print(f"Wrote {len(movies)} real movies and {len(ratings)} real ratings to {DATA_DIR}/")


if __name__ == "__main__":
    main()
