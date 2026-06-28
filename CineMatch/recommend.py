import os
import numpy as np
import pandas as pd
import tensorflow as tf

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
EMBED_DIM = 16


def load_data():
    movies_path = os.path.join(DATA_DIR, "movies.csv")
    ratings_path = os.path.join(DATA_DIR, "ratings.csv")
    if not (os.path.exists(movies_path) and os.path.exists(ratings_path)):
        print("No data found — generating sample data first...")
        os.system(f"python {os.path.join(os.path.dirname(__file__), 'scripts', 'make_sample_data.py')}")
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    return movies, ratings


def build_model(num_users, num_movies, num_genres):
    user_input = tf.keras.Input(shape=(), name="user_id", dtype=tf.int32)
    movie_input = tf.keras.Input(shape=(), name="movie_id", dtype=tf.int32)
    genre_input = tf.keras.Input(shape=(), name="genre_id", dtype=tf.int32)

    user_vec = tf.keras.layers.Embedding(num_users, EMBED_DIM)(user_input)
    movie_vec = tf.keras.layers.Embedding(num_movies, EMBED_DIM)(movie_input)
    genre_vec = tf.keras.layers.Embedding(num_genres, 4)(genre_input)

    x = tf.keras.layers.Concatenate()([user_vec, movie_vec, genre_vec])
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    output = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs=[user_input, movie_input, genre_input], outputs=output)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def main():
    movies, ratings = load_data()

    # Map raw IDs -> dense 0..N-1 indices the Embedding layers need
    user_ids = sorted(ratings["user_id"].unique())
    movie_ids = sorted(movies["movie_id"].unique())
    genres = sorted(movies["movie_genre"].unique())

    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    movie_to_idx = {m: i for i, m in enumerate(movie_ids)}
    genre_to_idx = {g: i for i, g in enumerate(genres)}

    movie_genre_lookup = dict(zip(movies["movie_id"], movies["movie_genre"]))
    movie_title_lookup = dict(zip(movies["movie_id"], movies["movie_title"]))

    df = ratings.copy()
    df["u"] = df["user_id"].map(user_to_idx)
    df["m"] = df["movie_id"].map(movie_to_idx)
    df["g"] = df["movie_id"].map(movie_genre_lookup).map(genre_to_idx)

    # Shuffle + split into train/test
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split = int(0.8 * len(df))
    train_df, test_df = df.iloc[:split], df.iloc[split:]

    model = build_model(len(user_ids), len(movie_ids), len(genres))

    print("Training...")
    model.fit(
        x={"user_id": train_df["u"], "movie_id": train_df["m"], "genre_id": train_df["g"]},
        y=train_df["rating"],
        validation_data=(
            {"user_id": test_df["u"], "movie_id": test_df["m"], "genre_id": test_df["g"]},
            test_df["rating"],
        ),
        epochs=10,
        batch_size=64,
        verbose=2,
    )

    # ---- Recommend for one sample user ----
    sample_user_id = user_ids[0]
    u_idx = user_to_idx[sample_user_id]
    rated_movie_ids = set(ratings[ratings["user_id"] == sample_user_id]["movie_id"])
    candidate_movie_ids = [m for m in movie_ids if m not in rated_movie_ids]

    m_idx = np.array([movie_to_idx[m] for m in candidate_movie_ids])
    g_idx = np.array([genre_to_idx[movie_genre_lookup[m]] for m in candidate_movie_ids])
    u_idx_arr = np.full(len(candidate_movie_ids), u_idx)

    preds = model.predict(
        {"user_id": u_idx_arr, "movie_id": m_idx, "genre_id": g_idx}, verbose=0
    ).flatten()

    ranked = sorted(zip(candidate_movie_ids, preds), key=lambda x: -x[1])

    print(f"\nTop 5 recommendations for user {sample_user_id}:")
    for movie_id, score in ranked[:5]:
        title = movie_title_lookup[movie_id]
        genre = movie_genre_lookup[movie_id]
        print(f" - {title}  [{genre}]  (predicted rating: {score:.2f})")


if __name__ == "__main__":
    main()
