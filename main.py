import random

import numpy as np
from loguru import logger

from load_data import load_fold

# n20~1.12

n_latent_factors = 20
learning_rate = 0.001
regularizer = 0.02
max_epochs = 100


def get_triples(from_set):
    triples = list()

    for user, movie_ratings in from_set.items():
        for movie, rating in movie_ratings.items():
            triples.append((user, movie, rating))

    return triples


def get_movies(from_set):
    movies = set()

    for user, movie_ratings in from_set.items():
        movies.update(movie_ratings.keys())

    return movies


def get_singular_vectors(n_movies, n_users):
    """ Return the left and right singular vectors """
    # Initialize singular value vectors
    A = np.random.rand(n_movies, n_latent_factors)
    B = np.random.rand(n_latent_factors, n_users)

    return A, B
    # Matrix multiplication results in a rating matrix
    # Should be size (movies, users)
    # m = np.matmul(A, B)


def calculate_rmse(on_set, A, B):
    # Compute the rating matrix R
    R = np.matmul(A, B)

    n_instances = 0
    sum_squared_errors = 0
    for user, movie_rating in on_set.items():
        n_instances += len(movie_rating.keys())

        for movie, rating in movie_rating.items():
            sum_squared_errors += pow(R[movie][user] - rating, 2)

    return np.sqrt(sum_squared_errors / n_instances)


def run():
    train, test = load_fold(1)

    # Construct the singular vectors
    movies = get_movies(train)
    A, B = get_singular_vectors(max(movies) + 1, max(train.keys()) + 1)

    # Training instances are represented as a list of tripes
    triples = get_triples(train)

    for epoch in range(max_epochs):
        # At the start of every epoch, we shuffle the dataset
        # Shuffling may not be strictly necessary, but is an attempt to avoid overfitting
        random.shuffle(triples)

        # Calculate RMSE for training set
        logger.info(f'Epoch {epoch}, RMSE: {calculate_rmse(train, A, B)}')

        for user, movie, rating in triples:
            # Update values in vector A
            for k in range(n_latent_factors):
                t_sum = 0
                for i in range(n_latent_factors):
                    t_sum += A[movie][i] * B[i][user]

                gradient = (rating - t_sum) * B[k][user]
                A[movie][k] += learning_rate * (gradient - regularizer * A[movie][k])

            # Update values in vector B
            for k in range(n_latent_factors):
                t_sum = 0
                for i in range(n_latent_factors):
                    t_sum += A[movie][i] * B[i][user]

                gradient = (rating - t_sum) * A[movie][k]

                # Update the kth factor with respect to the gradient and learning rate
                A[movie][k] += learning_rate * (gradient - regularizer * B[k][user])


if __name__ == "__main__":
    run()
