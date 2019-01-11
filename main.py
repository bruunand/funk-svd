import random

import numpy as np
from random import choice

from loguru import logger

from load_data import load_fold

n_latent_factors = 20
learning_rate = 0.001
regularizer = 0.00
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


def get_random_pair(training_set):
    """ Get a random movie-user pair with the associated rating """
    user = choice(list(training_set.keys()))
    movie = choice(list(training_set[user].keys()))

    return user, movie, training_set[user][movie]


def get_singular_vectors(n_movies, n_users):
    """ Return the left and right singular vectors """
    # Initialize singular value vectors
    A = np.random.rand(n_movies, n_latent_factors)
    B = np.random.rand(n_latent_factors, n_users)

    return A, B
    # Matrix multiplication results in a rating matrix
    # Should be size (movies, users)
    # m = np.matmul(A, B)


def run_sgd():
    pass


def calculate_rmse(on_set, A, B):
    # Compute the rating matrix R
    R = np.matmul(A, B)

    # Calculate number of test instances
    n_test_instances = 0
    sum_squared_errors = 0
    for user, movie_rating in on_set.items():
        n_test_instances += len(movie_rating.keys())

        for movie, rating in movie_rating.items():
            sum_squared_errors += pow(R[movie][user] - rating, 2)

    return np.sqrt(sum_squared_errors / n_test_instances)


def run():
    train, test = load_fold(1)

    # Retrieve the singular vectors
    # A can be interpreted as movie factors
    # B can be interpreted as user factors
    A, B = get_singular_vectors(1700, 2000)

    triples = get_triples(train)
    for epoch in range(max_epochs):
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
                A[movie][k] += learning_rate * (gradient - regularizer * B[k][user])


if __name__ == "__main__":
    run()
