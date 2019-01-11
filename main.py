import numpy as np
from random import choice

from loguru import logger

from load_data import load_fold

n_latent_factors = 20
learning_rate = 0.001
regularizer = 0.02


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
    # The ith entry corresponds to the ith users
    A = np.random.rand(n_movies, n_latent_factors)
    B = np.random.rand(n_latent_factors, n_users)

    return A, B
    # Matrix multiplication results in a rating matrix
    # Should be size (movies, users)
    # m = np.matmul(A, B)


def run_sgd():
    pass


def calculate_rmse(on_set, A, B):
    # Calculate R
    R = np.matmul(A, B)

    # Calculate number of test instances
    n_test_instances = 0
    sum_squared_errors = 0
    for user, movie_rating in on_set.items():
        n_test_instances += len(movie_rating.keys())

        for movie, rating in movie_rating.items():
            sum_squared_errors += pow(R[movie][user] - rating, 2)

    print(sum_squared_errors)

    return np.sqrt(sum_squared_errors / n_test_instances)


def run():
    train, test = load_fold(1)

    # Retrieve the singular vectors
    # A can be interpreted as movie factors
    # B can be interpreted as user factors
    A, B = get_singular_vectors(1700, 2000)

    iterations = 0
    a, b, r = get_random_pair(train)
    while True:
        movie, user, rating = get_random_pair(train)

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

        if iterations % 5000 == 0:
            logger.info(f'Iteration {iterations}, RMSE: {calculate_rmse(train, A, B)}')
            logger.info(f'Test RMSE: {calculate_rmse(test, A, B)}')
        iterations += 1


if __name__ == "__main__":
    run()
