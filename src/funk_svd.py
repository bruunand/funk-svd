import pickle
import random

import numpy as np
from loguru import logger

from load_data import load_all_folds

n_latent_factors = 20
learning_rate = 0.01
regularizer = 0.02
max_epochs = 100
stop_threshold = 0.005


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


def initialize_latent_vectors(n_movies, n_users):
    movie_values = np.random.rand(n_movies, n_latent_factors)
    user_values = np.random.rand(n_latent_factors, n_users)

    return movie_values, user_values


def calculate_rmse(on_set, movie_values, user_values):
    # Compute the predicted rating matrix
    predicted = np.matmul(movie_values, user_values)

    n_instances = 0
    sum_squared_errors = 0
    for user, movie_rating in on_set.items():
        n_instances += len(movie_rating.keys())

        for movie, rating in movie_rating.items():
            sum_squared_errors += (predicted[movie][user] - rating) ** 2

    return np.sqrt(sum_squared_errors / n_instances)


def run(train):
    movie_set = get_movies(train)
    movie_values, user_values = initialize_latent_vectors(max(movie_set) + 1, max(train.keys()) + 1)

    # Training instances are represented as a list of triples
    triples = get_triples(train)
    last_rmse = None

    for epoch in range(max_epochs):
        # At the start of every epoch, we shuffle the dataset
        # Shuffling may not be strictly necessary, but is an attempt to avoid overfitting
        random.shuffle(triples)

        # Calculate RMSE for training set, stop if change is below threshold
        rmse = calculate_rmse(train, movie_values, user_values)
        logger.info(f'Epoch {epoch}, RMSE: {rmse}')
        if last_rmse and abs(rmse - last_rmse) < stop_threshold:
            break
        last_rmse = rmse

        for user, movie, rating in triples:
            # Update values in vector movie_values
            for k in range(n_latent_factors):
                error = sum(movie_values[movie][i] * user_values[i][user] for i in range(n_latent_factors)) - rating

                # Compute the movie gradient
                # Update the kth movie factor for the current movie
                movie_gradient = error * user_values[k][user]
                movie_values[movie][k] -= learning_rate * (movie_gradient - regularizer * movie_values[movie][k])

                # Compute the user gradient
                # Update the kth user factor the the current user
                user_gradient = error * movie_values[movie][k]
                user_values[k][user] -= learning_rate * (user_gradient - regularizer * user_values[k][user])

    return movie_values, user_values


def test_latent_factors(factors, train_folds, test_folds, n_folds=5):
    n_folds = min(n_folds, len(train_folds), len(test_folds))
    results = dict()

    for factor in factors:
        global n_latent_factors
        n_latent_factors = factor
        test_rmse_results = []
        train_rmse_results = []

        # Test for each fold
        for i in range(n_folds):
            train = train_folds[i]
            test = test_folds[i]

            movie_values, user_values = run(train)
            train_rmse_results.append(calculate_rmse(train, movie_values, user_values))
            test_rmse_results.append(calculate_rmse(test, movie_values, user_values))

        train_mean = np.mean(train_rmse_results)
        test_mean = np.mean(test_rmse_results)

        logger.info(f'Finished testing for latent dimension size of {factor}')
        logger.info(f'Average test RMSE: {test_mean}')
        logger.info(f'Average train RMSE: {train_mean}')

        results[factor] = {'train': train_mean, 'test': test_mean}

        # Dump results after every factor
        pickle.dump(results, open('results.pkl', 'wb'))


if __name__ == "__main__":
    test_latent_factors([5,  10, 15, 20, 25, 30, 35, 40], *load_all_folds(), n_folds=2)
