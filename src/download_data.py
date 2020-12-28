from loguru import logger

from urllib.request import urlretrieve
import os


if __name__ == '__main__':
    n_folds = 5
    target_dir = 'ml-100k'

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    for fold in range(1, n_folds + 1):
        for kind in ['test', 'base']:
            file = f'u{fold}.{kind}'

            logger.info(f'Downloading {file}')
            urlretrieve(f'http://files.grouplens.org/datasets/movielens/ml-100k/{file}', os.path.join(target_dir, file))
