def load_all_folds():
    train, test = ([] for _ in range(2))

    for i in range(1, 6):
        i_train, i_test = load_fold(i)

        train.append(i_train)
        test.append(i_test)

    return train, test


def load_fold(fold=1):
    path_base = f'ml-100k/u{fold}.base'
    path_test = f'ml-100k/u{fold}.test'
    train = {}
    test = {}

    def load(path, target_dict):
        with open(path, 'r') as f:
            lines = f.readlines()

            # Line contains: user_id | movie_id | rating | timestamp.
            for line in lines:
                # Ignore timestamp.
                split = line.split('\t')[:-1]

                user_id = int(split[0]) - 1
                movie_rating = {int(split[1]) - 1: int(split[2])}
                if user_id in target_dict:
                    target_dict[user_id].update(movie_rating)
                else:
                    target_dict[user_id] = movie_rating

    load(path_base, train)
    load(path_test, test)

    return train, test
