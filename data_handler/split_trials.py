import random


def split_trials(trials: list, percentage):
    split_index = len(trials) - int(len(trials) * percentage)

    # Randomly select elements for the first list
    random.seed(10)  # 4
    random.shuffle(trials)
    train = trials[:-split_index]
    val = trials[-split_index:]

    # Remove the selected elements from the original list to get the second list
    return train, val
