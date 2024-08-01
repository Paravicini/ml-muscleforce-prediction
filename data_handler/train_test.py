from sklearn.model_selection import train_test_split


def split(train_units_list: list, test_size: float = 0.2, random_state: int = 42):
    train, test = train_test_split(train_units_list, test_size=test_size, random_state=random_state)
    return train, test
