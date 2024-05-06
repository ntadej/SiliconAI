"""Transformer test sequence data generation."""

import random

import numpy as np
from numpy.typing import ArrayLike


def generate_random_data(n: int) -> ArrayLike:
    """Generate random data for testing."""
    token = np.array([1])
    padding = np.array([0])
    length = 8

    data = []

    # 8,8,8,8 -> 8,8,8,8,8
    for _ in range(n // 3):
        x = np.concatenate((token, np.ones(length) * 8, token))
        y = np.concatenate((np.ones(length) * 8, token, padding))
        data.append([x, y])

    # 5,5,5,5 -> 5,5,5,5,5
    for _ in range(n // 3):
        x = np.concatenate((token, np.ones(length) * 5, token))
        y = np.concatenate((np.ones(length) * 5, token, padding))
        data.append([x, y])

    # 5,8,5,8 -> 5,8,5,8,5
    for _ in range(n // 3):
        x = np.ones(length) * 5
        start = random.randint(0, 1)

        x[start::2] = 8

        y = np.ones(length) * 5
        if x[-1] == 5:  # noqa: PLR2004
            y[::2] = 8
        else:
            y[1::2] = 8

        x = np.concatenate((token, x, token))
        y = np.concatenate((y, token, padding))

        data.append([x, y])

    rng = np.random.default_rng(42)
    rng.shuffle(data)

    return data


random_data = generate_random_data(10000)
np.save("data/test_sequence.npy", random_data)
# print(random_data[:5])
