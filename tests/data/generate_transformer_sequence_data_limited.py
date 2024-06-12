"""Transformer test sequence data generation."""

import random

import numpy as np
from numpy.typing import ArrayLike


def generate_random_data(n: int, constants: list[int] | None = None) -> ArrayLike:
    """Generate random data for testing."""
    token_start = np.array([1])
    token_end = np.array([2])
    padding = np.array([0])
    length = 8

    data: list[list[int]] = []

    # 8,8,8,8 -> 8,8,8,8,8
    for _ in range(n // 3):
        x = np.concatenate((token_start, np.ones(length) * 8, token_end))
        y = np.concatenate((np.ones(length) * 8, token_end, padding))
        data.append([x, y])

    # 5,5,5,5 -> 5,5,5,5,5
    for _ in range(n // 3):
        x = np.concatenate((token_start, np.ones(length) * 5, token_end))
        y = np.concatenate((np.ones(length) * 5, token_end, padding))
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

        x = np.concatenate((token_start, x, token_end))
        y = np.concatenate((y, token_end, padding))

        data.append([x, y])

    rng = np.random.default_rng(42)
    rng.shuffle(data)

    if constants:
        data_array = np.asarray(data)
        data_array = data_array.reshape(-1, 2, 10, 1)
        data_array_split = np.array_split(data_array, len(constants))
        for i in range(len(constants)):
            data_array_split[i] = np.insert(
                data_array_split[i],
                1,
                [constants[i]],
                axis=3,
            )
        data_array = np.concatenate(data_array_split)

        data_array[:, 1, -1, 1] = 0

        rng.shuffle(data_array)

        return data_array

    return data


random_data = generate_random_data(10000, constants=[1, 2])
np.save("data/test_sequence.npy", random_data)
# print(random_data[:5])
