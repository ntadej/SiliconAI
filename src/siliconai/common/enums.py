# Copyright (C) 2024 Tadej Novak
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# SPDX-License-Identifier: MPL-2.0

"""Common enumerations."""

from enum import Enum


class DataLoadingType(str, Enum):
    """Data loading type."""

    fit = "fit"
    validate = "validate"
    test = "test"
    predict = "predict"


class ColumnType(Enum):
    """Column type."""

    Categorical = "categorical"
    Numerical = "numerical"


class ModelType(Enum):
    """Model type."""

    NanoGPT = "NanoGPT"
