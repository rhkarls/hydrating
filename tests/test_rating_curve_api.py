# -*- coding: utf-8 -*-
"""
Tests for the RatingCurve API skeleton.
"""

from collections import OrderedDict

import pandas as pd

from hydrating import RatingCurve, models


def test_rating_curve_initializes_with_data_columns():
    data = pd.DataFrame({"stage": [1.0, 2.0], "discharge": [10.0, 25.0]})

    rc = RatingCurve(data=data, h="stage", q="discharge")

    assert rc.h == "stage"
    assert rc.q == "discharge"
    assert rc.enabled == "enabled"
    assert rc.metadata is None
    assert rc.data is not data
    assert isinstance(rc.fits, OrderedDict)
    assert list(rc.fits.items()) == []


def test_missing_enabled_column_creates_all_true_enabled_column():
    data = pd.DataFrame({"stage": [1.0, 2.0], "discharge": [10.0, 25.0]})

    rc = RatingCurve(data=data, h="stage", q="discharge")

    assert rc.data["enabled"].tolist() == [True, True]
    assert rc.active_data["stage"].tolist() == [1.0, 2.0]


def test_missing_custom_enabled_column_creates_all_true_column():
    data = pd.DataFrame({"stage": [1.0, 2.0], "discharge": [10.0, 25.0]})

    rc = RatingCurve(data=data, h="stage", q="discharge", enabled="use_fit")

    assert rc.enabled == "use_fit"
    assert rc.data["use_fit"].tolist() == [True, True]


def test_existing_enabled_column_is_respected():
    data = pd.DataFrame(
        {
            "stage": [1.0, 2.0, 3.0],
            "discharge": [10.0, 25.0, 50.0],
            "enabled": [True, False, True],
        }
    )

    rc = RatingCurve(data=data, h="stage", q="discharge")

    assert rc.data["enabled"].tolist() == [True, False, True]
    assert rc.active_data["stage"].tolist() == [1.0, 3.0]


def test_public_imports_work():
    assert RatingCurve is not None
    assert models.PowerLaw is not None
