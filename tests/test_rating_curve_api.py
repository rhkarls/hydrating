# -*- coding: utf-8 -*-
"""
Tests for the RatingCurve API skeleton.
"""

from collections import OrderedDict

import pandas as pd
import pytest

from hydrating import RatingCurve, models


def test_rating_curve_initializes_with_data_columns():
    data = pd.DataFrame({"stage": [1.0, 2.0], "discharge": [10.0, 25.0]})

    rc = RatingCurve(data=data, h="stage", q="discharge")

    assert rc.h_col == "stage"
    assert rc.q_col == "discharge"
    assert rc.enabled_col == "enabled"
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

    assert rc.enabled_col == "use_fit"
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


def test_enable_updates_active_rows_from_boolean_mask():
    data = pd.DataFrame(
        {
            "stage": [1.0, 2.0, 3.0],
            "discharge": [10.0, 25.0, 50.0],
        }
    )

    rc = RatingCurve(data=data, h="stage", q="discharge")
    returned = rc.enable([True, False, True])

    assert returned is rc
    assert rc.data["enabled"].tolist() == [True, False, True]
    assert rc.active_data["stage"].tolist() == [1.0, 3.0]


def test_enable_rejects_mask_length_mismatch():
    data = pd.DataFrame({"stage": [1.0, 2.0], "discharge": [10.0, 25.0]})
    rc = RatingCurve(data=data, h="stage", q="discharge")

    with pytest.raises(ValueError, match="mask length must match data length"):
        rc.enable([True])


def test_enable_rejects_non_boolean_mask_values():
    data = pd.DataFrame({"stage": [1.0, 2.0], "discharge": [10.0, 25.0]})
    rc = RatingCurve(data=data, h="stage", q="discharge")

    with pytest.raises(ValueError, match="mask must contain boolean values"):
        # pyrefly: ignore [bad-argument-type]
        rc.enable([1, 0])


def test_enable_where_supports_callable_filters():
    data = pd.DataFrame(
        {
            "stage": [1.0, 2.0, 3.0],
            "discharge": [10.0, 25.0, 50.0],
            "method": ["adcp", "ice", "adcp"],
        }
    )

    rc = RatingCurve(data=data, h="stage", q="discharge")
    returned = rc.enable_where(method=lambda s: s != "ice")

    assert returned is rc
    assert rc.data["enabled"].tolist() == [True, False, True]
    assert rc.active_data["method"].tolist() == ["adcp", "adcp"]


def test_enable_where_supports_exact_value_filters():
    data = pd.DataFrame(
        {
            "stage": [1.0, 2.0, 3.0],
            "discharge": [10.0, 25.0, 50.0],
            "method": ["adcp", "float", "adcp"],
        }
    )

    rc = RatingCurve(data=data, h="stage", q="discharge")
    rc.enable_where(method="adcp")

    assert rc.data["enabled"].tolist() == [True, False, True]
    assert rc.active_data["stage"].tolist() == [1.0, 3.0]


def test_enable_where_combines_multiple_conditions():
    data = pd.DataFrame(
        {
            "stage": [1.0, 2.0, 3.0, 4.0],
            "discharge": [10.0, 25.0, 50.0, 80.0],
            "method": ["adcp", "adcp", "float", "adcp"],
            "quality": ["good", "review", "good", "good"],
        }
    )

    rc = RatingCurve(data=data, h="stage", q="discharge")
    rc.enable_where(method="adcp", quality="good")

    assert rc.data["enabled"].tolist() == [True, False, False, True]
    assert rc.active_data["stage"].tolist() == [1.0, 4.0]


def test_enable_methods_are_chainable():
    data = pd.DataFrame(
        {
            "stage": [1.0, 2.0, 3.0],
            "discharge": [10.0, 25.0, 50.0],
            "method": ["adcp", "float", "adcp"],
        }
    )

    rc = RatingCurve(data=data, h="stage", q="discharge")
    returned = rc.enable_where(method="adcp").enable([False, True, False])

    assert returned is rc
    assert rc.data["enabled"].tolist() == [False, True, False]
    assert rc.active_data["method"].tolist() == ["float"]


def test_enable_all_enables_every_row():
    data = pd.DataFrame(
        {
            "stage": [1.0, 2.0, 3.0],
            "discharge": [10.0, 25.0, 50.0],
            "enabled": [True, False, True],
        }
    )

    rc = RatingCurve(data=data, h="stage", q="discharge")
    returned = rc.enable_all()

    assert returned is rc
    assert rc.data["enabled"].tolist() == [True, True, True]
    assert rc.active_data["stage"].tolist() == [1.0, 2.0, 3.0]


def test_enable_none_disables_every_row():
    data = pd.DataFrame(
        {
            "stage": [1.0, 2.0, 3.0],
            "discharge": [10.0, 25.0, 50.0],
            "enabled": [True, False, True],
        }
    )

    rc = RatingCurve(data=data, h="stage", q="discharge")
    returned = rc.enable_none()

    assert returned is rc
    assert rc.data["enabled"].tolist() == [False, False, False]
    assert rc.active_data.empty


def test_enable_reset_restores_all_true_mask_when_enabled_column_was_missing():
    data = pd.DataFrame(
        {
            "stage": [1.0, 2.0, 3.0],
            "discharge": [10.0, 25.0, 50.0],
        }
    )

    rc = RatingCurve(data=data, h="stage", q="discharge")
    returned = rc.enable([False, True, False]).enable_reset()

    assert returned is rc
    assert rc.data["enabled"].tolist() == [True, True, True]
    assert rc.active_data["stage"].tolist() == [1.0, 2.0, 3.0]


def test_enable_reset_restores_constructor_mask_when_enabled_column_existed():
    data = pd.DataFrame(
        {
            "stage": [1.0, 2.0, 3.0],
            "discharge": [10.0, 25.0, 50.0],
            "enabled": [True, False, True],
        }
    )

    rc = RatingCurve(data=data, h="stage", q="discharge")
    rc.enable_all().enable_reset()

    assert rc.data["enabled"].tolist() == [True, False, True]
    assert rc.active_data["stage"].tolist() == [1.0, 3.0]


def test_public_imports_work():
    assert RatingCurve is not None
    assert models.PowerLaw is not None
