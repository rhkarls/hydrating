# -*- coding: utf-8 -*-
"""
Tests for the RatingCurve API.
"""

from collections import OrderedDict

import numpy as np
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


def test_fit_requires_model_argument(powerlaw_data):
    rc = RatingCurve(data=powerlaw_data, h="stage", q="discharge")

    with pytest.raises(TypeError, match="model"):
        # pyrefly: ignore [missing-argument]
        rc.fit("missing-model")


def test_fit_rejects_unsupported_backend(powerlaw_data):
    rc = RatingCurve(data=powerlaw_data, h="stage", q="discharge")

    with pytest.raises(NotImplementedError, match="Unsupported backend"):
        rc.fit("base", model=models.PowerLaw(), backend="other")


def test_fit_stores_named_fit(powerlaw_data):
    rc = RatingCurve(data=powerlaw_data, h="stage", q="discharge")

    fit = rc.fit("base", model=models.PowerLaw())

    assert fit.name == "base"
    assert fit.backend == "lmfit"
    assert fit.result_.success
    assert rc.fits["base"] is fit


def test_rating_curve_fit_powerlaw_exact_data_recovers_parameters(
    powerlaw_reference_data,
):
    data = pd.DataFrame(
        {
            "stage": powerlaw_reference_data["stage_exact"],
            "discharge": powerlaw_reference_data["discharge_exact"],
        }
    )
    rc = RatingCurve(data=data, h="stage", q="discharge")

    fit = rc.fit("exact", model=models.PowerLaw())

    assert fit.result_.success
    assert rc.fits["exact"] is fit
    assert fit.result_.best_values == pytest.approx(
        powerlaw_reference_data["true_params"]
    )


def test_rating_curve_fit_powerlaw_noisy_data_succeeds(powerlaw_reference_data):
    data = pd.DataFrame(
        {
            "stage": powerlaw_reference_data["stage_noisy"],
            "discharge": powerlaw_reference_data["discharge_noisy"],
        }
    )
    rc = RatingCurve(data=data, h="stage", q="discharge")

    fit = rc.fit("noisy", model=models.PowerLaw())

    assert fit.result_.success
    assert set(fit.result_.best_values) == {"a", "h_zero", "b"}


def test_rating_curve_fit_respects_fixed_powerlaw_parameter(powerlaw_reference_data):
    data = pd.DataFrame(
        {
            "stage": powerlaw_reference_data["stage_exact"],
            "discharge": powerlaw_reference_data["discharge_exact"],
        }
    )
    model = models.PowerLaw()
    model.parameters["b"].value = 2.6
    model.parameters["b"].vary = False
    rc = RatingCurve(data=data, h="stage", q="discharge")

    fit = rc.fit("fixed-b", model=model)

    assert fit.result_.success
    assert fit.result_.best_values["b"] == pytest.approx(2.6)


def test_fit_rejects_duplicate_name_without_overwrite(powerlaw_data):
    rc = RatingCurve(data=powerlaw_data, h="stage", q="discharge")
    rc.fit("base", model=models.PowerLaw())

    with pytest.raises(ValueError, match="already exists"):
        rc.fit("base", model=models.PowerLaw())


def test_fit_overwrite_replaces_existing_fit(powerlaw_data):
    rc = RatingCurve(data=powerlaw_data, h="stage", q="discharge")
    first = rc.fit("base", model=models.PowerLaw())

    second = rc.fit("base", model=models.PowerLaw(), overwrite=True)

    assert second is not first
    assert rc.fits["base"] is second


def test_fit_snapshots_enabled_rows(powerlaw_data):
    data = powerlaw_data.copy()
    data["method"] = ["adcp", "adcp", "float", "float", "adcp"]
    rc = RatingCurve(data=data, h="stage", q="discharge")

    fit = rc.enable_where(method="adcp").fit("adcp", model=models.PowerLaw())
    rc.enable_all()

    assert fit.active_data_["stage"].tolist() == [1.0, 2.0, 5.0]
    assert rc.active_data["stage"].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_fit_predict_returns_expected_shape_and_values(powerlaw_reference_data):
    data = pd.DataFrame(
        {
            "stage": powerlaw_reference_data["stage_exact"],
            "discharge": powerlaw_reference_data["discharge_exact"],
        }
    )
    rc = RatingCurve(data=data, h="stage", q="discharge")
    fit = rc.fit("exact", model=models.PowerLaw())
    stage = np.array([1.5, 3.0, 7.5])

    predicted = fit.predict(stage)

    assert predicted.shape == stage.shape
    np.testing.assert_allclose(
        predicted,
        models.PowerLaw().func(stage, **powerlaw_reference_data["true_params"]),
        rtol=1e-5,
    )


def test_fit_metrics_are_populated_after_fitting(powerlaw_reference_data):
    data = pd.DataFrame(
        {
            "stage": powerlaw_reference_data["stage_noisy"],
            "discharge": powerlaw_reference_data["discharge_noisy"],
        }
    )
    rc = RatingCurve(data=data, h="stage", q="discharge")

    fit = rc.fit("noisy", model=models.PowerLaw())

    metrics = [
        fit.aic,
        fit.bic,
        fit.reduced_chi,
        fit.r2,
        fit.mean_absolute_error,
        fit.mean_percentage_error,
        fit.mean_absolute_percentage_error,
    ]
    assert all(isinstance(metric, float) for metric in metrics)
    assert all(np.isfinite(metric) for metric in metrics)
    assert fit.r2 == pytest.approx(fit.result_.rsquared)


def test_fit_calculates_error_metrics_from_known_data():
    data = pd.DataFrame(
        {
            "stage": [12.0, 18.0, 44.0],
            "discharge": [10.0, 20.0, 40.0],
        }
    )
    model = models.PowerLaw()
    model.parameters["a"].value = 1.0
    model.parameters["a"].vary = False
    model.parameters["h_zero"].value = 0.0
    model.parameters["h_zero"].vary = False
    model.parameters["b"].value = 1.0
    model.parameters["b"].vary = False
    rc = RatingCurve(data=data, h="stage", q="discharge")

    fit = rc.fit("known", model=model)

    assert fit.mean_absolute_error == pytest.approx(8.0 / 3.0)
    assert fit.mean_percentage_error == pytest.approx(20.0 / 3.0)
    assert fit.mean_absolute_percentage_error == pytest.approx(40.0 / 3.0)


def test_fit_params_exposes_fitted_parameter_values(powerlaw_reference_data):
    data = pd.DataFrame(
        {
            "stage": powerlaw_reference_data["stage_exact"],
            "discharge": powerlaw_reference_data["discharge_exact"],
        }
    )
    rc = RatingCurve(data=data, h="stage", q="discharge")

    fit = rc.fit("exact", model=models.PowerLaw())

    assert fit.params_ == pytest.approx(powerlaw_reference_data["true_params"])
    assert fit.derived_params_ == {}


def test_fit_active_data_snapshot_is_independent_after_later_filtering(powerlaw_data):
    rc = RatingCurve(data=powerlaw_data, h="stage", q="discharge")

    fit = rc.enable([True, True, False, False, True]).fit(
        "selected", model=models.PowerLaw()
    )
    rc.enable_where(stage=3.0)

    assert fit.active_data_["stage"].tolist() == [1.0, 2.0, 5.0]
    assert rc.active_data["stage"].tolist() == [3.0]


def test_rating_curve_predict_works_with_one_fit(powerlaw_reference_data):
    data = pd.DataFrame(
        {
            "stage": powerlaw_reference_data["stage_exact"],
            "discharge": powerlaw_reference_data["discharge_exact"],
        }
    )
    stage = pd.Series(
        [1.5, 3.0],
        index=pd.to_datetime(["2026-01-01 00:00", "2026-01-01 01:00"]),
    )
    rc = RatingCurve(data=data, h="stage", q="discharge")
    fit = rc.fit("base", model=models.PowerLaw())

    predicted = rc.predict(stage)

    assert list(predicted.columns) == ["base"]
    assert predicted.index.equals(stage.index)
    np.testing.assert_allclose(predicted["base"].to_numpy(), fit.predict(stage))


def test_rating_curve_predict_works_with_multiple_fits(powerlaw_reference_data):
    data = pd.DataFrame(
        {
            "stage": powerlaw_reference_data["stage_exact"],
            "discharge": powerlaw_reference_data["discharge_exact"],
        }
    )
    fixed_b = models.PowerLaw()
    fixed_b.parameters["b"].value = 2.6
    fixed_b.parameters["b"].vary = False
    rc = RatingCurve(data=data, h="stage", q="discharge")
    rc.fit("base", model=models.PowerLaw())
    rc.fit("fixed-b", model=fixed_b)

    predicted = rc.predict([1.5, 3.0])

    assert list(predicted.columns) == ["base", "fixed-b"]
    assert predicted.shape == (2, 2)


def test_rating_curve_predict_supports_selected_fits(powerlaw_reference_data):
    data = pd.DataFrame(
        {
            "stage": powerlaw_reference_data["stage_exact"],
            "discharge": powerlaw_reference_data["discharge_exact"],
        }
    )
    fixed_b = models.PowerLaw()
    fixed_b.parameters["b"].value = 2.6
    fixed_b.parameters["b"].vary = False
    rc = RatingCurve(data=data, h="stage", q="discharge")
    rc.fit("base", model=models.PowerLaw())
    rc.fit("fixed-b", model=fixed_b)

    predicted = rc.predict([1.5, 3.0], fits=["fixed-b", "base"])

    assert list(predicted.columns) == ["fixed-b", "base"]


def test_rating_curve_predict_rejects_unknown_fit_name(powerlaw_data):
    rc = RatingCurve(data=powerlaw_data, h="stage", q="discharge")
    rc.fit("base", model=models.PowerLaw())

    with pytest.raises(KeyError, match="Unknown fit name"):
        rc.predict([1.0, 2.0], fits="missing")

def test_public_imports_work():
    assert RatingCurve is not None
    assert models.PowerLaw is not None
