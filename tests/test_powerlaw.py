# -*- coding: utf-8 -*-
"""
Tests for PowerLaw rating model.
"""

import numpy as np
import pytest
from lmfit import Model, Parameters

from hydrating.models import PowerLaw


def test_powerlaw_initializes_one_segment_with_lmfit_parameters():
    model = PowerLaw()

    assert model.segments == 1
    assert isinstance(model.parameters, Parameters)
    assert list(model.parameters.keys()) == ["a", "h0", "b"]
    assert model.parameters["a"].value == 1.0
    assert model.parameters["h0"].value == 0.0
    assert model.parameters["b"].value == 2.0
    assert model.parameter_distributions == {}


def test_powerlaw_func_uses_native_parameter_edits():
    model = PowerLaw()
    model.parameters["a"].value = 0.1
    model.parameters["h0"].value = 0.65
    model.parameters["b"].value = 2.5
    stage = np.array([1.0, 2.0, 3.0])

    discharge = model.func(stage)

    np.testing.assert_allclose(discharge, 0.1 * (stage - 0.65) ** 2.5)


def test_powerlaw_func_accepts_parameter_overrides():
    model = PowerLaw()
    stage = np.array([1.0, 2.0, 3.0])

    discharge = model.func(stage, a=0.2, h0=0.5, b=2.0)

    np.testing.assert_allclose(discharge, 0.2 * (stage - 0.5) ** 2.0)


def test_powerlaw_create_lmfit_model_uses_current_parameter_hints():
    model = PowerLaw()
    model.parameters["b"].value = 2.5
    model.parameters["b"].vary = False
    model.parameters["b"].min = 1.0

    lmfit_model = model.create_lmfit_model()
    parameters = lmfit_model.make_params()

    assert isinstance(lmfit_model, Model)
    assert lmfit_model.param_names == ["a", "h0", "b"]
    assert parameters["b"].value == 2.5
    assert parameters["b"].vary is False
    assert parameters["b"].min == 1.0


def test_powerlaw_lmfit_model_fits_exact_data(powerlaw_reference_data):
    model = PowerLaw()
    model.constrain_parameters(
        powerlaw_reference_data["stage_exact"],
        powerlaw_reference_data["discharge_exact"],
    )

    result = model.create_lmfit_model().fit(
        powerlaw_reference_data["discharge_exact"],
        params=model.parameters.copy(),
        h=powerlaw_reference_data["stage_exact"],
    )

    assert result.success
    assert result.best_values == pytest.approx(powerlaw_reference_data["true_params"])


def test_powerlaw_constrain_parameters_limits_zero_flow_stage():
    model = PowerLaw()

    returned = model.constrain_parameters(
        h=np.array([1.2, 1.5, 2.0]), q=np.array([10.0, 20.0, 40.0])
    )

    assert returned is model.parameters
    assert model.parameters["h0"].max == pytest.approx(1.2 - 1e-10)


def test_powerlaw_constrain_parameters_preserves_lower_existing_maximum():
    model = PowerLaw()
    model.parameters["h0"].max = 0.8

    model.constrain_parameters(
        h=np.array([1.2, 1.5, 2.0]), q=np.array([10.0, 20.0, 40.0])
    )

    assert model.parameters["h0"].max == 0.8


def test_powerlaw_segmented_constrain_parameters_places_breaks_within_stage_range():
    model = PowerLaw(segments=3)

    model.constrain_parameters(
        h=np.array([10.0, 20.0, 30.0]), q=np.array([100.0, 200.0, 300.0])
    )

    assert model.parameters["break1"].value == pytest.approx(50.0 / 3.0)
    assert model.parameters["break2"].value == pytest.approx(70.0 / 3.0)
    assert model.parameters["break1"].min > 10.0
    assert model.parameters["break2"].max < 30.0
    assert model.parameters["h0"].max == pytest.approx(10.0 - 1e-10)


def test_powerlaw_inverse_matches_forward_function():
    model = PowerLaw()
    stage = np.array([1.0, 2.0, 3.0])
    params = {"a": 0.1, "h0": 0.65, "b": 2.5}
    discharge = model.func(stage, **params)

    inverse_stage = model.inverse(discharge, **params)

    np.testing.assert_allclose(inverse_stage, stage)


@pytest.mark.parametrize("segments", [0, -1, 1.5, "1", True])
def test_powerlaw_rejects_invalid_segments(segments):
    with pytest.raises(ValueError, match="segments must be an integer >= 1"):
        PowerLaw(segments=segments)


def test_powerlaw_initializes_two_segment_parameters():
    model = PowerLaw(segments=2)

    assert list(model.parameters.keys()) == [
        "a1",
        "h0",
        "b1",
        "break1",
        "c2",
        "b2",
    ]


def test_powerlaw_initializes_three_segment_parameters():
    model = PowerLaw(segments=3)

    assert list(model.parameters.keys()) == [
        "a1",
        "h0",
        "b1",
        "break1",
        "c2",
        "b2",
        "break2",
        "c3",
        "b3",
    ]


def test_powerlaw_segmented_create_lmfit_model_uses_current_parameter_hints():
    model = PowerLaw(segments=2)
    model.parameters["break1"].value = 2.5
    model.parameters["break1"].vary = False
    model.parameters["b2"].min = 1.0

    lmfit_model = model.create_lmfit_model()
    parameters = lmfit_model.make_params()

    assert lmfit_model.param_names == [
        "a1",
        "h0",
        "b1",
        "break1",
        "c2",
        "b2",
    ]
    assert parameters["break1"].value == 2.5
    assert parameters["break1"].vary is False
    assert parameters["b2"].min == 1.0


def test_powerlaw_two_segment_predictions_are_continuous_at_breakpoint():
    model = PowerLaw(segments=2)
    params = {
        "a1": 2.0,
        "h0": 0.0,
        "b1": 1.0,
        "break1": 2.0,
        "c2": 1.0,
        "b2": 2.0,
    }

    derived = model.derived_parameters(params)
    discharge = model.func(np.array([1.0, 2.0, 3.0]), **params)

    assert derived == pytest.approx({"a2": 4.0})
    np.testing.assert_allclose(discharge, np.array([2.0, 4.0, 16.0]))
    assert model.func(2.0, **params) == pytest.approx(
        derived["a2"] * (params["break1"] - params["c2"]) ** params["b2"]
    )


def test_powerlaw_three_segment_derived_parameters_are_recursive():
    model = PowerLaw(segments=3)
    params = {
        "a1": 2.0,
        "h0": 0.0,
        "b1": 1.0,
        "break1": 2.0,
        "c2": 1.0,
        "b2": 2.0,
        "break2": 4.0,
        "c3": 3.0,
        "b3": 1.0,
    }

    derived = model.derived_parameters(params)
    discharge = model.func(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), **params)

    assert derived == pytest.approx({"a2": 4.0, "a3": 36.0})
    np.testing.assert_allclose(discharge, np.array([2.0, 4.0, 16.0, 36.0, 72.0]))


def test_powerlaw_segmented_rejects_unordered_breakpoints():
    model = PowerLaw(segments=3)
    params = {
        "a1": 2.0,
        "h0": 0.0,
        "b1": 1.0,
        "break1": 4.0,
        "c2": 1.0,
        "b2": 2.0,
        "break2": 3.0,
        "c3": 1.0,
        "b3": 1.0,
    }

    with pytest.raises(ValueError, match="strictly increasing"):
        model.func(np.array([1.0, 2.0, 5.0]), **params)


def test_powerlaw_segmented_rejects_first_segment_offset_at_stage_boundary():
    model = PowerLaw(segments=2)
    params = {
        "a1": 2.0,
        "h0": 1.0,
        "b1": 1.0,
        "break1": 2.0,
        "c2": 1.0,
        "b2": 2.0,
    }

    with pytest.raises(ValueError, match="h0"):
        model.func(np.array([1.0, 2.0, 3.0]), **params)


def test_powerlaw_segmented_rejects_later_segment_offset_at_lower_boundary():
    model = PowerLaw(segments=2)
    params = {
        "a1": 2.0,
        "h0": 0.0,
        "b1": 1.0,
        "break1": 2.0,
        "c2": 2.0,
        "b2": 2.0,
    }

    with pytest.raises(ValueError, match="c2"):
        model.func(np.array([1.0, 2.0, 3.0]), **params)
