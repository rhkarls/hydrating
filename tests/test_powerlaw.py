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
    assert list(model.parameters.keys()) == ["a", "h_zero", "b"]
    assert model.parameters["a"].value == 1.0
    assert model.parameters["h_zero"].value == 0.0
    assert model.parameters["b"].value == 2.0
    assert model.parameter_distributions == {}


def test_powerlaw_func_uses_native_parameter_edits():
    model = PowerLaw()
    model.parameters["a"].value = 0.1
    model.parameters["h_zero"].value = 0.65
    model.parameters["b"].value = 2.5
    stage = np.array([1.0, 2.0, 3.0])

    discharge = model.func(stage)

    np.testing.assert_allclose(discharge, 0.1 * (stage - 0.65) ** 2.5)


def test_powerlaw_func_accepts_parameter_overrides():
    model = PowerLaw()
    stage = np.array([1.0, 2.0, 3.0])

    discharge = model.func(stage, a=0.2, h_zero=0.5, b=2.0)

    np.testing.assert_allclose(discharge, 0.2 * (stage - 0.5) ** 2.0)


def test_powerlaw_create_lmfit_model_uses_current_parameter_hints():
    model = PowerLaw()
    model.parameters["b"].value = 2.5
    model.parameters["b"].vary = False
    model.parameters["b"].min = 1.0

    lmfit_model = model.create_lmfit_model()
    parameters = lmfit_model.make_params()

    assert isinstance(lmfit_model, Model)
    assert lmfit_model.param_names == ["a", "h_zero", "b"]
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
    assert model.parameters["h_zero"].max == pytest.approx(1.2 - 1e-10)


def test_powerlaw_constrain_parameters_preserves_lower_existing_maximum():
    model = PowerLaw()
    model.parameters["h_zero"].max = 0.8

    model.constrain_parameters(
        h=np.array([1.2, 1.5, 2.0]), q=np.array([10.0, 20.0, 40.0])
    )

    assert model.parameters["h_zero"].max == 0.8


def test_powerlaw_inverse_matches_forward_function():
    model = PowerLaw()
    stage = np.array([1.0, 2.0, 3.0])
    params = {"a": 0.1, "h_zero": 0.65, "b": 2.5}
    discharge = model.func(stage, **params)

    inverse_stage = model.inverse(discharge, **params)

    np.testing.assert_allclose(inverse_stage, stage)


@pytest.mark.parametrize("segments", [0, -1, 1.5, "1", True])
def test_powerlaw_rejects_invalid_segments(segments):
    with pytest.raises(ValueError, match="segments must be an integer >= 1"):
        PowerLaw(segments=segments)


def test_powerlaw_reports_unimplemented_segmented_model():
    with pytest.raises(NotImplementedError, match="segments > 1"):
        PowerLaw(segments=2)
