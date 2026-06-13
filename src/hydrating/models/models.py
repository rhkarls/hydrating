# -*- coding: utf-8 -*-
"""
Rating curve models.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from lmfit import Model, Parameters


class RatingModel(Protocol):
    """
    Protocol class for rating curve models.

    Methods for RatingModel:

    parameters: the native lmfit Parameters object for user parameter edits
    func: the rating curve equation, with arguments stage and **parameters, returning
          discharge
    create_lmfit_model: function that returns the lmfit Model
    constrain_parameters: function that puts limits on Parameters based on observations
                          e.g. zero flow stage cannot exceed observed stage with flow
    inverse: function that return stage for a given discharge, the inverse of func()
    """

    parameters: Parameters
    parameter_distributions: dict[str, Any]

    def func(self, h: np.ndarray, **parameters: float) -> np.ndarray:
        """
        The rating curve equation.

        Parameters
        ----------
        h : np.ndarray
            Stage values.
        **parameters : float
            Model parameters.

        Returns
        -------
        float or np.ndarray
            Discharge for the provided stage.
        """
        ...

    def create_lmfit_model(self) -> Model:
        """
        Create the lmfit Model.

        Returns
        -------
        Model
            The lmfit Model for the rating curve.
        """
        ...

    def constrain_parameters(self, h: np.ndarray, q: np.ndarray) -> Parameters:
        """
        Constraining on the parameters based on x and y values.
        This function should put limits on Parameters based on observations.
        For example, zero flow stage cannot exceed observed stage with flow.

        Parameters
        ----------
        h : np.ndarray
            Stage values.
        q : np.ndarray
            Discharge values.

        Returns
        -------
        Parameters
            The lmfit Parameters with limits set based on observations.
        """
        ...

    def inverse(
        self, q: np.ndarray, initial_guess: float | None = None, **parameters: float
    ) -> np.ndarray:
        """
        Inverse the rating curve to find the stage at a given discharge.
        This method should return the stage corresponding to the given discharge.

        Parameters
        ----------
        q : np.ndarray
            Discharge values.
        initial_guess : float, optional
            Initial guess for the stage corresponding to the given discharge, used for numerical methods if needed.
        **parameters : float
            Model parameters.

        Returns
        -------
        np.ndarray
            Stage values corresponding to the given discharge.
        """
        ...


class PowerLaw:
    """
    Power law rating curve model.
    """

    def __init__(self, segments: int = 1):
        if type(segments) is not int or segments < 1:
            raise ValueError("segments must be an integer >= 1.")
        if segments != 1:
            raise NotImplementedError("PowerLaw segments > 1 are not implemented yet.")

        self.segments = segments
        self.parameters = self._default_parameters()
        self.parameter_distributions: dict[str, Any] = {}

    def func(self, h: np.ndarray, **params: float) -> np.ndarray:
        """
        The power law rating curve function.

        .. math::
            Q(h) = a \\times (h-h_zero)^b

        where :math:`Q` is discharge, :math:`h` is stage, :math:`h_zero` is
        stage at zero flow, and :math:`a` and :math:`b` are fitted parameters.

        Parameters
        ----------
        h : float or np.ndarray
            Stage.
        **params : float
            Optional parameter overrides. Missing values are read from
            ``self.parameters``.

        Returns
        -------
        float or np.ndarray
            Discharge for the provided stage.
        """
        values = self._parameter_values(params)
        return self._power_law(h, a=values["a"], h_zero=values["h_zero"], b=values["b"])

    def create_lmfit_model(self) -> Model:
        """
        Create the lmfit Model.

        Returns
        -------
        Model
            The lmfit Model for the power law rating curve.
        """
        lmfit_model = Model(self._func_for_lmfit)

        # copy the parameter settings to the model
        # to make internal parameter behaviour consistent with the lmfit Model.fit() method
        for name, parameter in self.parameters.items():
            hint = {
                "value": parameter.value,
                "vary": parameter.vary,
                "min": parameter.min,
                "max": parameter.max,
            }
            if parameter.expr is not None:
                hint["expr"] = parameter.expr
            if parameter.brute_step is not None:
                hint["brute_step"] = parameter.brute_step
            lmfit_model.set_param_hint(name, **hint)
        return lmfit_model

    def constrain_parameters(self, h: np.ndarray, q: np.ndarray) -> Parameters:
        """
        Constraining on the parameters based on observed values.
        This function sets the maximum value of h_zero to the minimum value of h, i.e. stage.
        This is to avoid fitting a rating curve with zero flow stage that is higher than the observed stage with flow, which is not physically possible.
        Avoid using this constrain if the observed stage goes below zero flow (i.e. flow is zero in the timeseries).

        Parameters
        ----------
        h : np.ndarray
            Stage values.
        q : np.ndarray
            Discharge values. Not used in this function.

        Returns
        -------
        Parameters
            The lmfit Parameters with limits set for max h_zero.
        """
        del q

        h_zero_ceiling = min(float(np.min(h)) - 1e-10, self.parameters["h_zero"].max)
        self.parameters["h_zero"].max = h_zero_ceiling

        return self.parameters

    def inverse(
        self, q: np.ndarray, initial_guess: float | None = None, **params: float
    ) -> np.ndarray:
        """
        Inverse the rating curve to find the stage at a given discharge.

        Parameters
        ----------
        q : np.ndarray
            Discharge values.
        initial_guess : float, optional
            Initial guess not used in this function.
        **params : float
            Optional parameter overrides. Missing values are read from
            ``self.parameters``.

        Returns
        -------
        np.ndarray
            Stage values corresponding to the given discharge.
        """
        del initial_guess

        values = self._parameter_values(params)
        return (np.asarray(q) / values["a"]) ** (1 / values["b"]) + values["h_zero"]

    @staticmethod
    def _default_parameters() -> Parameters:
        parameters = Parameters()
        parameters.add("a", value=1.0)
        parameters.add("h_zero", value=0.0)
        parameters.add("b", value=2.0)
        return parameters

    @staticmethod
    def _power_law(h: np.ndarray, *, a: float, h_zero: float, b: float) -> np.ndarray:
        return a * (np.asarray(h) - h_zero) ** b

    def _func_for_lmfit(
        self, h: np.ndarray, a: float = 1.0, h_zero: float = 0.0, b: float = 2.0
    ) -> np.ndarray:
        """The equation/function that lmfit will fit."""
        return self._power_law(h, a=a, h_zero=h_zero, b=b)

    def _parameter_values(self, overrides: dict[str, float]) -> dict[str, float]:
        unknown = set(overrides) - set(self.parameters)
        if unknown:
            raise KeyError(f"Unknown PowerLaw parameter(s): {sorted(unknown)}")
        values = {
            name: float(parameter.value) for name, parameter in self.parameters.items()
        }
        values.update(overrides)
        return values
