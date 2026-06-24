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

        self.segments = segments
        self.parameters = self._default_parameters()
        self.parameter_distributions: dict[str, Any] = {}

    def func(self, h: float | np.ndarray, **params: float) -> np.ndarray:
        """
        The power law rating curve function.

        For a single segment (or section) rating curve, the function is:
        .. math::
            Q(h) = a \\times (h-h0)^b

        where :math:`Q` is discharge, :math:`h` is stage, :math:`h0` is
        stage at zero flow, and :math:`a` and :math:`b` are fitted parameters.

        For a multi-segment rating curve the powerlaw is applied to each segmented, defined by the breakpoint stage.
        For example, for a two segment rating curve the function is:
        .. math::
            Q(h) = \\begin{cases}
                a_1 \\times (h-h0)^{b_1} & h < break1 \\\\
                a_2 \\times (h-c_2)^{b_2} & h \\geq break1
            \\end{cases}

        where :math:`break1` is the stage at which the rating curve changes from the first segment to the second segment.

        For continuity, the second segment and the following segments are defined by the previous segment parameters and the breakpoint stage,
        so that the rating curve is continuous at the breakpoint stage. For example, for a two segment rating curve, :math:`a_2` is defined as:

        :math:`a_2 = a_1 \\times (break1 - h0)^{b_1} / (break1 - c_2)^{b_2}`.

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
        if self.segments == 1:
            return self._power_law(h, a=values["a"], h0=values["h0"], b=values["b"])
        return self._segmented_power_law(h, values)

    def create_lmfit_model(self) -> Model:
        """
        Create the lmfit Model.

        Returns
        -------
        Model
            The lmfit Model for the power law rating curve.
        """
        lmfit_model = Model(
            self._create_lmfit_function(),
            independent_vars=["h"],
            param_names=list(self.parameters.keys()),
        )

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
        Constraining on the parameters based on observed values and what is physically possible for the function.
        This function sets the maximum value of h0 to the minimum value of h, i.e. stage.
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
            The lmfit Parameters with limits set for max h0.
        """
        del q

        h_min = float(np.min(h))
        h_max = float(np.max(h))
        eps = 1e-10

        if self.segments == 1:
            h0_ceiling = min(h_min - eps, self.parameters["h0"].max)
            self.parameters["h0"].max = h0_ceiling
            return self.parameters

        self.parameters["h0"].max = min(h_min - eps, self.parameters["h0"].max)

        # FIXME
        # TODO need to test properly different scenarios
        # use expressions?
        # of this, right not this is not production ready
        # breakpoint values should also be regularized, and min-max range being dynamic and not fixed
        # lmfit allow for this I think
        # when setting breakpoint values we need to take into account
        # that some can be set to initial values/min/max etc and others not

        break_names = [f"break{idx}" for idx in range(1, self.segments)]
        break_values = np.array(
            [self.parameters[name].value for name in break_names], dtype=float
        )
        if (
            not np.all(np.isfinite(break_values))
            or np.any(break_values <= h_min)
            or np.any(break_values >= h_max)
            or np.any(np.diff(break_values) <= 0)
        ):
            default_break_values = np.linspace(h_min, h_max, self.segments + 1)[1:-1]
            for name, value in zip(break_names, default_break_values, strict=True):
                self.parameters[name].value = float(value)

        for idx in range(1, self.segments):
            break_param = self.parameters[f"break{idx}"]
            break_param.min = max(break_param.min, h_min + eps)
            break_param.max = min(break_param.max, h_max - eps)
            self.parameters[f"c{idx + 1}"].max = min(
                self.parameters[f"c{idx + 1}"].max,
                break_param.value - eps,
            )

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
        if self.segments != 1:
            raise NotImplementedError(
                "PowerLaw.inverse() is only implemented for one segment."
            )

        values = self._parameter_values(params)
        return (np.asarray(q) / values["a"]) ** (1 / values["b"]) + values["h0"]

    def _inverse_segmented_powerlaw(self, q: np.ndarray, params: dict[str, float]):
        """Use the same inversion method as the single segment power law, but with the parameters of the corresponding segment."""
        raise NotImplementedError(
            "PowerLaw._inverse_segmented_powerlaw() is not implemented."
        )

    def derived_parameters(self, params: dict[str, float]) -> dict[str, float]:
        if self.segments == 1:
            return {}

        values = self._parameter_values(params)
        scales = self._scale_parameters(values)
        return {f"a{idx}": scales[f"a{idx}"] for idx in range(2, self.segments + 1)}

    def _default_parameters(self) -> Parameters:
        parameters = Parameters()
        eps = 1e-10
        if self.segments == 1:
            parameters.add("a", value=1.0)
            parameters.add("h0", value=0.0)
            parameters.add("b", value=2.0)

            parameters["a"].min = eps
            parameters["b"].min = eps

            return parameters

        parameters.add("a1", value=1.0)
        parameters.add("h0", value=0.0)
        parameters.add("b1", value=2.0)
        parameters["a1"].min = eps
        parameters["b1"].min = eps

        for idx in range(1, self.segments):
            parameters.add(f"break{idx}", value=float(idx))
            parameters.add(f"c{idx + 1}", value=0.0)
            parameters.add(f"b{idx + 1}", value=2.0)
            parameters[f"b{idx+1}"].min = eps # note that a2.._idx are derived parameters

        return parameters

    @staticmethod
    def _power_law(
        h: float | np.ndarray, *, a: float, h0: float, b: float
    ) -> np.ndarray:
        return a * (np.asarray(h) - h0) ** b

    def _func_for_lmfit(
        self, h: np.ndarray, a: float = 1.0, h0: float = 0.0, b: float = 2.0
    ) -> np.ndarray:
        """The equation/function that lmfit will fit."""
        return self._power_law(h, a=a, h0=h0, b=b)

    def _create_lmfit_function(self):
        param_names = list(self.parameters.keys())

        def func(h, **params):
            return self.func(h, **params)

        func.__name__ = "power_law"
        # pyrefly: ignore [missing-attribute]
        func.argnames = ["h", *param_names]
        # pyrefly: ignore [missing-attribute]
        func.kwargs = [(name, self.parameters[name].value) for name in param_names]
        return func

    def _parameter_values(self, overrides: dict[str, float]) -> dict[str, float]:
        unknown = set(overrides) - set(self.parameters)
        if unknown:
            raise KeyError(f"Unknown PowerLaw parameter(s): {sorted(unknown)}")
        values = {
            name: float(parameter.value) for name, parameter in self.parameters.items()
        }
        values.update(overrides)
        return values

    def _segmented_power_law(
        self, h: np.ndarray | float, params: dict[str, float]
    ) -> np.ndarray:
        h_values = np.asarray(h, dtype=float)
        scalar_input = h_values.ndim == 0
        h_array = np.atleast_1d(h_values)

        self._validate_segment_parameters(params, h_array)
        scales = self._scale_parameters(params)

        discharge = np.empty_like(h_array, dtype=float)
        breaks = self._break_values(params)

        for idx in range(1, self.segments + 1):
            if idx == 1:
                mask = h_array <= breaks[0]
            elif idx == self.segments:
                mask = h_array > breaks[-1]
            else:
                mask = (h_array > breaks[idx - 2]) & (h_array <= breaks[idx - 1])

            # h0 for first segment, c{idx} for others
            h_offset_key = "h0" if idx == 1 else f"c{idx}"
            discharge[mask] = self._power_law(
                h_array[mask],
                a=scales[f"a{idx}"],
                h0=params[h_offset_key],
                b=params[f"b{idx}"],
            )

        if scalar_input:
            return discharge[0]
        return discharge.reshape(h_values.shape)

    def _scale_parameters(self, params: dict[str, float]) -> dict[str, float]:
        self._validate_breakpoints(params)

        scales = {"a1": params["a1"]}
        for idx in range(2, self.segments + 1):
            boundary = params[f"break{idx - 1}"]
            current_offset_key = f"c{idx}"
            previous_offset_key = "h0" if idx - 1 == 1 else f"c{idx - 1}"
            if params[current_offset_key] >= boundary:
                raise ValueError(
                    f"{current_offset_key} must be below the lower stage boundary "
                    f"for segment {idx}."
                )
            previous_discharge = self._power_law(
                boundary,
                a=scales[f"a{idx - 1}"],
                h0=params[previous_offset_key],
                b=params[f"b{idx - 1}"],
            )
            denominator = (boundary - params[current_offset_key]) ** params[f"b{idx}"]
            scales[f"a{idx}"] = float(previous_discharge / denominator)

        return scales

    def _validate_segment_parameters(
        self, params: dict[str, float], h: np.ndarray
    ) -> None:
        self._validate_breakpoints(params)

        h_min = float(np.min(h))
        if params["h0"] >= h_min:
            raise ValueError("h0 must be below the lower stage boundary for segment 1.")

        for idx in range(2, self.segments + 1):
            lower_boundary = params[f"break{idx - 1}"]
            if params[f"c{idx}"] >= lower_boundary:
                raise ValueError(
                    f"c{idx} must be below the lower stage boundary for segment {idx}."
                )

    def _validate_breakpoints(self, params: dict[str, float]) -> None:
        breaks = self._break_values(params)
        if not np.all(np.diff(breaks) > 0):
            raise ValueError("PowerLaw breakpoints must be strictly increasing.")

    def _break_values(self, params: dict[str, float]) -> np.ndarray:
        return np.array(
            [params[f"break{idx}"] for idx in range(1, self.segments)],
            dtype=float,
        )
