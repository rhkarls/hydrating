# -*- coding: utf-8 -*-
"""
Rating curve models.
"""

from typing import Protocol, Union

import numpy as np
from lmfit import Model, Parameters


class RatingModel(Protocol):
    """
    Protocol class for rating curve models.

    Methods for RatingModel:

    func: the rating curve equation, with arguments stage and **parameters, returning
          discharge
    create_model: function that returns the lmfit Model
    create_parameters: function that return lmfit Parameters for the model
    constrain_pars_with_obs: function that puts limits on Parameters based on observations
                             e.g. zero flow stage cannot exceed observed stage with flow
    inverse: function that return stage for a given discharge, the inverse of func()
    """

    def func(self, x: np.ndarray, **parameters: float) -> np.ndarray:
        """
        The rating curve equation.

        Parameters
        ----------
        x : np.ndarray
            Stage values.
        **parameters : float
            Model parameters.

        Returns
        -------
        float or np.ndarray
            Discharge for the provided stage.
        """
        ...

    def create_model(self) -> Model:
        """
        Create the lmfit Model.

        Returns
        -------
        Model
            The lmfit Model for the rating curve.
        """
        ...

    def create_parameters(
        self, model: Model, parameters: Union[None, Parameters, dict]
    ) -> Parameters:
        """
        Create the lmfit Parameters for the rating curve model, and set initial values.
        This function should provide some default values if parameters is None or a parameter is missing.

        Parameters
        ----------
        model : Model
            The lmfit Model for the rating curve.
        parameters : Union[None, Parameters, dict]
            Initial parameters for the model. Can be None, a dict, or lmfit Parameters.
        """
        ...

    def constrain_pars_with_obs(
        self, x: np.ndarray, y: np.ndarray, parameters: Parameters
    ) -> Parameters:
        """
        Constraining on the parameters based on x and y values.
        This function should put limits on Parameters based on observations.
        For example, zero flow stage cannot exceed observed stage with flow.

        Parameters
        ----------
        x : np.ndarray
            Stage values.
        y : np.ndarray
            Discharge values.
        parameters : Parameters
            The lmfit Parameters for the model.

        Returns
        -------
        Parameters
            The lmfit Parameters with limits set based on observations.
        """
        ...

    def inverse(
        self, y: np.ndarray, initial_guess: float, **parameters: float
    ) -> np.ndarray:
        """
        Inverse the rating curve to find the stage at a given discharge.
        This method should return the stage corresponding to the given discharge.

        Parameters
        ----------
        y : np.ndarray
            Discharge values.
        initial_guess : float
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

    @classmethod
    def func(cls, h: np.ndarray, a: float, h0: float, b: float) -> np.ndarray:
        """
        The power law rating curve function.

        .. math::
            Q(h) = a \\times (h-h0)^b

        where :math:`Q` is discharge, :math:`h` is stage, :math:`h0` is stage at
        zero flow, and :math:`a` and :math:`b` are fitted parameters.

        Parameters
        ----------
        h : float or np.ndarray
            Stage.
        a : float
            Parameter a.
        h0 : float
            Stage at zero flow.
        b : float
            Parameter b.

        Returns
        -------
        float or np.ndarray
            Discharge for the provided stage.
        """
        return a * (h - h0) ** b

    @classmethod
    def create_model(cls):
        """
        Create the lmfit Model.

        Returns
        -------
        Model
            The lmfit Model for the power law rating curve.
        """
        return Model(cls.func)

    @classmethod
    def create_parameters(cls, model, parameters=None):
        """
        Create the lmfit Parameters for the power law rating curve model.
        This function sets default values for the parameters if they are missing.

        Parameters
        ----------
        model : Model
            The lmfit Model for the rating curve.
        parameters : Union[None, Parameters, dict]
            Initial parameters for the model. Can be None, a dict, or lmfit Parameters.

        Returns
        -------
        Parameters
            The lmfit Parameters for the model, with default values set for missing parameters.
        """
        # default parameters, used if missing from parameters argument
        params_default = model.make_params()
        params_default["h0"].value = 0
        params_default["b"].value = 2
        params_default["a"].value = 1
        if parameters is None:
            return params_default

        # for dict and Parameters add the missing pars, if any
        for k in set(params_default.keys()) - set(parameters.keys()):
            parameters[k] = params_default.copy()[k]
        if isinstance(parameters, Parameters):
            return parameters
        if isinstance(parameters, dict):
            return model.make_params(**parameters)

    @classmethod
    def constrain_pars_with_obs(cls, x, y, parameters):
        """
        Constraining on the parameters based on observed values.
        This function sets the maximum value of h0 to the minimum value of x, i.e. stage.
        This is to avoid fitting a rating curve with zero flow stage that is higher than the observed stage with flow, which is not physically possible.
        Avoid using this constrain if the observed stage goes below zero flow (i.e. flow is zero in the timeseries).

        Parameters
        ----------
        x : np.ndarray
            Stage values.
        y : np.ndarray
            Discharge values. Not used in this function.
        parameters : Parameters
            The lmfit Parameters for the model.

        Returns
        -------
        Parameters
            The lmfit Parameters with limits set for max h0.
        """
        # check if h0 was already set to a max value that is lower than the
        # limit based on observations
        h0_ceiling = min(np.min(x) - 1e-10, parameters["h0"].max)
        parameters["h0"].max = h0_ceiling

        return parameters

    @classmethod
    def inverse(
        cls, y: np.ndarray, initial_guess: float, a: float, h0: float, b: float
    ) -> np.ndarray:
        """
        Inverse the rating curve to find the stage at a given discharge.

        Parameters
        ----------
        y : np.ndarray
            Discharge values.
        initial_guess : float
            Initial guess not used in this function.
        a : float
            Parameter a.
        h0 : float
            Parameter h0.
        b : float
            Parameter b.

        Returns
        -------
        np.ndarray
            Stage values corresponding to the given discharge.
        """

        # todo see what's best, pass parameters dict or individual?
        # a = parameters['a']
        # b = parameters['b']
        # h0 = parameters['h0']

        return (y / a) ** (1 / b) - h0  # FIXME NOT TESTED WITH h0
