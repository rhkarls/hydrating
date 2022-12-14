# -*- coding: utf-8 -*-
"""
Rating curve models
"""

from typing import Protocol, Union

import numpy as np
from lmfit import Model, Parameters


class RatingModel(Protocol):
    """ Protocol class for rating curve models
    
    Methods for RatingModel:
    
    func: the rating curve equation, with arguments stage and **parameters, returning
          discharge
    create_model: function that returns the lmfit Model
    create_parameters: function that return lmfit Parameters for the model
    constrain_pars_with_obs: function that puts limits on Parameters based on observations
                             e.g. zero flow stage cannot exceed observed stage with flow
    inverse: function that return stage for a given discharge, the inverse of func()
    """
    def func(self,
             x: np.ndarray,
             **parameters: float) -> np.ndarray:
        ...
        
    def create_model(self) -> Model:
        ...
        
    def create_parameters(self,
                          model:Model, 
                          parameters:Union[None, Parameters, dict]) -> Parameters:
        ...

    def constrain_pars_with_obs(self,
                             x: np.ndarray,
                             y: np.ndarray,
                             parameters: Parameters) -> Parameters:
        ...

    def inverse(self,
           y: np.ndarray,
           initial_guess: float,
           **parameters: float
           ) -> np.ndarray:
        ...


class PowerLaw:
    @classmethod 
    def func(cls,
             h: np.ndarray,
             a: float,
             h0: float,
             b: float) -> np.ndarray:
        """
        The power law rating curve, where
        
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
            Discharge for provided stage.

        """
        return a * (h-h0)**b

    @classmethod 
    def create_model(cls):
        return Model(cls.func)

    @classmethod 
    def create_parameters(cls, model, parameters=None):
        # default parameters, used if missing from parameters argument
        params_default = model.make_params()
        params_default['h0'].value = 0
        params_default['b'].value = 2
        params_default['a'].value = 1
        if parameters is None:
            return params_default

        # for dict and Parameters add the missing pars, if any
        for k in (set(params_default.keys()) - set(parameters.keys())):
            parameters[k] = params_default.copy()[k]
        if isinstance(parameters, Parameters):
             return parameters
        if isinstance(parameters, dict):
            return model.make_params(**parameters)

    @classmethod     
    def constrain_pars_with_obs(cls, x, y, parameters):
        """ Set constrains on the parameters based on x and y values """
        # check if h0 was already set to a max value that is lower than the
        # limit based on observations
        h0_ceiling = min(np.min(x)-1e-10, parameters['h0'].max)
        parameters['h0'].max = h0_ceiling
        
        return parameters

    @classmethod     
    def inverse(cls,
           y: np.ndarray,
           initial_guess: float, # not used for this Model
           a: float,
           h0: float,
           b: float) -> np.ndarray:
        
        # todo see whats best, pass parameters dict or individual?
        # a = parameters['a']
        # b = parameters['b']
        # h0 = parameters['h0']
        
        return (y/a)**(1/b) - h0 # FIXME NOT TESTED WITH h0


class VNotchWeir:
    def func(self,
             h: np.ndarray,
             angle: int,
             cd: float,
             k: float,
             beta: float = 2.5,
             g: float = 9.81) -> np.ndarray:
        """
        V-notch weir equation for head, h, in meters and discharge
        , Q, in :math:`m^3 s^{-1}`.
        Equation is for fully contracted V-notch weirs

        .. math::
            Q(h) = \\frac{8}{15} \\times C_d \\times \\sqrt{2*g}
            \\times tan(\\frac{\\theta}{2}) \\times (h+k)^{5/2}

        where :math:`C_d` is the discharge coefficient, :math:`g` in the gravity
        acceleration, :math:`\\theta` is the notch angle, :math:`k` is the
        head correction.

        Parameters
        ----------
        h : np.ndarray
            Head in meters above V-notch
        angle : int
            Notch angle in degrees.
        cd : float
            Discharge coefficient for weir, see function calc_cd.
        k : float
            Head correction in meters for weir, see function calc_k_meter.
        beta : float, optional
            Exponent. The default is 2.5.
        g : float, optional
            Gravity acceleration in meters per second. The default is 9.81.

        Returns
        -------
        float
            Discharge in cubic meters per second (m3/s).

        Notes
        -----
        For restrictions of this equation see [1]_

        [1] https://www.usbr.gov/tsc/techreferences/mands/wmm/chap07_07.html

        """

        return 8 / 15 * cd * np.sqrt(2 * g) * np.tan(angle / 2) * (h + k) ** beta

    def create_model(self):
        return Model(self.func)

    def create_parameters(self, model, parameters=None):
        if parameters is None:
            # default parameters for 90 degree weir
            params = model.make_params()
            default_angle = 90
            params['angle'].value = default_angle
            params['cd'] = self.calc_cd(default_angle)
            params['k'] = self.calc_k_meter(default_angle)
            params['beta'].value = 2.5
            params['g'].value = 9.81

        if isinstance(parameters, Parameters):
            return parameters
        if isinstance(parameters, dict):
            params = model.make_params(**parameters)

        params['g'].vary = False
        params['k'].vary = False
        params['angle'].vary = False
        params['beta'].vary = False

        return params

    def constrain_pars_with_obs(self, x, y, parameters):
        """ Set constrains on the parameters based on x and y values """
        return parameters

    def inverse(self,
                y: np.ndarray,
                initial_guess: float,
                **parameters) -> np.ndarray:
        raise NotImplementedError

    # Extra methods for weir class
    @staticmethod
    def calc_cd(angle):
        """Calculate V-notch weir discharge coefficient $C_d$

        Source: https://www.usbr.gov/tsc/techreferences/mands/wmm/ Figure 7-6b
        Data has been digitized and fitted using 3 degree polynomial

        Parameters
        ----------
        angle : int or float
            V-notch angle in degrees. Must be between 20 and 100.

        Returns
        -------
        float
            Discharge coefficient.

        """
        if (angle < 20) or (angle > 100):
            raise ValueError("Angle must be between 20 and 100 degrees, for other angles "
                             "please use manual estimates.")

        return 0.000000022452 * angle**3 + 0.000002075359 * angle**2 - 0.000648097745 * angle + 0.603292230884

    @staticmethod
    def calc_k_meter(angle):
        """Calculate head correction for V-notch weirs in meters.

        Source: https://www.usbr.gov/tsc/techreferences/mands/wmm/ Figure 7-6a
        Data has been digitized and fitted using 3 degree polynomial

        Parameters
        ----------
        angle : int or float
            V-notch angle in degrees..

        Returns
        -------
        float
            Head correction in meters.
        """

        if (angle < 20) or (angle > 100):
            raise ValueError("Angle must be between 20 and 100 degrees, for other angles "
                             "please use manual estimates.")

        return (0.304785
                * (-0.000000014499 * angle**3
                   + 0.000003964572 * angle**2
                   - 0.000377042597 * angle
                   + 0.015025350564))
