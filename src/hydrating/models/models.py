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
           initial_guess: float,
           a: float,
           h0: float,
           b: float) -> np.ndarray:
        
        # todo see whats best, pass parameters dict or individual?
        # a = parameters['a']
        # b = parameters['b']
        # h0 = parameters['h0']
        
        return (y/a)**(1/b) - h0 # FIXME NOT TESTED WITH h0

