# -*- coding: utf-8 -*-
"""
RatingCurve class
"""

import pandas as pd
import lmfit
import numpy as np
import matplotlib.pyplot as plt

from typing import Union, Optional

from ..models import RatingModel
# from .grades import Grades


class RatingCurve:
    
    def __init__(self, 
                 model: RatingModel,
                 initial_parameters: dict = None, #or Parameters
                 rating_name: str = '' # optional  NOT A KEY)
                ):
        
        self.rating_name = rating_name
        self.rating_model = model # instance is crated in the setter
        self.initial_parameters = initial_parameters # done by model setter
        
        # self.rc_grade = None # grade stage intervals with Enum grades
        # def set_grades():
        #     ...# @setter/getter properties?
        
        # self.rc_limit = None # tuple, min, max stage its applicable
        # self.rc_period = None # tuple (datetime-like to from validity)
        
        self.fit_result = None
        
        self.stage_series = None # timeseries of stage, can be used to show extrapolation
                                 # of the rating curve, when comparing rating curves
        
        self._dataf = None # user can add a dataframe using add_data, but this is strictly not needed
                          # can also just call .fit() with two numpy arrays
        self._user_df_added = False
        
        # def set_limits():
        #     ...  #@setter/getter properties?
        # def set_valid_periods():
        #     ...  #@setter/getter properties?
        
    # or use property??
    # rc.model = Callable
    # also for parameter
    # rc.parameter = dict # overwrites the default
    @property
    def rating_model(self):
        return self._rating_model
    
    @rating_model.setter
    def rating_model(self,
              model: RatingModel):
        self._rating_model = model()

    @property
    def initial_parameters(self):
        return self._initial_parameters
    
    @initial_parameters.setter
    def initial_parameters(self, params):
        model = self.rating_model.create_model()
        if isinstance(params, dict):
            self._initial_parameters = self.rating_model.create_parameters(model, params)
        elif isinstance(params, lmfit.Parameters):
            self._initial_parameters = params
        elif params is None:
            self._initial_parameters = self.rating_model.create_parameters(model) 
        else:
            raise ValueError()

        # Check the parameters

    # @property
    # def grade(self):
    #     return self._grade
    
    # @grade.setter
    # def grade(self, value):
    #     # [{'from': 1.5,
    #     #  'to': 2.5,
    #     #  'grade': Grades.Good},] ?
    #     self._grade = value
    
    def add_data(self,
                 data: pd.DataFrame,
                 stage: str,
                 discharge: str,
                 datetime_start: Optional[str] = None, # optional, key in data df
                 datetime_end: Optional[str] = None,  # optional, key in data df
                 method: Optional[str] = None,  # optional, key in data df
                 party: Optional[str] = None,  # optional, key in data df
                 agency: Optional[str] = None,  # optional, key in data df
                 uncertainty: Optional[str] = None,  # optional, key in data df
                 grade: Optional[str] = None,  # optional, key in data df
                 enabled: Optional[str] = None,  # optional, key in data df
                 note: Optional[str] = None,  # optional, key in data df
                 identifier: Optional[str] = None):  # optional, key in data df
    
        
        self._k_discharge = discharge
        self._k_stage = stage
        self._k_datetime_start = datetime_start
        self._k_datetime_end = datetime_end
        self._k_method = method
        self._k_party = party
        self._k_agency = agency
        self._k_uncertainty = uncertainty
        self._k_grade = grade
        self._k_enabled = enabled
        self._k_note = note
        self._k_identifer = identifier
        
        # take the data DataFrame and make a copy
        # keep all columns of DataFrame
        # create a new columns "enabled" for True/False flag to turn on off
        self._dataf = data.copy()
        if self._k_enabled is None:
            self._k_enabled = "enabled"
            self._dataf["enabled"] = True
            
        self._user_df_added = True
        
    # TODO use a setter/getter for dataf in addition to this function to add_data?
    # better to hide the self.dataf
            
    def _user_data_from_fit(self, x, y):
        """ If data is passed to fit() create _dataf if the user 
        has not done so themselves by calling add_data()"""
        
        self._k_stage = 'stage'
        self._k_discharge = 'discharge'
        self._dataf = pd.DataFrame(data={self._k_stage: x,
                                         self._k_discharge: y})
        

    # fit uses the model and parameters
    def fit(self,
            x: pd.Series=None, # FIXME np.ndarray, pd.Series or str for key in dataf
            y: pd.Series=None,
            engine: str = 'lmfit',
            weights: pd.Series = None,
            **kwargs):
        
            # Note: cannot use self. in default arguments as they are eval'ed at 
            # creation time
            if x is None:
                # try:
                xf = self._dataf[self._k_stage].to_numpy()
                # except: # TODO what error is this if dataf is None?
                    # ... # message to add data or pass data to fit
            elif isinstance(x, str):
                xf = self._dataf[x].to_numpy()
            else: # is np.ndarray or Series, check for that?
                xf = x.copy() 
                
            if y is None:
                yf = self._dataf[self._k_discharge].to_numpy()
            elif isinstance(y, str):
                yf = self._dataf[y].to_numpy()
            else: # is np.ndarray or Series, check for that?
                yf = y.copy() 
                    
            if not self._user_df_added:
                self._user_data_from_fit(xf, yf)
                
                
            if engine != 'lmfit':
                raise NotImplementedError(f"{engine} is not supported, only lmfit is "
                                           "currently supported")
            
            # lmfit engine
            lmfit_model = self.rating_model.create_model()
            lmfit_init_pars = self.initial_parameters
            lmfit_weights = np.ones(len(yf)) if weights is None else weights.to_numpy()
            
            # constrain parameters as defined in RatingModel
            lmfit_init_pars = self.rating_model.constrain_pars_with_obs(xf,
                                                                        yf,
                                                                        lmfit_init_pars)
            
            result = lmfit_model.fit(data=yf,
                                     params=lmfit_init_pars,
                                     weights=lmfit_weights,
                                     h=xf,
                                     **kwargs) # test with using kwargs = {'method':'differential_evolution'}
            
            self._fit_obs_data = pd.DataFrame(data=yf, index=xf, columns=['observed'])
            
            self.fit_result = result
            self.fit_best_parameters = result.best_values
            
            self._calc_fit_residuals()
            
    def _calc_fit_residuals(self):
        self._fit_obs_data['predicted'] = self.predict(stage=self._fit_obs_data.index)
        self._fit_obs_data['residual'] =  (self._fit_obs_data['predicted'] 
                                           - self._fit_obs_data['observed'])
        self._fit_obs_data['percent_error'] = (self._fit_obs_data['residual'] 
                                               / self._fit_obs_data['observed'] * 100)
        
        
    def add_stage_series(self, stage_series):
        self.stage_series = stage_series.copy()
        
    def predict(self,
                stage=None):
        if stage is None and self.stage_series is not None:
            return self.rating_model.func(self.stage_series, **self.fit_best_parameters)
        if stage is not None:
            return self.rating_model.func(stage, **self.fit_best_parameters)
        return None
    
    def inverse(self,
                discharge,
                initial_guess=None):
        
        # if model has inverse attribute, use that, else inverse it with minimizing
        # this requires an initial guess
        self.rating_model.inverse(discharge, initial_guess, **self.fit_result.best_values)
    
    def fit_summary(self):
        print(self.fit_result.fit_report())
    
    def plot_residuals(self, 
             scale='linear',
             label_discharge='Discharge',
             label_stage='Stage',
             stage_on_y=False,
             labels=None):
        
        if self.fit_result is None:
            ValueError("Rating curve is not fitted, call .fit() first.")# TODO custom exception
        
        if self.stage_series is not None:
            min_stage = self.stage_series.min()
            max_stage = self.stage_series.max()
        else:
            min_stage = self._fit_obs_data.index.min()
            max_stage = self._fit_obs_data.index.max()
        
        stage_plt = np.linspace(min_stage, max_stage, num=100)
        q_plt = self.predict(stage=stage_plt)
        
        fig, axes = plt.subplots(3, 1, sharex=True)
                
        if stage_on_y:
            y_plt = stage_plt
            x_plt = q_plt
            
            y_pts = self._fit_obs_data.index.to_numpy()
            x_pts = self._fit_obs_data['observed']
            
            y_label = label_stage
            x_label = label_discharge
        else:
            y_plt = q_plt
            x_plt = stage_plt
            
            y_pts = self._fit_obs_data['observed']
            x_pts = self._fit_obs_data.index.to_numpy()
            
            x_label = label_stage
            y_label = label_discharge

        axes[0].plot(x_pts, y_pts, 'ko')
        axes[0].plot(x_plt, y_plt, 'k-')
        
        axes[1].axhline(0, color='k', linewidth=0.5)
        axes[1].plot(x_pts, self._fit_obs_data['residual'], 'ko')
        
        axes[2].axhline(0, color='k', linewidth=0.5)
        axes[2].plot(x_pts, self._fit_obs_data['percent_error'], 'ko')
        
        axes[0].set_yscale(scale)
        axes[2].set_xscale(scale)
        
        axes[0].set_ylabel(y_label)
        axes[1].set_ylabel('Absolute error')
        axes[2].set_ylabel('Percent error')
        axes[2].set_xlabel(x_label)
        
        axes[1].set_ylim((-max(np.abs(axes[1].get_ylim())), max(np.abs(axes[1].get_ylim()))))
        axes[2].set_ylim((-max(np.abs(axes[2].get_ylim())), max(np.abs(axes[2].get_ylim()))))
            
        return fig, axes
        
    def plot(self, 
             scale='linear',
             label_discharge='Discharge',
             label_stage='Stage',
             stage_on_y=False,
             labels=None,
             cross_section=False):
        
        if self.fit_result is None:
            ValueError("Rating curve is not fitted, call .fit() first.") # TODO custom exception
        
        if self.stage_series is not None:
            min_stage = self.stage_series.min()
            max_stage = self.stage_series.max()
        else:
            min_stage = self._fit_obs_data.index.min()
            max_stage = self._fit_obs_data.index.max()

        stage_plt = np.linspace(min_stage, max_stage, num=100)
        q_plt = self.predict(stage=stage_plt)
        
        fig, ax = plt.subplots()
          
        if cross_section:
            stage_on_y=True
            
        if stage_on_y:
            y_plt = stage_plt
            x_plt = q_plt
            
            y_pts = self._fit_obs_data.index.to_numpy()
            x_pts = self._fit_obs_data['observed']
            
            y_label = label_stage
            x_label = label_discharge
        else:
            y_plt = q_plt
            x_plt = stage_plt
            
            y_pts = self._fit_obs_data['observed']
            x_pts = self._fit_obs_data.index.to_numpy()
            
            x_label = label_stage
            y_label = label_discharge

        ax.plot(x_pts, y_pts, 'ko')
        ax.plot(x_plt, y_plt, 'k-')
        
        ax.set_yscale(scale)
        ax.set_xscale(scale)
        
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
            
        return fig, ax

    # TODO
    # def plot_shift():
    #     ...
        

    # TODO
    # def compare(self, other):
    #     ...
    
    # TODO
    # def add_obs_point(self):
    #     ...


    @staticmethod
    def _dict_from_parameters(parameters):
        return parameters.valuesdict()