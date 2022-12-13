# -*- coding: utf-8 -*-
"""
Tests for PowerLaw rating model
"""

import pytest

from numpy.random import SeedSequence, default_rng
import pandas as pd
from lmfit import Parameters

from hydrating import RatingCurve
from hydrating.models import PowerLaw 

rng = default_rng(SeedSequence(12))

# Test data for power law rating
a = 0.1
b = 2.5
h0 = 0.65

stage_c = rng.uniform(1,10,size=20)
discharge_c = PowerLaw().func(stage_c, a, h0, b)

stage_n = stage_c + rng.normal(0, 0.01, size=20) # add some noise
discharge_n = discharge_c + discharge_c * rng.normal(0, 0.1, size=20) # add some noise

df_c = pd.DataFrame({'stage_key': stage_c,
                     'discharge_key': discharge_c})
df_n = pd.DataFrame({'stage_key': stage_n,
                     'discharge_key': discharge_n})

partial_par_dict = {'b': 2}
full_par_dict = {'a': 1, 'b': 2, 'h0': 0}

lmfit_partial_par = Parameters()
lmfit_partial_par.add('b', value=2)

lmfit_full_par = Parameters()
lmfit_full_par.add('a', value=1)
lmfit_full_par.add('b', value=2)
lmfit_full_par.add('h0', value=0)

# passing no pars
# passing lmfit Parmeters
# passing dict parameters
@pytest.mark.parametrize("parameters, stage_data, discharge_data, test_par_vals", [
    (None, stage_c, discharge_c, True), # 
    (None, stage_n, discharge_n, False),
    (lmfit_partial_par, stage_c, discharge_c, True),
    (lmfit_partial_par, stage_n, discharge_n, False),
    (lmfit_full_par, stage_c, discharge_c, True),
    (lmfit_full_par, stage_n, discharge_n, False),
    (partial_par_dict, stage_c, discharge_c, True),
    (partial_par_dict, stage_n, discharge_n, False),
    (full_par_dict, stage_c, discharge_c, True),
    (full_par_dict, stage_n, discharge_n, False),
    
])
def test_powerlaw_fit(parameters, stage_data, discharge_data, test_par_vals):
    rc_c = RatingCurve(PowerLaw, initial_parameters=parameters)
    rc_c.fit(stage_data, discharge_data)

    assert rc_c.fit_result.success
    if test_par_vals:
        assert pytest.approx(rc_c.fit_result.best_values) == {'a': a, 'b': b, 'h0': h0}


# Test passing a dataframe with data, and passing None or str to fit call
@pytest.mark.parametrize("parameters, stage_data, discharge_data, test_par_vals, data_df", [
    (None, 'stage_key', 'discharge_key', True, df_c),  #
    (None, 'stage_key', 'discharge_key', False, df_n),
    (lmfit_partial_par, 'stage_key', 'discharge_key', True, df_c),
    (lmfit_partial_par, 'stage_key', 'discharge_key', False, df_n),
    (lmfit_full_par, 'stage_key', 'discharge_key', True, df_c),
    (lmfit_full_par, 'stage_key', 'discharge_key', False, df_n),
    (partial_par_dict, 'stage_key', 'discharge_key', True, df_c),
    (partial_par_dict, 'stage_key', 'discharge_key', False, df_n),
    (full_par_dict, 'stage_key', 'discharge_key', True, df_c),
    (full_par_dict, 'stage_key', 'discharge_key', False, df_n),

])
def test_powerlaw_fit_df(parameters, stage_data, discharge_data, test_par_vals, data_df):
    rc_c = RatingCurve(PowerLaw, initial_parameters=parameters)
    rc_c.add_data(data_df, stage=stage_data, discharge=discharge_data)
    rc_c.fit(stage_data, discharge_data)

    assert rc_c.fit_result.success
    if test_par_vals:
        assert pytest.approx(rc_c.fit_result.best_values) == {'a': a, 'b': b, 'h0': h0}


# test fixing parameter (by changing .initial_parameters)
# test bounds on parameters  (by changing .initial_parameters)
# same by providing directly parameters that are fixed/bounded
# Test fixing a lmfit parameter
@pytest.mark.parametrize("parameters_raw, stage_data, discharge_data, test_par_vals", [
    (lmfit_partial_par, stage_c, discharge_c, True),
    (lmfit_partial_par, stage_n, discharge_n, False),
    (lmfit_full_par, stage_c, discharge_c, True),
    (lmfit_full_par, stage_n, discharge_n, False),
])
def test_powerlaw_fixed_input_par(parameters_raw, stage_data, discharge_data, test_par_vals):
    parameters = parameters_raw.copy()
    parameters['b'].value = 2.6
    parameters['b'].vary = False

    rc_c = RatingCurve(PowerLaw, initial_parameters=parameters)
    rc_c.fit(stage_data, discharge_data)

    assert rc_c.fit_result.success
    if test_par_vals:
        assert pytest.approx(rc_c.fit_result.best_values) == {'a': 0.07546474311331504,
                                                              'b': 2.6,
                                                              'h0': 0.4257742480337223}
    else:
        assert pytest.approx(rc_c.fit_result.best_values['b']) == 2.6

# def test_powerlaw_bounded_input_par # TODO
# def test_powerlaw_fixed_init_par # TODO
# def test_powerlaw_bounded_init_par # TODO


# passing kwargs to lmfit.Model.fit
def test_lmfit_kwargs():
    ...

# test setting the stage series and predicting with timeseries to get a Q timeseries
def test_setting_stage_series_and_predict_q():
    ...

