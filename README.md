# hydrating

[![pypi_shield](https://img.shields.io/pypi/v/hydrating.svg)](https://pypi.org/project/hydrating/)
[![pypi_license](https://badgen.net/pypi/license/hydrating/)](https://pypi.org/project/hydrating/)
![tests_workflow](https://github.com/rhkarls/hydrating/actions/workflows/run_flake8_pytest.yml/badge.svg)

## Overview
hydrating is a python package for fitting hydrological rating curves. 

hydrating aims to provide an easy interface to fitting rating curve for any 
kind of model while giving the user full control on the model parameters 
(provided through lmfit library). hydrating also aims at providing robust methods 
for estimating rating curve uncertainty.

This initial preview only provides basic powerlaw model fitting, see example below. 

Development Status: Pre-Alpha. 

**Due to personal obligations, development is mostly on pause until second half of 2023.**

Consider the API unstable, it may change at short/no notice.

## Requirements and installation

Requirements:

    numpy
	pandas
    lmfit
    matplotlib

Install from pypi using pip

    pip install hydrating

## General description and example usage
Functionality is currently limited to fitting basic powerlaw rating curve


```python
# imports for creating demo data
from numpy.random import SeedSequence, default_rng
import pandas as pd
from lmfit import Parameters

# imports for hydrating
from hydrating import RatingCurve
from hydrating.models import PowerLaw 

# Test data for power law rating
a = 0.1
b = 2.5
h0 = 0.65

rng = default_rng(SeedSequence(123))

stage = rng.uniform(1, 10, size=20) #+ rng.normal(0, 0.01, size=20) # uncomment for stage with some noise
discharge_c = PowerLaw().func(stage, a, h0, b)
discharge = discharge_c #+  discharge_c * rng.normal(0, 0.1, size=20) # uncomment for discharge with some noise

data_df = pd.DataFrame({'stage_key': stage,
                        'discharge_key': discharge})

# fit by passing numpy arrays of the data
rc_c = RatingCurve(PowerLaw)
rc_c.fit(x=stage, y=discharge)
rc_c.fit_summary() # for a fit summary
rc_c.fit_best_parameters
Out: {'a': 0.09999999999999991, 'h0': 0.6499999999999992, 'b': 2.5000000000000004}

# fit by adding a dataframe. In future versions this will allow for more options adding 
# other metadata to the rating curve
rc_c = RatingCurve(PowerLaw)
rc_c.add_data(data_df, stage='stage_key', discharge='discharge_key')
rc_c.fit() # will automatically use keys provided in add_data, but can also pass other keys here

# specify parameter bounds and fixed values, using lmfit.Parameters objects
input_parameters = Parameters()
input_parameters.add('a', value=1)
input_parameters.add('b', value=2.6, vary=False)
input_parameters.add('h0', value=0, max=0.4)
rc_c = RatingCurve(PowerLaw, initial_parameters=input_parameters)
rc_c.fit(x=stage, y=discharge)
Out: {'a': 0.07495641841621854, 'b': 2.6, 'h0': 0.3999999978452029}

# one can also let hydrating create the parameters simpy by ommiting
# initial_parameters keyword, and then modify them
# refer to lmfit documentation
rc_c = RatingCurve(PowerLaw)
rc_c.initial_parameters['b'].value = 2.55
rc_c.initial_parameters['b'].vary = False
rc_c.fit(x=stage, y=discharge)
rc_c.fit_best_parameters
Out: {'a': 0.08681743671415193, 'h0': 0.5320384707736181, 'b': 2.55}

```

## Feedback and issues

Please report issues here: https://github.com/rhkarls/hydrating/issues

General feedback is most welcome, please post that as well under issues.

