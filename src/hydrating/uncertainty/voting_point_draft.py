# Fixed draft to test method using the 3 seg example from m code

# 2022-10-16: matlab ca 50% faster for mcmc sampling, not so much difference
# can probably optimize a bit using numba
# part inside while is_outside_bounds: is called a lot, optmize here (functions -> numba)
# all equations as own function, numba
# logistic L function, same, numba and speed up

# matlab licensed BSD 3 clause.
# to use MIT: https://softwareengineering.stackexchange.com/questions/278126/is-it-legal-to-include-bsd-licensed-code-within-an-mit-licensed-project
# (3 clause puts one more restriction compared to MIT)

from tqdm import tqdm
from voting_point_functions import logistic_likelihood

def power_rating_3seg(h, a1, a2, a3, b1, b2, b3, br1, br2, c1=0, c2=0, c3=0):
    # no transformations on parameters
    # note the m code does not use c1, c2, c3 at all...
    log_Q = np.piecewise(h, [h < br1, (h >=br1) & (h < br2), h >=br2 , np.isnan(h)], 
                    [lambda h:np.log(a1)+b1*np.log((h-c1)),
                     lambda h:np.log(a2)+b2*np.log((h-c2)),
                     lambda h:np.log(a3)+b3*np.log((h-c3)),
                     np.nan])
    
    return log_Q

def continuity_ax_par(a0, b0, c0, bx, cx, br0):
    #ax = (a0*(br0+c0)**b0)/((br0+cx)**bx)
    #where 0 means for section below x
    # no transformation on parameters
    return (a0*(br0+c0)**b0)/((br0+cx)**bx)
    

# % Prior distribution of parameters a1 b1 b2 b3 br1 br2 a2 a3 - specify prior bounds below
# % The specification of the priors is important, particulary in sections
# % where there is a large extrapolation of the rating curve. Priors can be
# % set using standard values (McMillan and Westerberg, 2015) or based on
# % hydraulic knowledge (Ocio et al., 2017).
# Lbound=[3.65 1.57 1.0 1.05 1.17 3.24 50.4 42.4];
# Ubound=[4.46 3.53 2.16 2.36 1.43 3.96 113.3 95.4];

import numpy as np

n_mcmc = 10_000

mean_q = 23.6540

gdata = np.genfromtxt(r"C:\Users\rehu0001\Documents\Python\Various_not_to_git\VotingPointMethod-main\stage_discharge_example.csv", delimiter=',')

gauged_h = gdata[:,0]
gauged_q = gdata[:,1]

lower_bound = {'a1': 3.65,
               'b1': 1.57,
               'b2': 1,
               'b3': 1.05,
               'br1': 1.17,
               'br2': 3.24,
               'a2': 50.4,
               'a3': 42.4}

upper_bound = {'a1': 4.46,
               'b1': 3.53,
               'b2': 2.16,
               'b3': 2.36,
               'br1': 1.43,
               'br2': 3.96,
               'a2': 113.3,
               'a3': 95.4}

# WIP use function to calculate rating (array of stage must be float):
power_rating_3seg(np.array([1.2,2]), **lower_bound)

#l_bound_arr = {k:np.array(v) for k, v in lower_bound.items()}

# HINT: dict is ordered since python 3.7, require anyway 3.9 and above for hydrating
par_keys = lower_bound.keys()
l_bound_arr = np.fromiter(lower_bound.values(), dtype=float)
u_bound_arr = np.fromiter(upper_bound.values(), dtype=float)

# to recreate par dict, needed for passing pars to function as dict
dict(zip(par_keys, l_bound_arr))

n_pars = len(l_bound_arr)

rng = np.random.default_rng()

# Initial covariance estimate
# HINT: matlab code excludes a2 and a3 from this, do we need to?
# will it be faster without?
n_initial_cov = 10_000
theta_init = l_bound_arr[:6] + (u_bound_arr[:6]-l_bound_arr[:6]) * rng.random((n_initial_cov, n_pars-2))
cov_theta = np.cov(theta_init.T) # HINT: np.cov rows are variables, columns single obs of all vars


theta = np.zeros((n_mcmc, n_pars))
theta_old = np.zeros(n_pars)
theta_new = np.zeros(n_pars)
prob = np.zeros(n_mcmc)

# %Initial parameters taken as centre of priors
# FIXME specific to rating function (index and exp)
#theta_old[0]=(np.exp(l_bound_arr[0])+np.exp(u_bound_arr[0]))/2 # FIXME keep it logged
theta_old[:]=(l_bound_arr[:]+u_bound_arr[:])/2
n_accepted = 0 # counter for accepted samples
prob_old = logistic_likelihood(theta_old, gauged_h, gauged_q, mean_q)

cov_recalc_n = int(n_mcmc/10)
  # %%  
#times_outside = 0

for i, _ in tqdm(enumerate(theta), total=n_mcmc): # TODO test speed against range() (Note cannot be parallelised, mcmc is sequencial?)
    # %Recalculate covariance during burn-in period
    #if t < N / 3 & fix(t/10000)==t/10000 # HINT: matlab has non-robust critera for this, will never hit for many N values
        #C=cov(theta(t-9999:t,1:6));
    
    # Burn in definition 1st third
    # no need to calculate cov all the time (heavy), only every cov_recalc_n
    
    # FIXME
    # The model does has problems below at multivariate_normal
    # because the recalculated covariance of the parameters is WORSE than 
    # the initial one (seen by printing print(theta_new_s[0]))
    # with and without the recalculation
    # might have something to do with accepted parameters???
    # or how cov is calculated??
    # try logging theta[0]? something is messed up with exp/log of theta[0]??
    # cov has [0] as linear space, multivariate_ is applied with log spaced
    # change this so that [0] is logged or not logged (try both, performance issue?)
    # NOTE: not possible to get same as matlab, since when doing that the 
    # multivariate sampling fails here (cov and mean not the same...)
    
    if ((i <= n_mcmc/3) & (i > 0)) & (i % cov_recalc_n == 0):# (int(i/cov_recalc_n) == i/cov_recalc_n): # TODO refactor second term (i % cov_recalc_n == 0)
        print(f"Recalc cov at {i}")
        cov_theta = np.cov(theta[i-cov_recalc_n:i,:6].T) # matlab code minor bug, includes current i which is just zeros
        
    #theta_old[0] = np.log(theta_old[0]) # theta[0] now kept as logged

    is_outside_bounds = True  
    while is_outside_bounds:
        #times_outside += 1
        # sample from multivariate normal distribution
        theta_new_s = rng.multivariate_normal(theta_old[:6], cov_theta, method='cholesky')
        #theta_new_s = np.random.multivariate_normal(theta_old[:6], cov_theta)
        #print(theta_new_s[1])
        theta_new = np.append(theta_new_s,[0,0]) # see if faster to append or just pass all of theta to multivariate_normal
        theta_new[6]=(np.exp(theta_new[0])*theta_new[4]**theta_new[1])/(theta_new[4]**theta_new[2]) #; %Estimation of a2 by continuity
        theta_new[7]=(theta_new[6]*theta_new[5]**theta_new[2])/(theta_new[5]**theta_new[3]) #; %Estimation of a3 by continuity
        is_outside_bounds = False
        # %Check if new sample is within prior bounds
        # If out of bounds set out to True
        # TODO refactor using any()
        # for j, _ in enumerate(theta_new):
        #     if (theta_new[j] > u_bound_arr[j]) | (theta_new[j] < l_bound_arr[j]):
        #         is_outside_bounds = True
        #         times_outside += 1
        
        if (any(theta_new > u_bound_arr) | any(theta_new < l_bound_arr)):
            is_outside_bounds = True
            

    #theta_old[0]=np.exp(theta_old[0]) # FIXME keep log?
    #theta_new[0]=np.exp(theta_new[0]) # FIXME keep log?
    # FIXME below, all a values logged?
    theta_new[6]=(np.exp(theta_new[0])*theta_new[4]**theta_new[1])/(theta_new[4]**theta_new[2]) #; %Estimation of a2 by continuity
    theta_new[7]=(theta_new[6]*theta_new[5]**theta_new[2])/(theta_new[5]**theta_new[3]) #; %Estimation of a3 by continuity
    prob_new = logistic_likelihood(theta_new, gauged_h, gauged_q, mean_q)
    # MCMC move
    r = min(1, prob_new/prob_old)
    #print(r)
    if (prob_old==0) & (prob_new==0):
        r = 0
    
    u = rng.random()
    if u < r:
        theta[i,:] = theta_new
        prob[i]=prob_new
        n_accepted += 1
        prob_old = prob_new
        theta_old = theta_new
    else:
        theta[i,:] = theta_old
        prob[i]=prob_old
    

# For all parameters calculate rating curves
# For speed, either remove duplicate items or use lru cache (try cache first)
# with cache can get same result as matlab, just for testing at least
# Consider: is this not a flaw with matlab version, that it repeats different accepted parameters??
# It selects every 10th parameter set, but who knows what that translates to in reality
# Applying rating should be faster with piecewise to avoid tested for if mess

# only keep after burn in (1/3rd)
theta_u = theta[int(n_mcmc/3):]
#theta_u = np.unique(theta_u, axis=0) # drop duplicates # TODO skip when comparing to matlab
# TODO also check if this influences the percentiles
# only keep every 10th parameter set (make xth set a parameter)
theta_u = theta_u[range(0, len(theta_u), 10)] 
