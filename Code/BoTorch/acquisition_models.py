'''
This is the file where people can customize the acquisition models. There is
one function called by the BasicBayesOpt object, _get_candidates. This 
function is largely empty and just selects which sub-method to call to 
actually get sampled points. The procedure for incorporating a new 
acquisition function is:

(1) Add an extra method
def _my_cool_acquisition_function(model,train_x,train_y)
    Documentation
    ...
    ...
    return new_x
(2) Add an option to _get_model
    if acq_options['type'] == 'MyCoolAcquisitionFunction':
        new_x = _my_cool_acquisition_function(model,train_x,train_y)
(3) Add function documentation. Any comments using triple quotes after 
the function can be called in the python code as:
print(_my_cool_acquisition_function.__doc__)
(4) Add documentation at the top under list of methods. Then anyone can see
what is the contents of the file by calling:
print(acquisition_models.__doc__)

###########
# Methods # 
###########
_get_candidates(model,train_x,train_y,bounds,acq_options)
    This is a wrapper function that calls the specific acquisition function
    specified in acq_options

_NoisyEI(model,train_x,train_y,bounds,acq_options)
    This uses standard noisy expected improvement in BoTorch.

_Random(model,train_x,train_y,bounds,acq_options)
    This randomly selects a point within the specified bounds. Mainly 
    used to demonstrate how to customize acq function.

Author : Austin Talbot <austin.talbot1993@gmail.com>
Chris Cline
Corey Keller

Version History
5/05/2022 - Created [AT] 
5/13/2022 - Added random acquisition function [AT]
'''
import numpy as np
import pickle
import os,sys
import torch
from botorch.models import FixedNoiseGP
from gpytorch.mlls.sum_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.transforms import standardize
from botorch.optim import optimize_acqf
from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement

import time
import warnings
from datetime import datetime as dt

def _get_candidates(model,train_x,train_y,bounds,acq_options):
    '''
    This is a wrapper function that calls the specific acquisition function
    specified in acq_options

    Parameters
    ----------

    Returns
    -------
    new_x : np.float,(self.p)
        Stimulation parameter recomendation
    '''
    tx = torch.from_numpy(train_x)
    ty = torch.from_numpy(train_y)
    bd = torch.from_numpy(bounds)
    if acq_options['type'] == 'noisy_ei':
        new_x = _NoisyEI(model,tx,ty,bd,acq_options)
    elif acq_options['type'] == 'random':
        new_x = _Random(model,tx,ty,bd,acq_options)
    else:
        raise ValueError('Unrecognized aquisition function %s'%acq_options['type'])
    return new_x

def _NoisyEI(model,train_x,train_y,bounds,acq_options):
    '''
    This uses standard noisy expected improvement in BoTorch.
    '''
    acq_func = qNoisyExpectedImprovement(model=model,
                                        X_baseline=train_x)

    ad = acq_options
    candidates,_ = optimize_acqf(acq_function=acq_func,
                            bounds=bounds,
                            q=ad['num_candidates'],
                            num_restarts=ad['num_restarts'],
                            raw_samples=ad['raw_samples'],  
                            options={"batch_limit": 5, "maxiter": 200})
    new_x = candidates.detach()
    return new_x.numpy()

def _Random(model,train_x,train_y,bounds,acq_options):
    '''
    This randomly selects a point within the specified bounds. Mainly 
    used to demonstrate how to customize acq function.
    '''
    bounds = bounds.numpy()
    p = bounds.shape[1]
    candidate = rand.uniform(low=0,high=1,size=p)
    param_range = bounds[1] - bounds[0]
    candidate = param_range*candidate - bounds[0]
    return candidate
