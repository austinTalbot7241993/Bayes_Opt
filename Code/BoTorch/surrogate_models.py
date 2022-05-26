'''
This is the file where people can customize the surrogate models. There is 
one function called by the BasicBayesOpt object, _get_model. This function
is largely empty and just selects which sub-method to call to actually
create the model and the associated training method. Thus, to develop an
alternative surrogate model the procedure is:

(1) Add an extra method
def _my_cool_new_surrogate_model(train_x,train_y,model_options,train_yvar):
    Documentation
    ...
    ...
    return model,mll
(2) Add an option to _get_model
    if model_options['type'] == 'MyCoolNewModel':
        model,mll = _my_cool_new_surrogate_model(...)
(3) Add function documentation. Any comments using triple quotes after 
the function can be called in the python code as:
print(_my_cool_new_surrogate_model.__doc__)
(4) Add documentation at the top under list of methods. Then anyone can see
what is the contents of the file by calling:
print(surrogate_models.__doc__)

###########
# Methods #
###########
_FixedNoiseGP(train_x,train_y,model_options)
    This uses a Gaussian process as surrogate model where noise is not 
    learned

_SingleTaskGP(train_x,train_y,model_options)
    This uses a Gaussian process as surrogate model where noise is 
    learned


Author : Austin Talbot <austin.talbot1993@gmail.com>
Chris Cline
Corey Keller

Version History
5/05/2022 - Created [AT] 
5/10/2022 - Changed to numpy inputs [AT]
'''
import numpy as np
import pickle
import os,sys
import torch
from botorch.models import FixedNoiseGP,SingleTaskGP
from gpytorch.mlls.sum_marginal_log_likelihood import ExactMarginalLogLikelihood

from botorch.utils.transforms import standardize
from botorch.optim import optimize_acqf
from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement

import time
import warnings
from datetime import datetime as dt

def _get_model(train_x,train_y,model_options):
    '''
    This determines the surrogate model given the data and the model_options

    Paramters
    ---------
    train_x : tf.torch(n_samples,)
        The stimulation locations previously recorded, should be scaled to
        unit cube

    train_y : tf.torch(n_samples,)
        The associated responses (which will be standardized)

    model_options : dict
        self.model_options: options for GP

    Returns
    -------
    model : botorch GP
        The surrogate model

    mll : likelihood
        The likelihood evaluation used to train the model
    '''
    tx = torch.from_numpy(train_x)
    ty = torch.from_numpy(train_y)
    if model_options['type'] == 'FixedNoiseGP':
        model,mll = _FixedNoiseGP(tx,ty,model_options)
    elif model_options['type'] == 'SingleTaskGP':
        model,mll = _SingleTaskGP(tx,ty,model_options)
    else:
        raise ValueError('Unrecognized GP %s'%model_options['type'])

    return model,mll

def _FixedNoiseGP(train_x,train_y,model_options):
    '''
    This uses a Gaussian process as surrogate model where noise is not 
    learned

    Options
    -------
    sd : Standard deviation of the noise
    '''
    train_yvar = torch.tensor(model_options['sd'])
    model = FixedNoiseGP(train_x,standardize(train_y),
                    train_yvar.expand_as(train_y)).to(train_x)
    mll = ExactMarginalLogLikelihood(model.likelihood,model)
    return model,mll

def _SingleTaskGP(train_x,train_y,model_options):
    '''
    This uses a Gaussian process as surrogate model where noise is 
    learned

    Options 
    -------
    None
    '''
    model = SingleTaskGP(train_x,standardize(train_y))
    mll = ExactMarginalLogLikelihood(model.likelihood,model)
    return model,mll

