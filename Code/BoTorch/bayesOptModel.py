'''
This implements Bayesian Optimization as a python object. This object has
two main methods, an ask method where the model tells you promising 
parameters based on previous data, and answer method where you
the enter the parameters and the associated response.

Initializing the model has 3 required inputs, initial parameter observations
(train_x), associated responses (train_y), and parameter 
bounds (bounds). While I do not make it the default behavior, you should
always scale parameters such that the bounds are [0,1]. So if
using coil angle (-90,90), the associated coordinate in train_x should be
(angle+90)/180. 

This model is designed to be easily modifiable to incorporate better 
methods/techniques. There are two optional dictionaries to pass in, 
acq_options and model_options. The former allows you to customize parameters
of the acquisition function while the latter allows you to customize
parameters of the model. As you get better designs add options to these
dictionaries and potentially modify the default values stored in 

The surrogate model will be stored as self.model (self is the name of
the object instance), every other attribute will not depend on pytorch. 
The model is designed to be easily usable even without a technical 
background for full details see the notebooks in demos.

###########
# Objects #
###########
BasicBayesOpt
    This is the wrapper object that keeps track of the data in an 
    experiment and methods allow for responses and suggestions to 
    be obtained.

###########
# Methods # 
###########
None

Author : Austin Talbot <austin.talbot1993@gmail.com>
Chris Cline 
Corey Keller

Version History:
4/15/2022 - Model creation [AT] (1.0)
5/05/2022 - Separated acq and surrogate options [AT] (1.1)
5/10/2022 - Adding a optimal value function [AT] (1.1.1)
5/12/2022 - Conversion of all model variables to numpy [AT] (1.2.0)
5/13/2022 - Added default bounds, check train_y dim [AT] (1.2.1)
5/14/2022 - Added save and load functions [AT] (1.2.2)
'''
import numpy as np
import pickle
from scipy.stats.qmc import Sobol

import os,sys
import torch
from gpytorch.mlls.sum_marginal_log_likelihood import ExactMarginalLogLikelihood

from botorch.utils.transforms import standardize
from botorch.optim import optimize_acqf
from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement

from datetime import datetime as dt

#################
# Local imports # 
#################
from utils_misc import fill_dict,pretty_string_dict
from acquisition_models import _get_candidates
from surrogate_models import _get_model

version = '1.2.2'

class BasicBayesOpt(object):
    
    def __init__(self,train_x,train_y,bounds=None,acq_options={},
                        model_options={}):
        '''
        This is the wrapper object that keeps track of the data in an 
        experiment and methods allow for responses and suggestions to 
        be obtained.

        Parameters
        ----------
        train_x : np.array-like,(n_samples,n_parameters)
           The initial sampled parameter values 
            
        train_y : np.array-like,(n_samples,1)
           The initial responses
            
        bounds : np.array-like,(2,n_parameters),default=None
            The bounds considered. If none defaults to [0,1]^n_parameters

        acq_options : dict,default={}
            The options used to customize the acquisition function

        model_options : dict,default={}
            The options used to customize the surrogate model

        Attributes
        ----------
        version : string
            The version of the saved model

        creationDate : date-time
            The day the model was created

        Example usage
        -------------
        import numpy as np
        import numpy.random as rand
        import torch
        from bayesOptModel import BasicBayesOpt

        # Generate (random) initial ``Observations''
        train_parameters = rand.randn(15,6)
        train_stim_response = rand.randn(15,1)

        acq_dict = {'raw_samples':1024}
        model_dict = {'type':'FixedNoiseGP'}
        model = BasicBayesOpt(train_parameters,train_stim_response,
                              acq_options=acq_dict,model_options=model_dict)
        print(model)

        # Now a sample of how an ``experiment'' would go. Also suppose that
        # we weren't quite able to test at the recommended value
        N_stimulations = 25

        for i in range(N_stimulations):
            parameters = model.ask()

            # Will be replaced with output from interface
            corrupted_params = (parameters+ 
                                        torch.tensor(rand.randn(1,6)))
            corrupted_response = torch.tensor(rand.randn(1,1))

            model.answer(corrupted_params,corrupted_response)
        '''
        self.train_x = train_x
        if train_y.ndim == 1:
            self.train_y = np.atleast_2d(train_y).T
        else:
            self.train_y = train_y

        self.p = train_x.shape[1]
        if bounds is not None:
            self.bounds = bounds
        else:
            self.bounds = np.zeros((2,self.p))
            self.bounds[1] = 1.0

        self.acq_options = self._fillAcqOpts(acq_options)
        self.model_options = self._fillModelOpts(model_options)

        self._initializeGP()

        self.version = version
        self.creationDate = dt.now()
        self.fitted = False

    def _fillAcqOpts(self,acq_options):
        '''
        This is a method for filling in default values into the acquistion
        function options dictionary. 
        '''
        default_dict = {'type':'noisy_ei','raw_samples':512,
                        'num_candidates':1,'num_restarts':20}
        return fill_dict(acq_options,default_dict)

    def _fillModelOpts(self,model_options):
        '''
        This is a method for filling in default values into the surrogate
        model options dictionary. 
        '''
        default_dict = {'type':'SingleTaskGP','sd':0.25}
        return fill_dict(model_options,default_dict)

    def _initializeGP(self):
        '''
        This intializes the surrogate model (self.model) and likelihood used
        for training (self.mll)
        '''
        train_x = self.train_x
        train_y = self.train_y
        self.model,self.mll = _get_model(train_x,train_y,self.model_options)
        fit_gpytorch_model(self.mll)

    def __repr__(self):
        out_str = 'BasicBayesOpt object\n'
        out_str += 'Version %s\n'%self.version
        out_str += 'Creation date %s\n'%self.creationDate
        out_str += 'Fitted %r\n'%self.fitted
        out_str = out_str + '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n'
        out_str = out_str + 'Acquisition function parameters:\n'
        out_str = out_str + pretty_string_dict(self.acq_options)
        out_str = out_str + '>>>>>>>>>>>>>>>>>\n'
        out_str = out_str + 'Model parameters:\n'
        out_str = out_str + pretty_string_dict(self.model_options)
        return out_str

    def optimum(self,empirical=False,tested=True,n_samples=1024,
                                        return_y=False):
        '''
        This method finds the guess of the best parameter given the 
        observations and trained surrogate model.

        Parameters
        ----------
        empirical : boolean,default=False
            If true, returns best tested observation

        tested : boolean,default=False
            If true and empirical is false returns tested point with highest
            surrogate value. Else return highest surrogate value

        return_y : boolean,default=False
            If true also return the best value

        n_samples : int,
            Number of samples to evaluate function on. SHOULD BE A POWER OF
            2!!!!

        Returns
        -------
        best_x : np.array,(n_params)
            The best parameters

        best_y : float,optional
            The associated expected response
        '''
        if empirical is True:
            best_idx = np.argmax(self.train_y)
            best_x = self.train_x[best_idx]
            best_y = self.train_y[best_idx]
        elif tested is True:
            posterior = self.model.posterior(torch.from_numpy(self.train_x))
            mean_y = posterior.mean.detach().numpy()
            best_idx = np.argmax(mean_y)
            best_x = self.train_x[best_idx]
            best_y = mean_y[best_idx]
        else:
            sampler = Sobol(self.p)
            sample = sampler.random(n_samples)
            posterior = self.model.posterior(torch.from_numpy(sample))
            mean_y = posterior.mean.detach().numpy()
            best_idx = np.argmax(mean_y)
            best_x = sample[best_idx]
            best_y = mean_y[best_idx]
            
        if return_y:
            return best_x,best_y
        else:
            return best_x
        
    def ask(self):
        '''
        This method is a wrapper that calls _get_candidates from 
        acquisition_models.

        Returns
        -------
        new_x : torch.tensor
            Suggested location
        '''
        new_x = _get_candidates(self.model,self.train_x,self.train_y,
                                self.bounds,self.acq_options)
        return new_x

    def answer(self,new_x,new_y):
        '''
        This method incorporates a new observation with parameters new_x and
        an associated response new_y and updates the surrogate model.

        Note that if new_x is a list you can only add 1 observation at a 
        time

        Paramters
        ---------
        new_x : float,list,np.array, or torch.tensor,shape=(n,p)
            The parameters

        new_y : float,list,np.array, or torch.tensor,shape=(1,1)
            The associated response

        Returns
        -------
        None
        '''
        if isinstance(new_x,list):
            new_x = np.array(new_x)
        if isinstance(new_y,float):
            new_y = new_y*np.ones((1,1))
        self.train_x = np.vstack((self.train_x,new_x))
        self.train_y = np.vstack((self.train_y,new_y))
        self._update_model_mll()

    def _update_model_mll(self):
        '''
        This method updates the surrogate model, either on initialization 
        with initial data or after self.answer is called.
        '''
        train_x = self.train_x
        train_y = self.train_y
        self.model,self.mll = _get_model(train_x,train_y,
                                        self.model_options)
        fit_gpytorch_model(self.mll)
        self.fitted = True

    def save(self,fileName):
        '''
        This saves this model to a binary pickle file named filename
        '''
        myDict = {'train_x':self.train_x}
        myDict['train_y'] = self.train_y
        myDict['acq_options'] = self.acq_options
        myDict['p'] = self.p
        myDict['bounds'] = self.bounds
        myDict['version'] = self.version
        myDict['creationDate'] = self.creationDate
        myDict['model_options'] = self.model_options
        pickle.dump(myDict,open(fileName,'wb'))

    def load(self,fileName):
        '''
        This loads a saved pickle file from a previous model and replaces
        all the current attributes.
        '''
        if self.fitted:
            raise ValueError("Don't overwrite a fitted model")

        myDict = pickle.load(open(fileName,'rb'))
        self.train_x = myDict['train_x']
        self.train_y = myDict['train_y']
        self.p = myDict['p']
        self.bounds = myDict['bounds']

        self.acq_options = myDict['acq_options']
        self.model_options = myDict['model_options']

        if self.version != myDict['version']:
            print('Warning: saved version does not match current code version')

        self.version = myDict['version']
        self.creationDate = myDict['creationDate']
        self._update_model_mll()
