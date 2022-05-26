'''
This code generates synthetic tuning curves using Gaussian processes. This 
is done by passing in x and y values and then learning a GP to the points.
This becomes the synthetic tuning curve. You can separately specify the 
noise model associated with the observations, allowing different types of 
heteroskadicity in the observations.

###########
# Objects #
###########
ResponseCurveGP(self,x_vals=None,y_vals=None,x_lim=None,kernel_options={},
                    noise_options={})
    This is an object that creates a random response curve and sampling 
    method to simulate an experiment.

###########
# Methods #
###########
_generate_kernel(kernel_options)
    This determines the kernel type (smoothness) of tuning curve mean

_noise_model(x,noise_options)
    This generates the observational noise when sampling

Author : Austin Talbot <austin.talbot1993@gmail.com>
Chris Cline
Corey Keller

Version History:
5/13/2022 - Model creation [AT] (1.0)
'''
import numpy as np
import pickle
from sklearn.gaussian_process import kernels
from sklearn.gaussian_process import GaussianProcessRegressor
import numpy.random as rand
import matplotlib.pyplot as plt

from datetime import datetime as dt

from utils_misc import fill_dict,pretty_string_dict

version = "1.1.0"

class ResponseCurveGP(object):
    def __init__(self,x_vals=None,y_vals=None,x_lim=None,kernel_options={},
                    noise_options={}):
        '''
        Parameters
        ----------
        x_vals : np.array-like(p,)
            The parameter values used to generate curve

        y_vals : np.array-like(p,)
            The measured ``responses'' to generate curve

        x_lim : list,default=[0,1]
            The limits of response curve

        kernel_options : dict
            Determines smoothness of response function

        noise_options : dict
            Options for noise (heteroskedastic etc)
        '''

        if x_vals is None:
            self.x_vals = np.linspace(0,1,num=10)
            self.y_vals = rand.randn(10) + 8*np.sin(.3*self.x_vals) - 2*self.x_vals
            self.x_lim = [0,1]
        elif x_lim is None:
            self.x_vals = x_vals
            self.y_vals = y_vals
            self.x_lim = [0,1]
        else:
            self.x_vals = x_vals
            self.y_vals = y_vals
            self.x_lim = x_lim

        self.n_observations = len(self.y_vals)

        self.kernel_options = self._fillKernelOpts(kernel_options)
        self.noise_options = self._fillNoiseOpts(noise_options)

        self._create_GP_model()
        self._compute_global_optimum()

        self.version = version
        self.creationDate = dt.now()

    def __repr__(self):
        out_str = 'ResponseCurveGP object\n'
        out_str += 'Version %s\n'%self.version
        out_str += 'Creation date %s\n'%self.creationDate
        out_str = out_str + '>>>>>>>>>>>>>>>>>>>>>>>>\n'
        out_str = out_str + 'Tuning Curve Parameters:\n'
        out_str = out_str + pretty_string_dict(self.kernel_options)
        out_str = out_str + '>>>>>>>>>>>>>>>>>\n'
        out_str = out_str + 'Noise parameters:\n'
        out_str = out_str + pretty_string_dict(self.noise_options)
        return out_str

    def _fillKernelOpts(self,kernel_options):
        '''
        This provides default values for the GP kernel
        '''
        default_dict = {'kernel':'Matern','length_scale':0.1,'nu':1.5}
        return fill_dict(kernel_options,default_dict)

    def _fillNoiseOpts(self,noise_options):
        '''
        This provides default values for the noise options.
        '''
        default_dict = {'type':'constant','sigma':.3,
                        'intercept':1.0,'slope':1.0,
                        'exponent':0.5}
        return fill_dict(noise_options,default_dict)

    def _create_GP_model(self):
        '''
        This fits the underlying response curve given the synthetic data
        '''
        kernel = _generate_kernel(self.kernel_options)
        self.gpr = GaussianProcessRegressor(kernel=kernel)
        self.gpr.fit(np.atleast_2d(self.x_vals).T,self.y_vals)
        
    def _compute_global_optimum(self):
        '''
        This finds the optimal parameters (param_max) and the associated
        true tuning curve value (response_max).
        '''
        x_estimate = np.linspace(self.x_lim[0],self.x_lim[1],num=10000)
        y_estimate = self.gpr.predict(np.atleast_2d(x_estimate).T)
        y_global_max = max(y_estimate)
        x_global_max = x_estimate[np.squeeze(y_estimate==y_global_max)]
        self.param_max = x_global_max
        self.response_max = y_global_max

    def mean_function(self,n_points=1000,return_xs=False):
        '''
        Return the posterior mean over the range of function

        Parameters
        ----------
        n_points : float
            Number of points to evaluate posterior mean

        return_xs : bool
            If true return where the posterior was evaluated at

        Returns
        -------
        mean_function : np.array(n_points,)
            The posterior mean 

        xs : np.array(n_points,)
            The points where posterior mean evaluated at 
        '''
        xs = np.linspace(self.x_lim[0],self.x_lim[1],num=n_points)
        mean_function = self.gpr.predict(np.atleast_2d(xs).T)
        if return_xs:
            return mean_function,xs
        else:
            return mean_function

    def response(self,x):
        '''
        Return the posterior mean at a particular point

        Parameters
        ----------
        x : np.array-like(n,1) 
            Parameter value 

        Returns
        -------
        mean_function : np.array(n_points,)
            The posterior mean 

        '''
        mean_function = self.gpr.predict(np.atleast_2d(x).T)
        return mean_function

    def sample(self,x):
        '''
        Stimulating at x, generate a synthetic response

        Parameters
        ----------
        x : float
            Parameter value 

        Returns
        -------
        response : float
            The ``observed'' from stimulation
        '''
        true_response = self.gpr.predict(x*np.ones((1,1)))
        noise = _noise_model(x,self.noise_options)
        response = true_response + noise*rand.randn()
        return response

    def plot(self,opts={},saveName='Default.png',saveFig=False):
        '''
        This plots the response curve and inherent sampling noise.

        Parameters
        ----------
        opts : dict
            Options for plot, including font size and colors

        saveName : string,default=default.png
            Name of file for saving the figure

        saveFig : bool,default=False
            Should figure be saved to png file

        Returns
        -------
        None
        '''
        xx = np.linspace(self.x_lim[0],self.x_lim[1],num=1000)
        yy_hat = np.squeeze(self.gpr.predict(np.atleast_2d(xx).T))
        yy_noise = np.array([_noise_model(xx[i],
                                self.noise_options) for i in range(1000)])

        ub = yy_hat + 1.96*yy_noise
        lb = yy_hat - 1.96*yy_noise

        default_dict = {'fs1':24,'fs2':16,'fs3':16,'lw':3,
                        'c1':'gold','c2':'dodgerblue','alpha':0.7,
                        'exponent':0.5}
        opts = fill_dict(opts,default_dict)

        plt.plot(xx,yy_hat,lw=opts['lw'],c=opts['c1'],label='Response')
        plt.fill_between(xx,lb,ub,label='95% CI',alpha=opts['alpha'],
                        color=opts['c2'])
        plt.xlim(self.x_lim)
        plt.title('Response Curve',fontsize=opts['fs1'])
        plt.ylabel('Response',fontsize=opts['fs2'])
        plt.xlabel('Parameter',fontsize=opts['fs2'])
        plt.legend(loc='lower left',fontsize=opts['fs3'])
        plt.tight_layout()

        if saveFig:
            plt.savefig(saveName,dpi=300)

def _generate_kernel(kernel_options):
    '''
    This determines the kernel type (smoothness) of tuning curve mean

    Parameters
    ----------
    kernel_options : dict
        The same dictionary used in ResponseCurveGP

    Returns
    -------
    kernel : sklearn.gasussian_process.kernel
        The GP kernel influencing smoothness of response curve
    '''
    length_scale = kernel_options['length_scale']
    nu = kernel_options['nu']
    if kernel_options['kernel'] == 'Matern':
        kernel = kernels.Matern(length_scale=length_scale,nu=nu,
                                length_scale_bounds='fixed') 
    elif kernel_options['kernel'] == 'RBF':
        kernel = kernels.Matern(length_scale=length_scale,
                                length_scale_bounds='fixed')
    elif kernel_options['kernel'] == 'RationalQuadratic':
        kernel = kernels.RationalQuadratic(length_scale=length_scale,
                                length_scale_bounds='fixed')
    else:
        raise ValueError('Unrecognized kernel')
   
    kernel = kernel + kernels.WhiteKernel()
    return kernel

def _noise_model(x,noise_options):
    '''
    This generates the observational noise when sampling

    Parameters
    ----------
    x : float
        Parameter value to measure response

    noise_options : dict
        The same dictionary in ResponseCurveGP

    Returns
    -------
    noise_value : float
        The response noise
    '''
    sigma = noise_options['sigma']
    intercept = noise_options['intercept']
    exponent = noise_options['exponent']
    slope = noise_options['slope']
    if noise_options['type'] == 'constant':
        noise_value = sigma
    elif noise_options['type'] == 'linear':
        noise_value = sigma*np.maximum(slope*x + intercept,0)
    elif noise_options['type'] == 'exponential':
        noise_value = sigma*np.exp(exponent*x + intercept)
    else:
        raise ValueError('Unrecognized noise type'%noise_options['type'])
    return noise_value

