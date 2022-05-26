'''
This tests a ResponseCurveGP object 
'''

# Author : Austin Talbot <austin.talbot1993@gmail.com>
# Corey Keller

import numpy as np
import numpy.random as rand
import numpy.linalg as la
import matplotlib.pyplot as plt
import sys,os

rand.seed(1993)

codeDir = '../../Code/Simulation_1D'
sys.path.append(codeDir)
from generate_response_curves_gp import ResponseCurveGP

sys.path.append('..')
from utils_unitTest import *

def test_ResponseCurveGP_init():
    print_mtm('ResponseCurveGP init and print')
    try:
        model = ResponseCurveGP()
        print(model)
        return False
    except:
        return True

def test_ResponseCurveGP_sample(plot_figure=True):
    print_mtm('ResponseCurveGP sample')
    try:
        model = ResponseCurveGP()
        xs = rand.uniform(low=0,high=1,size=10)
        ys = np.array([model.sample(xs[i]) for i in range(len(xs))])
        post_mean,x_values = model.mean_function(return_xs=True)
        if plot_figure:
            plt.plot(x_values,post_mean,label='Response curve')
            plt.scatter(xs,ys,c='gold',label='Sampled points')
            plt.scatter(model.x_vals,model.y_vals,c='deeppink',
                                label='Function points')
            plt.legend(loc='upper left')
            plt.show()
        return False    
    except:
        return True

def test_ResponseCurveGP_plot(plot_figure=True):
    print_mtm('ResponseCurveGP plot')
    try:
        model = ResponseCurveGP()
        model.plot()
        if plot_figure:
            plt.show()
        return False
    except:
        return True
def main_ResponseCurveGP():
    print_otm('ResponseCurveGP')
    failed = False
    if test_ResponseCurveGP_init():
        failed = True
    if test_ResponseCurveGP_sample(plot_figure=False):
        failed = True
    if test_ResponseCurveGP_plot(plot_figure=False):
        failed = True

    if failed:
        print('!!!!!!!!!!!')
        print('!!!!!!!!!!!')
        print('Test Failed')
        print('!!!!!!!!!!!')
        print('!!!!!!!!!!!')
    else:
        print('Success!')
if __name__ == "__main__":
    main_ResponseCurveGP()








