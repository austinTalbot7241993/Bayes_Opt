'''
This tests a BayesianOptimization object based on Botorch
'''

# Author : Austin Talbot <austin.talbot1993@gmail.com>
# Corey Keller

import numpy as np
import numpy.random as rand
import numpy.linalg as la
import matplotlib.pyplot as plt
import torch

import sys,os

rand.seed(1993)

codeDir = '../../Code/BoTorch'
sys.path.append(codeDir)
from bayesOptTMS import BasicBayesOpt

sys.path.append('..')
from utils_unitTest import *

def test_BasicBayesOpt_init():
    print_mtm('BasicBayesOpt init and print')
    try:
        x_train = rand.randn(15,6)
        y_train = rand.randn(15,1)
        bounds = -10*np.ones((2,6))
        bounds[1] = 10
        model = BasicBayesOpt(x_train,y_train,bounds)
        print(model)
        return False
    except:
        return True

def test_BasicBayesOpt_ask():
    print_mtm('BasicBayesOpt ask')
    try:
        x_train = rand.randn(15,6)
        y_train = rand.randn(15,1)
        bounds = -10*np.ones((2,6))
        bounds[1] = 10
        model_options = {'type':'SingleTaskGP'}
        model = BasicBayesOpt(x_train,y_train,bounds,
                                                model_options=model_options)
        ask_val = model.ask()
        return False
    except:
        return True

def test_BasicBayesOpt_answer():
    print_mtm('BasicBayesOpt answer')
    try:
        x_train = rand.randn(15,6)
        y_train = rand.randn(15,1)
        bounds = -10*np.ones((2,6))
        bounds[1] = 10
        model = BasicBayesOpt(x_train,y_train,bounds.T)
        ask_val = rand.randn(1,6)
        ask_y = rand.randn(1,1)
        model.answer(ask_val,ask_y)
        return False
    except:
        return True

def test_BasicBayesOpt_optimum():
    print_mtm('BasicBayesOpt optimum')
    try:
        x_train = rand.randn(15,6)
        y_train = rand.randn(15,1)
        bounds = -10*np.ones((2,6))
        bounds[1] = 10
        model = BasicBayesOpt(x_train,y_train,bounds.T)
        best_x = model.optimum(empirical=True)
        best_x = model.optimum(tested=True)
        best_x = model.optimum()
        best_x,best_y = model.optimum(return_y=True)
        return False
    except:
        return True

def test_BasicBayesOpt_save():
    print_mtm('BasicBayesOpt save')
    try:
        x_train = rand.randn(15,6)
        y_train = rand.randn(15,1)
        model_options = {'type':'FixedNoiseGP'}
        model = BasicBayesOpt(x_train,y_train,model_options=model_options)
        model.version = '-1.0'
        model.save('Example_file.p')
        return False
    except:
        return True

def test_BasicBayesOpt_load():
    print_mtm('BasicBayesOpt load')
    try:
        x_train = rand.randn(15,6)
        y_train = rand.randn(15,1)
        model = BasicBayesOpt(x_train,y_train)
        model.load('Example_file.p')
        return False
    except:
        return True

def main_BasicBayesOpt():
    print_otm('BasicBayesOpt')
    failed = False
    if(test_BasicBayesOpt_init()):
        failed = True
    if test_BasicBayesOpt_ask():
        failed = True
    if test_BasicBayesOpt_answer():
        failed = True
    if test_BasicBayesOpt_optimum():
        failed = True
    if test_BasicBayesOpt_save():
        failed = True
    if test_BasicBayesOpt_load():
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
    main_BasicBayesOpt()








