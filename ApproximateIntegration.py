from numpy import e, exp, cos, arccos, sin, arctan, tan, pi, sqrt; from numpy import array as ary; import numpy as np; tau = 2*pi
from matplotlib import pyplot as plt
from scipy.integrate import quad, quadrature
from math import fsum
from numpy import log as ln

NUM_ITER = 300 # how many different values of m to try.
LOWER_BOUND = 0 # assume P(x) = 0 for x<LOWER_BOUND
UPPER_BOUND = 1 # assume P(x) = 0 for x>UPPER_BOUND
STEP_SIZE = 100 #The largest m that will be tried is STEP_SIZE * (NUM_ITER-1)
PLOT_ONCE = True

def lnOmega_N(steps, func, int_func=None):
    P = func(steps)
    summation = []
    if type(int_func)==type(None): # use scipy's integrater.
        int_func = lambda a, b : quadrature(func, a, b)[0]

    for i in range( len(steps)-1 ) : #exclude the end point.
        Pj = int_func(steps[i], steps[i+1])
        summation.append( Pj*ln(Pj) )

    global PLOT_ONCE
    if PLOT_ONCE:
        PLOT_ONCE = False
        assert np.isclose(int_func(0,1), 1), "'func' needs to be normalized"
        plt.step(steps, func(steps), where='post', label=r'$P(x)$')
        plt.step(steps, lengthen(summation), where='post', label=r'$P(x)ln(P(x))$')
        plt.axhline(color='black', label='x-axis')
        plt.legend()
        plt.title(r"The area above the $P(x)ln(P(x))$ line will be summed"+"\nto approximate integration")
        plt.xlabel('x')
        plt.show()
    return -fsum(summation)

def lengthen(lst):
    l2 = lst.copy()
    l2.append(lst[-1])
    return l2

if __name__=='__main__':
    m, y = [], []

    ######## LOOK HERE ########
    def func(x): #make sure it's 
        '''
        Change the function below, 
        and make sure that it is normalized.
        '''
        return exp(-x)/(1-1/e)
        return 3*x**2 # x^2 function
        return 1 # flat distribution

    def int_func(a, b):
        '''Write its corresponding integral here'''
        return -(exp(-b)-exp(-a))/(1-1/e)
        return b**3 - a**3
        return b-a
    
    for i in range(1,NUM_ITER):
        num_steps = i*STEP_SIZE
        m.append(num_steps)
        x = np.linspace(LOWER_BOUND, UPPER_BOUND, num_steps, endpoint=True)
        '''Change 'int_func' below to 'None' if the integral is not specified.'''
        y.append(lnOmega_N(x, func, int_func))

    plt.plot(m, ary(y)/ln(m))
    plt.ylabel('entropy ='+r'$\frac{ln(\Omega)}{N} \frac{1}{ln(m)}$')
    plt.xlabel('m')
    plt.show()