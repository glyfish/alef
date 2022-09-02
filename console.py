# %%
%load_ext autoreload
%autoreload 2

import os
import sys
import numpy
import pandas
from copy import deepcopy

from matplotlib import pyplot
import statsmodels.api as sm

##
n = 5
μ = numpy.matrix([0.0, 0.0]).T
x0 = numpy.matrix([1.0, 2.0]).T
x0 = numpy.array(x0.T)
m, l = x0.shape
Ω = numpy.matrix(numpy.eye(m))
xt = numpy.zeros((n, l))

xt[0] = x0
xt

Φ = [numpy.matrix([[1.0, 2.0], [3.0, 4.0]])]
t1 =Φ[0]*numpy.matrix(xt[0]).T
t1
numpy.squeeze(numpy.array(t1), axis=1)
