# Generated with SMOP  0.41-beta
from libsmop import *
import numpy as np

from Halving import Halving
from SDAL import SDAL
from sklearn.metrics.pairwise import rbf_kernel


np.random.seed(0)
data = np.random.rand(800, 2)

K = rbf_kernel(data,data,1.8)

ID = Halving(K,400)

X = data(ID,arange())

SDAL(X,4)