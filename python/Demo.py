# Generated with SMOP  0.41-beta
from libsmop import *
# Demo.m

    # clear
    # clc
    K=rbfkernel(data,data,1.8)
# Demo.m:3
    ID=Halving(K,400)
# Demo.m:5
    X=data(ID,arange())
# Demo.m:7
    SDAL(X,4)