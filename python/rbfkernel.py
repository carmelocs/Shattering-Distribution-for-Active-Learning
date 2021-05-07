# Generated with SMOP  0.41-beta
from libsmop import *
# rbfkernel.m

    
@function
def rbfkernel(X=None,Y=None,sigma=None,*args,**kwargs):
    varargin = rbfkernel.varargin
    nargin = rbfkernel.nargin

    # RBFKERNEL		Calculate RBF kernel between data matrix X and Y
    
    #	DESCRIPTION
#	Calculate RBF kernel w.r.t. each data point in matrix X and Y. The
#	calculatioin is K(x,y) = exp(-1/sigma^2 * norm(x-y)).
    
    #	INPUT
#		X: N by K matrix where each column vector is a data point
#		Y: M by K matrix where each column vector is a data point
#		SIGMA: The parameter in the kernel calculation
    
    #	OUTPUT
#		K: N by M matrix, each entry K(i,j) is the kernel function on data
#		points X(i,:) and Y(j,:)
    
    N,K=size(X,nargout=2)
# rbfkernel.m:20
    M=size(Y,1)
# rbfkernel.m:21
    Kxy=(dot(ones(M,1),sum((X ** 2).T,1))).T + dot(ones(N,1),sum((Y ** 2).T,1)) - dot(2.0,(dot(X,(Y.T))))
# rbfkernel.m:23
    Kxy=exp(dot(- 0.5,Kxy) / sigma ** 2)
# rbfkernel.m:27