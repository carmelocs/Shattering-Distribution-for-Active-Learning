# Generated with SMOP  0.41-beta
from libsmop import *
# Halving.m

    
@function
def Halving(K=None,m=None,lambda_=None,candidate_index=None,*args,**kwargs):
    varargin = Halving.varargin
    nargin = Halving.nargin

    #############################################################################
# Shattering Distribution for Active Learning. Xiaofeng~Cao and Ivor W.~Tsang.
    
    #Email:xiaofeng.cao.uts@gmail.com
    
    #The halving step.
    
    # Usage: [index, variance] = transdesign_sq(K, m, candidate_index, lambda)
    
    # Sequential transductive experiment design for halving
# ridge regression. If kernel is linear, then the algorithm becomes an active
# learning approach for linear ridge regression.
    
    # Inputs:     
#             K--- Kernel matrix of all the data samples
#             m--- Number of data samples to select
#             candidate_index --- IDs of candidate samples from all the samples. 
#                        The default setting is the whole set. By specifing a
#                        random subset, one can save computationl cost.
#             lambda --- A regularization parameter for the kernel ridge regression
# 
# Outputs:
#             index--- Indices of selected data samples
    
    error(nargchk(1,4,nargin))
    if nargin < 4:
        candidate_index=[]
# Halving.m:31
    
    if nargin < 3:
        lambda_=[]
# Halving.m:34
    
    if nargin < 2:
        m=[]
# Halving.m:37
    
    if isempty(m):
        m=1
# Halving.m:40
    
    if isempty(candidate_index):
        candidate_index=concat([arange(1,size(K,1))]).T
# Halving.m:43
    
    if isempty(lambda_):
        lambda_=0.001
# Halving.m:46
    
    n=size(K,1)
# Halving.m:49
    m=min(n,m)
# Halving.m:50
    q=length(candidate_index)
# Halving.m:51
    index=zeros(m,1)
# Halving.m:53
    fprintf('selecting samples ... ')
    for i in arange(1,m).reshape(-1):
        score=zeros(1,q)
# Halving.m:56
        for j in arange(1,q).reshape(-1):
            k=candidate_index(j)
# Halving.m:58
            score[j]=dot(K(k,arange()),K(arange(),k)) / (K(k,k) + lambda_)
# Halving.m:59
        dummy,I=max(score,nargout=2)
# Halving.m:61
        index[i]=candidate_index(I)
# Halving.m:62
        K=K - dot(K(arange(),index(i)),K(index(i),arange())) / (K(index(i),index(i)) + lambda_)
# Halving.m:64
    
    fprintf('done \n')
    index=index(arange(1,m))
# Halving.m:67