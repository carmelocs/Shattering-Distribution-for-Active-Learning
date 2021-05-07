# Generated with SMOP  0.41-beta
from libsmop import *
# transdesign_kernelridge.m

    
@function
def transdesign_kernelridge(X=None,K=None,Y=None,trsize=None,repeat=None,lambda_=None,multiclass=None,*args,**kwargs):
    varargin = transdesign_kernelridge.varargin
    nargin = transdesign_kernelridge.nargin

    # Usage: accu = transdesign_kernelridge(X, K, Y, trsize, repeat, lambda, multiclass)
    
    # Simulation of transductive experimental design for kernel ridge regression. Since the
# data selection process is independent of labels, we first sequentially
# select a sequence of data samples and them simulate the process of 
# learning with more and more seleted examples.
    
    # Inputs:
#         X --- data matrix, n by d
#         K --- n by n kernel matrix of X
#         Y --- n by m matrix, either multilabel or multiclass (1-of-c coding)
#         trsize --- a row vector, the sizes of training data. Since we are
#                    simulating a sequential process, evalutation will be
#                    done on each size of training data
#         repeat --- number of repeated active learning experiments
#         lambda --- regularization parameter for kernel regression
#         multiclass --- true if Y is multiclass
# 
# Outputs:
#         accu --- repeat x size(trsize), record every evaluated
#                  classification accuracy
    
    max_size=max(trsize)
# transdesign_kernelridge.m:25
    N=size(X,1)
# transdesign_kernelridge.m:26
    accu=zeros(repeat,length(trsize))
# transdesign_kernelridge.m:27
    fprintf('Active learning via sequential transductive experimental design \n')
    # the learning curve is averaged over several random repeats.
    for i_repeat in arange(1,repeat).reshape(-1):
        fprintf('     Random repeat %d: ',i_repeat)
        #     ini_index = classevensamp(Y, 1);            # initialize training data, to ensure at least one class having one example
        candidate_index=randsamp(size(K,1),round(dot(size(K,1),0.5)))
# transdesign_kernelridge.m:35
        samp_index=transdesign_sq(K,max_size,lambda_,candidate_index)
# transdesign_kernelridge.m:36
        #     samp_index = [ini_index; samp_index];       # indices of selected data, in the sequencial order of being selected
        for step in arange(1,length(trsize)).reshape(-1):
            Ntr=trsize(step)
# transdesign_kernelridge.m:39
            Xtr=X(samp_index(arange(1,Ntr)),arange())
# transdesign_kernelridge.m:40
            Ytr=Y(samp_index(arange(1,Ntr)),arange())
# transdesign_kernelridge.m:41
            Ktr=submatrix(K,samp_index(arange(1,Ntr)),samp_index(arange(1,Ntr)))
# transdesign_kernelridge.m:43
            Ypred=dot(dot(K(arange(),samp_index(arange(1,Ntr))),inv(Ktr + dot(lambda_,eye(Ntr)))),Ytr)
# transdesign_kernelridge.m:44
            if multiclass == 1:
                Cpred=oneofc_inv(Ypred)
# transdesign_kernelridge.m:46
                Ctrue=oneofc_inv(Y)
# transdesign_kernelridge.m:47
                accu[i_repeat,step]=sum(Cpred == Ctrue) / N
# transdesign_kernelridge.m:48
            else:
                # if multilabel case ...
                result=multilabel_accu(Ypred,Y)
# transdesign_kernelridge.m:51
                accu[i_repeat,step]=mean(result.ROC)
# transdesign_kernelridge.m:52
    
    fprintf('\n')