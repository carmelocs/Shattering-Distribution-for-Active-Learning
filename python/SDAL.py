# Generated with SMOP  0.41-beta
from libsmop import *
# SDAL.m

    
    
@function
def SDAL(data=None,k=None,*args,**kwargs):
    varargin = SDAL.varargin
    nargin = SDAL.nargin

    ############################################################################
# Shattering Distribution for Active Learning. Xiaofeng~Cao and Ivor W.~Tsang.
    
    #Email:xiaofeng.cao.uts@gmail.com
    
    #The shattering step.
    
    # radius is set as 0.15
    
    # 1+varepsilon, varepsilon=.02
    
    # Users can adjust the parameters to adapt different datasets
############################################################################
    
    Id,Cen=kmeans(data,k,nargout=2)
# SDAL.m:18
    Center=copy(Cen)
# SDAL.m:20
    Radi=0.25
# SDAL.m:22
    T=0
# SDAL.m:22
    L,R=size(data,nargout=2)
# SDAL.m:23
    f=0
# SDAL.m:24
    f=NumberDensity(data,Center,Radi)
# SDAL.m:27
    while T < 50:

        for j in arange(1,k,1).reshape(-1):
            Ball=[]
# SDAL.m:32
            for i in arange(1,L,1).reshape(-1):
                dist[i]=norm(data(i,arange()) - Center(j,arange()))
# SDAL.m:34
                if dist(i) < Radi:
                    Ball=concat([[Ball],[data(i,arange())]])
# SDAL.m:36
            if length(Ball) == 0:
                Center[j,arange()]=Center(j,arange())
# SDAL.m:41
            else:
                Center[j,arange()]=mean(Ball,1)
# SDAL.m:44
        F=NumberDensity(data,Center,Radi)
# SDAL.m:52
        if F - f == 0 or length(find(pdist(Center) < dot(2,Radi))) > 0:
            break
        else:
            f=copy(F)
# SDAL.m:59
        T=T + 1
# SDAL.m:62
        Radi=dot((1 + 0.1),Radi)
# SDAL.m:63

    
    
    plot(data(arange(),1),data(arange(),2),'b.')
    hold('on')
    plot(Center(arange(),1),Center(arange(),2),'rs')
    hold('on')
    for j in arange(1,k,1).reshape(-1):
        viscircles(Center(j,arange()),Radi)
    
    Idx=knnsearch(data,Center)
# SDAL.m:76
    Center=data(Idx,arange())
# SDAL.m:77