# Generated with SMOP  0.41-beta
from libsmop import *
# NumberDensity.m

    
@function
def NumberDensity(data=None,Center=None,Radius=None,*args,**kwargs):
    varargin = NumberDensity.varargin
    nargin = NumberDensity.nargin

    f=0
# NumberDensity.m:2
    for i in arange(1,length(data),1).reshape(-1):
        Ball_dist=[]
# NumberDensity.m:4
        dist=[]
# NumberDensity.m:4
        for j in arange(1,length(Center),1).reshape(-1):
            dist[j]=norm(data(i,arange()) - Center(j,arange()))
# NumberDensity.m:6
            if dist(j) < Radius:
                Ball_dist=concat([[Ball_dist],[dist(j)]])
# NumberDensity.m:8
        f=f + sum(exp(Ball_dist / 1.8) ** 2) / (length(Ball_dist) + 1)
# NumberDensity.m:11
    