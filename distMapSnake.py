import numpy as np
from functools import reduce
import torch as th

from snake import Snake

from renderDistanceMap import cmptExtEnergyEuclDist_wGrad

class DistMapSnake(Snake):
    # a snake with external energy gradients sampled from a "gradient image"
    
    def __init__(self,graph,crop,stepsz,alpha,beta,ndims,distim,cropsz,dmax,
                 maxedgelength,gradfac=1.0):
        # gimg is the "gradient image"
        # a tensor of size ndim X h X w for 2D snake, or ndim X d X h X w, for 3D
        # gimg[i,h,w] contains the gradient of the external energy
        # with respect to the i-th coordinate of a control point located at (h,w)
        super(DistMapSnake,self).__init__(graph,crop,stepsz,alpha,beta,ndims)
        self.distim=distim
        self.cropsz=cropsz
        self.dmax= dmax
        self.maxedgelength=maxedgelength
        self.gradfac=gradfac
    
    def cuda(self):
        super(DistMapSnake,self).cuda()
        self.distim=self.distim.cuda()
        
    def step(self):
        # external gradient for each control point is extracted from gimg
        e,g=cmptExtEnergyEuclDist_wGrad(self.distim,self.getGraph(),self.s,
                                        self.n2i,self.distim.shape,self.cropsz,
                                        self.dmax,self.maxedgelength)
        self.lastenergy=e
        self.lastgrad=self.gradfac*g
        return super(DistMapSnake,self).step(self.lastgrad)
    
    def optim(self,niter):
        # update the snake niter times
        for i in range(niter):
            self.step()
        return self.s

