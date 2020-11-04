import numpy as np
import torch as th
import networkx as nx
import gradImSnake
import unittest

class test_makeGaussEdgeFltr(unittest.TestCase):
    def t_directionSensitivity(self,ndims):
        fstdev=1.0
        fltr=gradImSnake.makeGaussEdgeFltr(fstdev,ndims)
        
        for d in range(ndims):
            # fltr[i] should be sensitive to gradients along the i-th dim
            # and insensitive to gradients along the other dims
            f=fltr[d,0]
            for e in range(ndims):
                invinds=[slice(None)]*ndims
                invinds[e]=slice(-1,None,-1)
                g=f[invinds]
                if e==d:
                    self.assertTrue((f==-g).all())
                else:
                    self.assertTrue((f== g).all())

    def t_sign(self,ndims):
        fstdev=1.0
        fltr=gradImSnake.makeGaussEdgeFltr(fstdev,ndims)
        for d in range(ndims):
            sz=30
            shape=[1,1]+[1]*ndims
            shape[2+d]=sz
            enim=np.arange(sz).reshape(shape)
            shape=[1,1]+[sz]*ndims
            shape[2+d]=1
            enim=np.tile(enim,shape).astype(np.double)
            if ndims==2:
                gradim=th.nn.functional.conv2d(th.from_numpy(enim),th.from_numpy(fltr))
            elif ndims==3:
                gradim=th.nn.functional.conv3d(th.from_numpy(enim),th.from_numpy(fltr))
            else:
                raise ValueError("ndims can only be 2 or 3")

            for k in range(ndims):
                if k==d:
                    self.assertTrue(th.all(gradim[0,k]>0))
                else:
                    self.assertTrue(th.allclose(gradim[0,k],th.tensor([0.0],dtype=th.double)))

    def test_simple(self):
        self.t_directionSensitivity(2)
        self.t_directionSensitivity(3)
        self.t_sign(2)
        self.t_sign(3)

class testCmptExtGrad(unittest.TestCase):
    def t_Simple(self,ndim):
        snakepos=np.array([[10]*ndim,[15]*ndim,[20]*ndim])
        gradim=[]
        for d in range(ndim):
            shp=[ 1]*ndim
            mlp=[30]*ndim
            shp[d]=30
            mlp[d]= 1
            gradim.append(np.tile(np.arange(30).reshape(shp),mlp))
        gradim=np.stack(gradim,axis=0)
        
        spos=th.from_numpy(snakepos.astype(np.float))
        gimg=th.from_numpy(gradim  .astype(np.float))
        
        gradpos=gradImSnake.cmptExtGrad(spos,gimg)
        self.assertTrue(th.allclose(spos,gradpos))
        
        shp=[ndim]+[1]*ndim
        delta=th.tensor([0]*(ndim-1)+[1],dtype=th.double).reshape(shp)
        gradpos=gradImSnake.cmptExtGrad(spos,gimg+delta)
        self.assertTrue(th.allclose(spos,gradpos-delta.reshape(1,ndim)))
        
        delta=th.tensor([[0]*(ndim-1)+[1]],dtype=th.double)
        spos=spos+delta
        gradpos=gradImSnake.cmptExtGrad(spos,gimg)
        self.assertTrue(th.allclose(spos,gradpos))  

    def test_Simple(self):
        self.t_Simple(2)
        self.t_Simple(3)


class test_gradImSnake(unittest.TestCase):
    def test_convergence_simple(self):
        def test_(dim,ndims):
            sz=30
            target=15
            initial=10
            marg=5
            G=nx.Graph()
            nind=0
            for a in range(marg,sz,marg):
                pos=np.array([a]*ndims)
                pos[dim]=initial
                G.add_node(nind,pos=pos)
                nind+=1
                
            for a in range(1,nind):
                G.add_edge(a-1,a)
                
            stepsz=0.2
            extparam=1
            alpha=0.0
            beta=0.0
            crop=[slice(-100,100)]*ndims
            fltrstdev=1
            extparam=1
    
            enim=th.abs(th.arange(sz,dtype=th.double)-target)
            shp=[1]*(ndims+2)
            shp[dim+2]=sz
            mlt=[1,1]+[sz]*ndims
            enim=enim.reshape(shp).expand(mlt)

            fltr=gradImSnake.makeGaussEdgeFltr(1.0,ndims,)
            gimg=gradImSnake.cmptGradIm(enim,th.from_numpy(fltr))

            s=gradImSnake.GradImSnake(G,crop,stepsz,alpha,beta,ndims,gimg[0])
            
            niter=100
            s.optim(niter)
    
            p=s.getPos()
            self.assertTrue(th.norm(s.getPos()[:,dim]-target)<1e-3)

        test_(0,2)
        test_(1,2)

        test_(0,3)
        test_(1,3)
        test_(2,3)


if __name__ == '__main__':
    unittest.main()
