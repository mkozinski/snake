
import torch as th
import torch.nn as nn
import distMapSnake

class Loss_MSE_DistSnake_noGrad(nn.Module):
    def __init__(self, stepsz,alpha,beta,ndims,nsteps,
                       cropsz,dmax,maxedgelen,extgradfac):
        super(Loss_MSE_DistSnake_noGrad,self).__init__()
        self.stepsz=stepsz
        self.alpha=alpha
        self.beta=beta
        self.ndims=ndims
        self.cropsz=cropsz
        self.dmax=dmax
        self.maxedgelen=maxedgelen
        self.extgradfac=extgradfac
        self.nsteps=nsteps

        self.iscuda=False

    def cuda(self):
        super(Loss_MSE_DistSnake_noGrad,self).cuda()
        self.iscuda=True
        return self

    def forward(self,pred_dmap,lbl_graphs):
    
        pred_=pred_dmap.detach()
        snake_dmap=[]

        for l,p in zip(lbl_graphs,pred_):
            imsz=p.shape[1:]
            crop=[slice(0,s) for s in imsz]
            s=distMapSnake.DistMapSnake(l,crop,self.stepsz,self.alpha,
                                       self.beta,self.ndims,
                                       p[0],self.cropsz,self.dmax,self.maxedgelen,
                                       self.extgradfac)
            if self.iscuda: s.cuda()

            s.optim(self.nsteps)

            dmap=s.renderDistanceMap(imsz,self.cropsz,self.dmax,
                                     self.maxedgelen)
            snake_dmap.append(dmap)

        snake_dm=th.stack(snake_dmap,0).unsqueeze(1)
        loss=th.pow(pred_dmap-snake_dm,2).mean()
                  
        self.snake=s
        self.dmap=dmap
        
        return loss

