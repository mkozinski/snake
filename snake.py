import numpy as np
from functools import reduce
import torch as th

def nodeInside(pos,crop):
    # pos i an np-array containing the dimensions of a single point in k dimensions
    # crop is a tuple of slice objects, defining a crop of k-dimensional space
    # the function returns True if pos lies inside crop, using pytorch index arithmetics
    # (i.e., location 29.5 is outside of array of size 30, but location 0 is inside)
    
    assert len(pos)==len(crop)
    for p,l in zip(pos,crop):
        if p<l.start or p>l.stop-1:
            return False
    return True
    
def cropGraph(G,crop):
    # G is a nx.Graph, whose nodes have attributes called "pos";
    # each of these attributes is an np.array
    # and determines a position of the node
    # (the coordinates go in the standard order: 0th, 1st, 2nd, etc)
    # G.nodes[n]["pos"]==np.array([1,2,3]) means node n is at position 1,2,3
    #
    # crop is a tuple of slice objects
    #
    # this function returns another graph H,
    # which contains the nodes of G that lie inside the crop,
    # and, for each edge of G that crosses crop boundary,
    # a node lying on the crop boundary,
    # and connected to the end of the edge that is inside the crop;
    # these "boundary nodes" have an attribute called "fixedDim",
    # set to the index of the dimension perpendicular to the crop boundary that they traverse
    H=G.copy()
    nodes2delete=[]
    boundaryNodes=[]
    maxind=0
    for n in H.nodes:
        maxind=max(maxind,n)
        p=H.nodes[n]["pos"]
        if not nodeInside(p,crop):
            nodes2delete.append(n)
        else:
            # for each edge that goes outside of the crop, establish a new node
            # at the point where the edge crosses the crop boundary
            for m in H[n]:
                q=H.nodes[m]["pos"]
                if not nodeInside(q,crop):
                    # find the position at which the edge cuts the crop boundary
                    a=1.0
                    dim=0
                    for pp,qq,l,ind in zip(p,q,crop,range(len(crop))):
                        b=2.0
                        if qq<l.start:
                            b=(l.start-pp)/(qq-pp)
                        elif qq>l.stop-1:
                            b=(l.stop-1 -pp)/(qq-pp)
                        if b<a:
                            a=b
                            dim=ind
                    inters=a*q+(1-a)*p
                    boundaryNodes.append((inters,n,dim)) 
    H.remove_nodes_from(nodes2delete)
    
    newnode=maxind
    for position,ind,dim in boundaryNodes:
        newnode=newnode+1
        H.add_node(newnode,pos=position,fixedDim=dim)
        H.add_edge(newnode,ind)
        
    return H
def getA(G,alpha,beta,dims):
    # G is an nx.graph
    #   nodes have attributes "pos", nd-arrays, encoding their positions
    #   (note, that the order of the dimensions is not reversed)
    #   some nodes have an attribute "fixedDim", a scalar
    # alhpha is a scalar
    # beta is a scalar
    #
    # let u,v,w denote snake control points
    # let E be the set of snake edges
    # let T be the set of triplets (u,v,w), such that u and w are the only neighbors of v
    # the snake energy is \sum_{(u,v)\in E} |u-v|^2 + \sum_{(u,v,w)\in T} |u-2v+w|^2
    # Note, that the energy can be separated across dimensions of the space in which the control points live
    # 
    # the gradient of the snake energy with respect to control point coordinates in one of the dimensions
    # can be expressed as Ax, where A is a matrix and x is a vector of positions of snake control points
    # 
    # this function computes the matrix A for each dimension and
    # returns:
    #     a tensor A of size k X k X d, where A[:,:,i] is the matrix A for dimension i
    #     a tensor snake0 of size k X d, containing the positions of snake control points
    #     a boolean tensor fixedDim of size k X d,
    #         where fixedDim[i,j]=True if j-th coordinate of point i should be frozen
    #         we freeze coordinates of points that are on the boundary of a crop (see cropGraph)
    #     a mapping node2ind from graph nodes to indices of snake0 and fixedDim
    
    ind=0
    node2ind={}
    for n in G.nodes:
        node2ind[n]=ind
        ind+=1
    
    A       =np.zeros((ind,ind,dims))
    snake0  =np.zeros((ind,dims))
    fixedDim=np.zeros((ind,dims),dtype=np.bool)
    
    for u,v in G.edges:
        # add the gradient of \alpha|u-v|^2
        # with respect to u and v
        i1=node2ind[u]
        i2=node2ind[v]
        A[i1,i1]+= alpha
        A[i1,i2]+=-alpha
        A[i2,i2]+= alpha
        A[i2,i1]+=-alpha
        
    for n in G.nodes:
        i1=node2ind[n]
        snake0[i1]=G.nodes[n]["pos"]
        if len(G[n])==2:
            # add the gradient of \beta|u-2v+w|^2
            # with respect to u,v and w
            n2,n3=G[n]
            i2,i3=node2ind[n2],node2ind[n3]
            A[i1,i1]+= 4*beta
            A[i1,i2]+=-2*beta
            A[i1,i3]+=-2*beta
            A[i2,i1]+=-2*beta
            A[i3,i1]+=-2*beta
            A[i2,i2]+=   beta
            A[i3,i3]+=   beta
            A[i2,i3]+=   beta
            A[i3,i2]+=   beta
        if "fixedDim" in G.nodes[n]:
            # zero the gradient for fixed nodes
            A[i1,:,G.nodes[n]["fixedDim"]]=0.0
            fixedDim[i1,G.nodes[n]["fixedDim"]]=True
    
    return A,snake0,fixedDim,node2ind

def invertA(A,stepsz):
    # A is shaped k x k x d
    # stepsz is a scalar
    # returns C shaped k x k x d
    # where C[:,:,i]=(stepsz*A[:,:,d]+I)^-1
    invs=[]
    for d in range(A.shape[-1]):
        invs.append(np.linalg.inv(stepsz*A[:,:,d]+np.eye(A.shape[0])))
        
    return np.stack(invs,axis=2)

    
def makeGaussEdgeFltr(stdev,d):
    # make a Gaussian-derivative-based edge filter
    # filter size is determined automatically based on stdev
    # the filter is ready to be used with pytorch conv 
    # input params:
    #   stdev - the standard deviation of the Gaussian
    #   d - number of dimensions
    # output:
    #   fltr, a np.array of size d X 1 X k X k,
    #         where k is an odd number close to 4*stdev
    #         fltr[i] contains a filter sensitive to gradients
    #         along the i-th dimension

    fsz=round(2*stdev)*2+1 # filter size - make the it odd

    n=np.arange(0,fsz).astype(np.float)-(fsz-1)/2.0
    s2=stdev*stdev
    v=np.exp(-n**2/(2*s2)) # a Gaussian
    g=n/s2*v # negative Gaussian derivative

    # create filter sensitive to edges along dim0
    # by outer product of vectors
    shps = np.eye(d,dtype=np.int)*(fsz-1)+1
    reshaped = [x.reshape(y) for x,y in zip([g]+[v]*(d-1), shps)]
    fltr=reduce(np.multiply,reshaped)
    fltr=fltr/np.sum(np.abs(fltr))
    
    # add the out_channel, in_channel initial dimensions
    fltr_=fltr[np.newaxis,np.newaxis]
    # transpose the filter to be sensitive to edges in all directions 
    fltr_multidir=np.concatenate([np.moveaxis(fltr_,2,k) for k in range(2,2+d)],axis=0)
    
    return fltr_multidir

def cmptGradIm(img,fltr):
    # convolves img with fltr, with replication padding
    # fltr is assumed to be of odd size
    # img  is either 2D: batch X channel X height X width
    #             or 3D: batch X channel X height X width X depth
    #      it is a torch tensor
    # fltr is either 2D: 2 X 1 X k X k
    #             or 3D: 3 X 1 X k X k X k
    #      it is a torch tensor
    
    if img.dim()==4:
        img_p=th.nn.ReplicationPad2d(fltr.shape[2]//2).forward(img)
        return th.nn.functional.conv2d(img_p,fltr)
    if img.dim()==5:
        img_p=th.nn.ReplicationPad3d(fltr.shape[2]//2).forward(img)
        return th.nn.functional.conv3d(img_p,fltr)
    else:
        raise ValueError("img should have 4 or 5 dimensions")

def cmptExtGrad(snakepos,eGradIm):
    # returns the values of eGradIm at positions snakepos
    # snakepos  is a k X d matrix, where snakepos[i,j,:] represents a d-dimensional position of the j-th node of the i-th snake
    # eGradIm   is a tensor containing the energy gradient image, either of size
    #           3 X d X h X w, for 3D, or of size
    #           2     X h X w, for 2D snakes
    # returns a tensor of the same size as snakepos,
    # containing the values of eGradIm at coordinates specified by snakepos
    
    # scale snake coordinates to match the hilarious requirements of grid_sample
    scale=th.tensor(eGradIm.shape[1:]).reshape((1,-1)).type_as(snakepos)-1.0
    sp=2*snakepos/scale-1.0
    
    if eGradIm.shape[0]==3:
        # invert the coordinate order to match other hilarious specs of grid_sample
        spi=th.einsum('km,md->kd',[sp,th.tensor([[0,0,1],[0,1,0],[1,0,0]]).type_as(sp).to(sp.device)])
        egrad=th.nn.functional.grid_sample(eGradIm[None],spi[None,None,None])
        egrad=egrad.permute(0,2,3,4,1)
    if eGradIm.shape[0]==2:
        # invert the coordinate order to match other hilarious specs of grid_sample
        spi=th.einsum('kl,ld->kd',[sp,th.tensor([[0,1],[1,0]]).type_as(sp).to(sp.device)])
        egrad=th.nn.functional.grid_sample(eGradIm[None],spi[None,None])
        egrad=egrad.permute(0,2,3,1)
        
    return egrad.reshape_as(snakepos)


def snakeStep(snakepos,extgrad,cmat,stepsz):
    # the update equation is ((stepsz*A+I)^-1)*(snakepos-stepsz*extgrad)
    # cmat represents (stepsz*A+I)^-1
    #
    # all the arguments are torch tensors
    # snakepos  is a k X d matrix, where snakepos[i,j,:] represents a d-dimensional position of the j-th node of the i-th snake
    # extgrad   is a k X d matrix, where extgrad[i,j,:] represents the gradient of the external energy of the i-th snake w.r.t. the j-th control point
    # cmat      is a k X k X d tensor; cmat[:,:,i] is a matrix (stepsz*A+I)^-1 for dimension i
    # stepsz    is a scalar; it is the implicit step size
    # 
    # this function returns newsnakepos= cmat * (snakepos - stepsz*extparam*extgrad)
    # where the first multiplication is matrix-vector;
    # this calculation should be performed separately for each dimension 0<=i<d
    
    # we can pack the calculation into a single function call
    newsnakepos=th.einsum("lkd,kd->ld",[cmat,snakepos-stepsz*extgrad])
    
    return newsnakepos

class Snake():
    # represents the topology, position, and internal energy of a single snake
    
    def __init__(self,graph,crop,stepsz,alpha,beta,ndims):
        
        self.stepsz=stepsz
        self.alpha =alpha
        self.beta  =beta
        self.ndims=ndims
        
        self.h=cropGraph(graph,crop)
        a,s,fd,n2i=getA(self.h,self.alpha,self.beta,self.ndims)
        c=invertA(a,self.stepsz)
        self.c = th.from_numpy(c)
        self.s = th.from_numpy(s)
        self.fd= th.from_numpy(fd.astype(np.uint8))>0
        self.n2i=n2i
    
    def cuda(self):
        self.c =self.c .cuda()
        self.s =self.s .cuda()
        self.fd=self.fd.cuda()
    
    def step(self,gradext):
        gradext[self.fd]=0.0
        self.s=snakeStep(self.s,gradext,self.c,self.stepsz)
        return self.s
    
    def getPos(self):
        return self.s
    
    def getGraph(self):
        
        g=self.h.copy()
        for n in g.nodes:
            g.nodes[n]["pos"]=self.s[self.n2i[n],:].cpu().numpy()
        
        return g
    
class GradimSnake(Snake):
    # a snake with external energy gradients sampled from a "gradient image"
    
    def __init__(self,graph,crop,stepsz,alpha,beta,ndims,gimg):
        super(GradimSnake,self).__init__(graph,crop,stepsz,alpha,beta,ndims)
        self.gimg=gimg
    
    def cuda(self):
        super(GradimSnake,self).cuda()
        self.gimg=self.gimg.cuda()
        
    def step(self):
        return super(GradimSnake,self).step(cmptExtGrad(self.s,self.gimg))
    
    def optim(self,niter):
        for i in range(niter):
            self.step()
        return self.s
