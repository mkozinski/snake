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
    # the snake energy is \alpha\sum_{(u,v)\in E} |u-v|^2 + \beta\sum_{(u,v,w)\in T} |u-2v+w|^2
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

def invertALambdaI(A,stepsz):
    # A is shaped k x k x d
    # stepsz is a scalar
    # returns C shaped k x k x d
    # where C[:,:,i]=(stepsz*A[:,:,d]+I)^-1
    invs=[]
    for d in range(A.shape[-1]):
        invs.append(np.linalg.inv(stepsz*A[:,:,d]+np.eye(A.shape[0])))
        
    return np.stack(invs,axis=2)

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
    # this class lets evolve the position of snake control points to minimize 
    # the sum of internal and external energies
    # the internal energy is:
    #   let u,v,w denote snake control points
    #   let E be the set of snake edges
    #   let T be the set of triplets (u,v,w), such that u and w are the only neighbors of v
    #   the snake energy is \alpha \sum_{(u,v)\in E} |u-v|^2 + \beta \sum_{(u,v,w)\in T} |u-2v+w|^2
    # the gradient of the external energy is delivered as the argument of the step method
    # 
    
    def __init__(self,graph,crop,stepsz,alpha,beta,ndims):
        # a snake is created for a crop of a graph
        #
        # graph is an nx.graph
        #   nodes have attributes "pos", nd-arrays, encoding their positions
        #   (note, that the order of the dimensions is not reversed)
        # crop is a tuple of slice objects; it defines a hypercube;
        #   the snake consists of the sub-graph of "graph" that contains nodes in this hypercube
        #   edges crossing the faces of the hypercube are cut and new nodes on the faces are created
        #   these nodes are constrained to lie on the faces
        # stepsz, a scalar, is the implicit step size
        # alpha is a scalar, the coefficient of the pairwise (spring) internal energy term
        # beta is a scalar, the coefficient of the triplet (curvature) internal energy term
        # ndims should equal 2 or 3, and is the number of dimensions of the space in which the graph nodes live
        #
        
        self.stepsz=stepsz
        self.alpha =alpha
        self.beta  =beta
        self.ndims=ndims
        
        self.h=cropGraph(graph,crop)
        a,s,fd,n2i=getA(self.h,self.alpha,self.beta,self.ndims)
        c=invertALambdaI(a,self.stepsz)
        self.c = th.from_numpy(c)
        self.s = th.from_numpy(s)
        self.fd= th.from_numpy(fd.astype(np.uint8))>0
        self.n2i=n2i
    
    def cuda(self):
        self.c =self.c .cuda()
        self.s =self.s .cuda()
        self.fd=self.fd.cuda()
    
    def step(self,gradext):
        # update the position of the control nodes
        # using the external gradient "gradext"
        # gradext should have the same shape as self.s
        gradext[self.fd]=0.0
        self.s=snakeStep(self.s,gradext,self.c,self.stepsz)
        return self.s
    
    def getPos(self):
        return self.s
    
    def getGraph(self):
        # returns a graph reflecting the current position of snake nodes
        # note, that the graph only contains the nodes that lie within the crop area
        # (see the init method)
        g=self.h.copy()
        for n in g.nodes:
            g.nodes[n]["pos"]=self.s[self.n2i[n],:].cpu().numpy()
        
        return g
    

