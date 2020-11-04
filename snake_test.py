import numpy as np
import torch as th
import networkx as nx
import snake
import unittest

class test_nodeInside(unittest.TestCase):
    def test_simple(self):
        self.assertTrue (snake.nodeInside(np.array([1,2,3]),[slice(0,2),slice(1,3),slice(2,4)]))
        self.assertFalse(snake.nodeInside(np.array([2,2,4]),[slice(0,2),slice(1,3),slice(2,4)]))


class test_cropGraph(unittest.TestCase):
    def test_dim0upper1(self):
        g=nx.Graph()
        g.add_node(1,pos=np.array([1,4,1]))
        g.add_node(2,pos=np.array([2,3,1]))
        g.add_node(3,pos=np.array([4,1,1]))
        g.add_edge(1,2)
        g.add_edge(2,3)
        
        h=snake.cropGraph(g,[slice(0,4),slice(0,5),slice(0,6)])
        self.assertTrue(len(h.nodes)==3)
        self.assertTrue((h.nodes[1]["pos"]==g.nodes[1]["pos"]).all())
        self.assertTrue((h.nodes[2]["pos"]==g.nodes[2]["pos"]).all())
        self.assertTrue((h.nodes[4]["pos"]==np.array([3,2,1])).all())
            
    def test_dim0upper2(self):
        g=nx.Graph()
        g.add_node(1,pos=np.array([1,4,1]))
        g.add_node(2,pos=np.array([2,3,1]))
        g.add_node(3,pos=np.array([4,1,1]))
        g.add_edge(1,2)
        g.add_edge(2,3)
        
        h=snake.cropGraph(g,[slice(0,5),slice(0,4.5),slice(0,6)])
        self.assertTrue(len(h.nodes)==3)
        self.assertTrue((h.nodes[4]["pos"]==np.array([1.5,3.5,1])).all())
        self.assertTrue((h.nodes[2]["pos"]==g.nodes[2]["pos"]).all())
        self.assertTrue((h.nodes[3]["pos"]==g.nodes[3]["pos"]).all())
            
    def test_dim2(self):
        g=nx.Graph()
        g.add_node(1,pos=np.array([1,4,6]))
        g.add_node(2,pos=np.array([2,3,4]))
        g.add_node(3,pos=np.array([4,1,3]))
        g.add_edge(1,2)
        g.add_edge(2,3)
        
        h=snake.cropGraph(g,[slice(0,5),slice(0,5),slice(0,6)])
        self.assertTrue(len(h.nodes)==3)
        self.assertTrue((h.nodes[4]["pos"]==np.array([1.5,3.5,5])).all())
        self.assertTrue((h.nodes[2]["pos"]==g.nodes[2]["pos"]).all())
        self.assertTrue((h.nodes[3]["pos"]==g.nodes[3]["pos"]).all())
        
    def test_lower(self):
        g=nx.Graph()
        g.add_node(1,pos=np.array([1,4,6]))
        g.add_node(2,pos=np.array([2,3,4]))
        g.add_node(3,pos=np.array([4,1,3]))
        g.add_edge(1,2)
        g.add_edge(2,3)
        
        h=snake.cropGraph(g,[slice(0,5),slice(2,5),slice(0,7)])
        self.assertTrue(len(h.nodes)==3)
        self.assertTrue((h.nodes[1]["pos"]==g.nodes[1]["pos"]).all())
        self.assertTrue((h.nodes[2]["pos"]==g.nodes[2]["pos"]).all())
        self.assertTrue((h.nodes[4]["pos"]==np.array([3,2,3.5])).all())
        
    def test_lower_neg(self):
        g=nx.Graph()
        g.add_node(1,pos=np.array([-1,-4,-6]))
        g.add_node(2,pos=np.array([-2,-3,-4]))
        g.add_node(3,pos=np.array([-4,-1,-3]))
        g.add_edge(1,2)
        g.add_edge(2,3)
        
        h=snake.cropGraph(g,[slice(-5,0),slice(-5,0),slice(-5,0)])
        self.assertTrue(len(h.nodes)==3)
        self.assertTrue((h.nodes[4]["pos"]==np.array([-1.5,-3.5,-5])).all())
        self.assertTrue((h.nodes[2]["pos"]==g.nodes[2]["pos"]).all())
        self.assertTrue((h.nodes[3]["pos"]==g.nodes[3]["pos"]).all())
    

def create_X_graph():
    # create an X-shaped graph
    g=nx.Graph()
    g.add_node(0,pos=np.array([0.0,0.0]))
    for sign in [( 1, 1), (-1, 1), (-1,-1), ( 1,-1)]:
        prev=0
        for i in range(4):
            n=len(g)
            g.add_node(n,pos=np.array([sign[0]*float(i+1),sign[1]*float(i+1)]))
            g.add_edge(prev,n)
            prev=n
    return g
        

class test_getA(unittest.TestCase):
    def test_compare(self):
        def createA(G, alpha, beta):
            # old and messy functtion for creating the matrix, for comparison
            A = np.zeros((len(G.nodes), len(G.nodes)))
            edges = np.array(nx.adjacency_matrix(G).todense())
            a = beta
            b = -alpha-4*beta
            c = 2*alpha+6*beta
            e = alpha+5*beta
            f = -2*beta
            ee = 2*alpha+5*beta
            ff = -alpha-2*beta
            zz = alpha+beta
            zzz = alpha
            zzz_= beta
            yy = -alpha-2*beta
            yyy =-alpha
            yyy_=-2*beta
            xx = beta
            for i in range(len(G.nodes)):
                n = G.nodes[i]
                rn = np.sum(edges[i,:])
                if rn == 2:
                    n1 = np.where(edges[i,:] == 1)[0][0]
                    n2 = np.where(edges[i,:] == 1)[0][1]
                    rn1 = np.sum(edges[n1,:])
                    rn2 = np.sum(edges[n2,:])
                    if rn1 == 2 and rn2 == 2:
                        n11 = np.where(edges[n1,:] == 1)[0][np.where(edges[n1,:] == 1)[0] != i]
                        n22 = np.where(edges[n2,:] == 1)[0][np.where(edges[n2,:] == 1)[0] != i]
                        A[i,i] = c
                        A[i,n1] = b
                        A[i,n2] = b
                        A[i,n11] = a
                        A[i,n22] = a
                    elif rn1 != 2 and rn2 == 2:
                        n22 = np.where(edges[n2,:] == 1)[0][np.where(edges[n2,:] == 1)[0] != i]
                        ##A[i,i] = e
                        A[i,i] = ee
                        ##A[i,n1] = f
                        A[i,n1] = ff
                        A[i,n2] = b
                        A[i,n22] = a
                    elif rn1 == 2 and rn2 != 2:
                        n11 = np.where(edges[n1,:] == 1)[0][np.where(edges[n1,:] == 1)[0] != i]
                        ##A[i,i] = e
                        A[i,i] = ee
                        A[i,n1] = b
                        ##A[i,n2] = f
                        A[i,n2] = ff
                        A[i,n11] = a
            #         else:
            #             A[i,i] = 
            #             A[i,n1] = 
            #             A[i,n2] =
        
                elif rn == 1:
                    n1 = np.where(edges[i,:] == 1)[0][0]
                    rn1 = np.sum(edges[n1,:])
                    A[i,i]=zzz
                    A[i,n1]
                    if rn1 == 2:
                        n11 = np.where(edges[n1,:] == 1)[0][np.where(edges[n1,:] == 1)[0] != i]
                        ##A[i,i] = a
                        A[i,i] = zz
                        ##A[i,n1] = f
                        A[i,n1] = yy
                        ##A[i,n11] = a
                        A[i,n11] = xx
                    else:
                        A[i,i] = zzz
                        A[i,n1] = yyy
        
                elif rn > 2:
                    ##ns = np.where(edges[i,:] == 1)[0]
                    ##rns = np.sum(edges[ns,:], axis=1)
                    ##A[i,i] = rn * beta
                    A[i,i]=rn*alpha
                    ##A[i,ns] = f
                    ns = np.where(edges[i,:] == 1)[0]
                    #print(ns)
                    for nn in ns:
                        rns2 = np.sum(edges[nn,:])
                        if rns2==2:
                            A[i,nn] = -alpha-2*beta
                            n11 = np.where(edges[nn,:] == 1)[0][np.where(edges[nn,:] == 1)[0] != i]
                            A[i,n11]= beta
                            A[i,i]+=beta
                        else:
                            A[i,nn] = -alpha
                        #n11 = np.where(edges[nn,:] == 1)[0][np.where(edges[nn,:] == 1)[0] != i]
                        #A[i,n11] = a
            return A

        g=create_X_graph()
        
        # compare the two methods
        A1=createA(g,1,2)
        A2,s0,fd,n2i=snake.getA(g,1,2,2)
        
        self.assertTrue((A1==A2[:,:,0]).all())

    def test_invertA(self):
        g=create_X_graph()
        
        A,s0,fd,n2i=snake.getA(g,1,3,2)
        
        stepsz=0.1
        C=snake.invertALambdaI(A,stepsz)
        
        Cinv=stepsz*A+np.eye(A.shape[0])[:,:,np.newaxis]
        
        D=np.einsum('kld,lmd->kmd',C,Cinv)
        # assert D is close to identity
        self.assertTrue(np.linalg.norm(D-np.eye(A.shape[0])[:,:,np.newaxis])<1e-10)

class test_Snake(unittest.TestCase):
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
    
            def compGrad(s,target):
                g=s-th.tensor([[target]*ndims],dtype=th.double)
                for d in range(ndims):
                    if d==dim: continue
                    g[:,d]=0.0
                return g
                
            s=snake.Snake(G,crop,stepsz,alpha,beta,ndims)
            
            niter=50
            for i in range(niter):
                g=compGrad(s.getPos(),target)
                s.step(g)
    
            p=s.getPos()
            self.assertTrue(th.norm(s.getPos()[:,dim]-target)<1e-3)

        test_(0,2)
        test_(1,2)

        test_(0,3)
        test_(1,3)
        test_(2,3)

if __name__ == '__main__':
    unittest.main()
