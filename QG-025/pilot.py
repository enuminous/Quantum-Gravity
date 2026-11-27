import numpy as np
from itertools import permutations
eta=np.diag([1.,-1.,-1.,-1.])
def dot(a,b):return a@eta@b
def outer(a,b):return np.outer(eta@a,eta@b)
def rest(m,w,theta):
    c,s=np.cos(theta),np.sin(theta); wp=w/(1+w/m*(1-c)); p=np.array([m,0,0,0.]); k=w*np.array([1,0,0,1.]); l=wp*np.array([1,s,0,c]); P=p+k-l
    e=(np.array([0,1,0,0.])+1j*np.array([0,0,1,0.]))/np.sqrt(2); f=(np.array([0,c,0,-s])-1j*np.array([0,0,1,0.]))/np.sqrt(2)
    return p,P,k,l,e,f

def bracket(p,P,k,l,e,f):
    a,b=dot(p,k),dot(p,l)
    return dot(p,e)*dot(P,f)/a-dot(P,e)*dot(p,f)/b-dot(e,f)
def factorized(p,P,k,l,e,f):
    a,b,c=dot(p,k),dot(p,l),dot(k,l)
    return .5*a*b/c*bracket(p,P,k,l,e,f)**2

def scalar_vertex(p,P,h,m):
    return -p@h@P + .5*np.trace(eta@h)*(dot(p,P)-m*m)

def cubic(hs,qs):
    inv=[-eta@h@eta for h in hs]; dens=[inv[i]+.5*np.trace(eta@h)*eta for i,h in enumerate(hs)]
    Cs=[]
    for h,q in zip(hs,qs):
        d=1j*eta@q
        # C[d,m,n] = (partial_m h_dn + partial_n h_dm - partial_d h_mn)/2
        C=.5*(np.einsum('m,dn->dmn',d,h)+np.einsum('n,dm->dmn',d,h)-np.einsum('d,mn->dmn',d,h))
        Cs.append(C)
    G=[np.einsum('ad,dmn->amn',eta,C) for C in Cs]
    def pair(g,A,B):
        return np.einsum('mn,amb,bna->',g,A,B)-np.einsum('mn,amn,bab->',g,A,B)
    val=0j
    for a,b,c in permutations(range(3)):
        G2=np.einsum('ad,dmn->amn',inv[a],Cs[b])
        val+=pair(dens[a],G[b],G[c])+pair(eta,G2,G[c])+pair(eta,G[c],G2)
    return -2*val

def diagrams(p,P,k,l,e,f,m):
    h1=outer(e,e);h2=outer(f,f)
    a,b,c=dot(p,k),dot(p,l),dot(k,l)
    sv=-scalar_vertex(p,p+k,h1,m)*scalar_vertex(p+k,P,h2,m)/(2*a)
    uv=-scalar_vertex(p,p-l,h2,m)*scalar_vertex(p-l,P,h1,m)/(-2*b)
    # determinant and densitized inverse cross coefficient for arbitrary tensors
    tr1,tr2=np.trace(eta@h1),np.trace(eta@h2)
    det=.25*tr1*tr2-.5*np.trace(eta@h1@eta@h2)
    C=eta@h1@eta@h2@eta+eta@h2@eta@h1@eta-.5*tr1*eta@h2@eta-.5*tr2*eta@h1@eta+det*eta
    contact=(eta@p)@C@(eta@P)-det*m*m
    V=-.5*(np.outer(p,P)+np.outer(P,p)-eta*(dot(p,P)-m*m))
    h=eta@V@eta-.5*eta*np.trace(eta@V)
    tv=-cubic([h1,h2,h],[k,-l,l-k])/(-2*c)
    return np.array([sv,uv,contact,tv])

if __name__=='__main__':
    for theta in [.5,1.5,2.8]:
        m=1.3;p,P,k,l,e,f=rest(m,.7,theta)
        for f0 in [f,f.conj()]:
            d=diagrams(p,P,k,l,e,f0,m);F=factorized(p,P,k,l,e,f0);print(theta,d,d.sum(),F,d.sum()/F)
            print('gauge',diagrams(p,P,k,l,k.astype(complex),f0,m).sum(),diagrams(p,P,k,l,e,f0+l,m).sum()-d.sum())
