#!/usr/bin/env python3
"""QG-014. Local linear rank-2 controls and independent finite Z3 charge model."""
import json, sys, hashlib
from pathlib import Path
import numpy as np
import scipy
from scipy.linalg import null_space
from scipy.sparse import coo_matrix, diags, eye, kron
from scipy.sparse.linalg import eigsh, expm_multiply
OUT=Path(__file__).resolve().parent
rng=np.random.default_rng(1414)
I=np.eye(3)
BAS=np.array([np.diag([1,0,0]),np.diag([0,1,0]),np.diag([0,0,1]),(np.outer(I[0],I[1])+np.outer(I[1],I[0]))/np.sqrt(2),(np.outer(I[0],I[2])+np.outer(I[2],I[0]))/np.sqrt(2),(np.outer(I[1],I[2])+np.outer(I[2],I[1]))/np.sqrt(2)])
t=np.einsum('aii->a',BAS)
vec=lambda T: np.einsum('aij,ij->a',BAS,T)
def op(f): return np.column_stack([vec(f(X)) for X in BAS])
def cross(q): return np.cross(q,I).T

def matrices(q):
    r=q@q; X=cross(q); D=np.einsum('i,aij->ja',q,BAS); s=np.einsum('i,aij,j->a',q,BAS,q)
    P=I-np.outer(q,q)/r
    TT=op(lambda A:P@A@P-P*np.trace(P@A)/2)
    B=op(lambda A:X@A@X.T)
    # Linearized Ricci, then Schouten, then symmetrized curl; signs/i irrelevant to B^*B.
    def cotton(A):
        R=(r*A-np.outer(q,q@A)-np.outer(A@q,q)+np.outer(q,q)*np.trace(A))/2
        S=R-I*np.trace(R)/4
        return X@S-S@X # factor two gives unit TT amplitude |q|^3
    C=op(cotton)
    M=np.stack([np.eye(6)*r, D.T@D, np.outer(s,t)+np.outer(t,s),r*np.outer(t,t)])
    return D,s,TT,B,C,M

def maxabs(x): return float(np.max(np.abs(x)))

def mode_lab():
    rows=[]; eq=[]; err={k:0. for k in ['vector_gauge','cotton_gauge','quartic_gauge','cotton_divergence','cotton_trace','quartic_identity','cotton_identity','constraint_flow','helicity_rotation']}
    neg=[]
    dirs=np.vstack([I,[[1,1,1]],rng.normal(size=(8,3))]);dirs=dirs/np.linalg.norm(dirs,axis=1)[:,None]
    for n in dirs:
      for k in [.015625,.03125,.0625,.125,.25,.5]:
        q=k*n; D,s,TT,B,C,M=matrices(q); T=np.vstack([D,t]); U=null_space(D); W=null_space(T)
        K2=B.T@B; K3=C.T@C; K4=B.T@(np.eye(6)-np.outer(t,t)/2)@B
        V=D.T; Z=T.T
        models={'vector_double_curl':(U,K2),'traceless_cotton':(W,K3),'traceless_quartic':(W,K4),'nonlocal_relativistic_control':(W,k*k*TT)}
        vals={name:np.sqrt(np.maximum(0,np.linalg.eigvalsh(Q.T@K@Q))).tolist() for name,(Q,K) in models.items()}
        rows.append(dict(n=n.tolist(),k=k,frequencies=vals))
        checks={'vector_gauge':K2@V,'cotton_gauge':C@Z,'quartic_gauge':K4@Z,'cotton_divergence':D@C,'cotton_trace':t@C,'quartic_identity':K4-k**4*TT,'cotton_identity':K3-k**6*TT,'constraint_flow':T@K4@W}
        for name,v in checks.items():err[name]=max(err[name],maxabs(v))
        # Compare gauge-invariance equations at unit momentum, avoiding scale bias.
        _,_,_,_,_,A=matrices(n); Dn=matrices(n)[0]; Zn=np.vstack([Dn,t]).T
        eq.extend(np.stack([(m@Zn).ravel() for m in A],axis=1))
        neg.append(maxabs(T@(K4+.1*np.eye(6))@Z))
        # Rotation around n: explicitly compare helicity-two real basis.
        e=I[np.argmin(abs(n))];u=np.cross(n,e);u/=np.linalg.norm(u);v=np.cross(n,u)
        plus=(np.outer(u,u)-np.outer(v,v))/np.sqrt(2); times=(np.outer(u,v)+np.outer(v,u))/np.sqrt(2)
        theta=.37; ur=np.cos(theta)*u+np.sin(theta)*v;vr=-np.sin(theta)*u+np.cos(theta)*v
        err['helicity_rotation']=max(err['helicity_rotation'],maxabs((np.outer(ur,ur)-np.outer(vr,vr))/np.sqrt(2)-np.cos(2*theta)*plus-np.sin(2*theta)*times))
    slopes={}
    for name in rows[0]['frequencies']:
        slopes[name]=np.polyfit(np.log([r['k'] for r in rows[:6]]),np.log([r['frequencies'][name][0] for r in rows[:6]]),1)[0].item()
    S=np.linalg.svd(np.array(eq),compute_uv=False)
    # Actual local three-dimensional periodic stencils, not just Fourier tests.
    def deriv(x,i): return (np.roll(x,-1,axis=i)-np.roll(x,1,axis=i))/2
    def grad(v): return np.stack([deriv(v,i) for i in range(3)],axis=-2)
    def curl_left(a): return np.stack([deriv(a[...,2,:],1)-deriv(a[...,1,:],2),deriv(a[...,0,:],2)-deriv(a[...,2,:],0),deriv(a[...,1,:],0)-deriv(a[...,0,:],1)],axis=-2)
    def curl2(a): return curl_left(curl_left(a).swapaxes(-1,-2)).swapaxes(-1,-2)
    def cotton_field(a):
        div=np.stack([sum(deriv(a[...,i,j],i) for i in range(3)) for j in range(3)],axis=-1)
        g=grad(div);tr=np.trace(a,axis1=-2,axis2=-1)
        hess=grad(np.stack([deriv(tr,i) for i in range(3)],axis=-1))
        lap=sum(deriv(deriv(a,i),i) for i in range(3));R=(g+g.swapaxes(-1,-2)-lap-hess)/2
        S=R-np.trace(R,axis1=-2,axis2=-1)[...,None,None]*I/4
        b=curl_left(S);return b+b.swapaxes(-1,-2)
    real=[]
    for L in [5,7,9,8]:
        shape=(L,)*3; a=rng.normal(size=shape+(3,3));a=(a+a.swapaxes(-1,-2))/2
        g=grad(rng.normal(size=shape+(3,)));g=(g+g.swapaxes(-1,-2))/2
        w=rng.normal(size=shape)[...,None,None]*I
        b=curl2(a);b1=curl2(a+g+w)
        energy=lambda b:float(np.sum(b*b)-.5*np.sum(np.trace(b,axis1=-1,axis2=-2)**2))
        # Central difference has extra zeros on even grids.
        q=np.sin(2*np.pi*np.arange(L)/L);zeros=int(np.sum(abs(q)<1e-12))**3
        real.append(dict(L=L,zero_momentum_symbols=zeros,vector_double_curl_error=maxabs(curl2(a+g)-b),cotton_error=maxabs(cotton_field(a+g+w)-cotton_field(a)),quartic_energy_error=abs(energy(b1)-energy(b))/max(1,abs(energy(b))),double_curl_weyl_break=maxabs(b1-b)))
    assert max(err.values())<1e-10 and min(neg)>.01 and S[-1]>1e-4
    assert all(max(r['vector_double_curl_error'],r['cotton_error'],r['quartic_energy_error'])<1e-10 for r in real)
    return dict(mixed_cotton_quartic=[dict(k=k,omega=float(np.sqrt(k**6+.1*k**4)),local_log_slope=float(2+k*k/(k*k+.1))) for k in [.001,.01,.1,.5,1.]],momentum_rows=rows,max_residuals=err,dispersion_exponents=slopes,quadratic_local_ansatz_singular_values=S.tolist(),quadratic_ansatz_rank=int(np.linalg.matrix_rank(eq)),mass_term_gauge_violation_range=[min(neg),max(neg)],realspace=real)

def rankmod(A,p=3):
    A=np.array(A,dtype=int)%p;r=0
    for j in range(A.shape[1]):
        ix=np.where(A[r:,j]!=0)[0]
        if not len(ix):continue
        i=r+ix[0];A[[r,i]]=A[[i,r]];A[r]=A[r]*pow(int(A[r,j]),-1,p)%p
        for i in range(A.shape[0]):
            if i!=r:A[i]=(A[i]-A[i,j]*A[r])%p
        r+=1
        if r==A.shape[0]:break
    return r

def quantum_lab():
    # Finite Z3, 3-site ring, E=(xx,yy,zz,xy,xz,yz). Forward divergence of E_zj.
    N=3;p=3;G=np.zeros((12,18),int)
    for i in range(N):
        for j,c in enumerate([4,5,2]):G[3*i+j,6*((i+1)%N)+c]=1;G[3*i+j,6*i+c]=-1
        G[9+i,6*i:6*i+3]=1
    moves=[]
    for i in range(3):
        a=np.zeros(18,int);a[6*i]=1;a[6*i+1]=-1;moves.append(a)
        b=np.zeros(18,int);b[6*i+3]=1;moves.append(b)
    r=rankmod(G);assert r==9 and all(np.all(G@m%3==0) for m in moves)
    bad=np.zeros(18,int);bad[0]=1;assert np.any(G@bad%3)
    # All allowed states: 3 global flux variables plus xx and xy at each site.
    digits=np.indices((3,)*9).reshape(9,-1).T;E=np.zeros((len(digits),3,6),int)
    E[:,:,4]=digits[:,0,None];E[:,:,5]=digits[:,1,None];E[:,:,2]=digits[:,2,None]
    E[:,:,0]=digits[:,3:6];E[:,:,3]=digits[:,6:9];E[:,:,1]=(-E[:,:,0]-E[:,:,2])%3
    assert np.all((E.reshape(-1,18)@G.T)%3==0)
    # Solve the zero-global-flux block; six independent qutrits.
    d=np.indices((3,)*6).reshape(6,-1).T;dim=len(d);idx=np.arange(dim);base=3**np.arange(5,-1,-1)
    def shift(v):
        dest=((d+v)%3)@base
        return coo_matrix((np.ones(dim),(dest,idx)),shape=(dim,dim)).tocsr()
    unit=np.eye(6,dtype=int);cost=np.where(d==0,0.,1.)
    H=diags(.4*np.sum(cost[:,:3]*2+cost[:,3:],axis=1)).tocsr()
    for v in unit:
        S=shift(v);H=H-.23*(S+S.T)
    # Local nearest-neighbor invariant shifts.
    H1=H.copy()
    for i in range(3):
        S=shift(unit[i]+unit[(i+1)%3]);H1=H1-.12*(S+S.T)
    e0,gs=eigsh(H,k=6,which='SA',v0=np.random.default_rng(34).normal(size=dim),ncv=40);e0=np.sort(e0)
    ev,gv=eigsh(H1,k=6,which='SA',v0=np.random.default_rng(34).normal(size=dim),ncv=40);order=np.argsort(ev);ev=ev[order];psi=gv[:,order[0]]
    # Charged blocks use identical H1 (deliberate model choice), trace charge r at site 0.
    # V shifts trace charge by ±1, with amplitude depending on a physical qutrit.
    Z=coo_matrix((np.ones(3),((np.arange(3)+1)%3,np.arange(3))),shape=(3,3)).tocsr()
    f=1+.5*(d[:,3]==0);F=diags(f);eps=.18
    V=eps*kron(Z+Z.T,F,format='csr');Hbase=kron(eye(3),H1,format='csr');P=diags(np.repeat([1.,0.,0.],dim));Q=eye(3*dim)-P
    comm=lambda X: maxabs((P@X-X@P).data) if (P@X-X@P).nnz else 0.
    times=np.linspace(0,20,101);init=np.r_[psi,np.zeros(2*dim)];rows=[]
    invariant_traj=expm_multiply(-1j*Hbase,init,start=0,stop=20,num=len(times))
    invariant_leakage=maxabs(np.sum(abs(invariant_traj[:,dim:])**2,axis=1))
    assert invariant_leakage==0
    for gap in [0.,4.,16.,64.]:
        full=Hbase+gap*Q+V
        evfull=np.sort(eigsh(full,k=6,which='SA',v0=np.random.default_rng(34).normal(size=3*dim),ncv=40,return_eigenvectors=False))
        traj=expm_multiply(-1j*full,init,start=0,stop=20,num=len(times));norm=np.sum(abs(traj)**2,axis=1);leak=np.sum(abs(traj[:,dim:])**2,axis=1)
        Etime=np.einsum('ti,it->t',traj.conj(),full@traj.T).real
        row=dict(gap=gap,energies=evfull.tolist(),max_leakage=float(max(leak)),norm_error=maxabs(norm-1),energy_error=maxabs(Etime-Etime[0]),leakage=leak.tolist())
        if gap:
            eff=H1-diags(2*eps**2*f*f/gap);ee=np.sort(eigsh(eff,k=6,which='SA',v0=np.random.default_rng(34).normal(size=dim),ncv=40,return_eigenvectors=False));row['effective_energies']=ee.tolist();row['bare_spectral_error']=maxabs(evfull-ev);row['corrected_spectral_error']=maxabs(evfull-ee)
        rows.append(row)
    assert comm(Hbase)==0 and comm(V)>0 and max(x['norm_error'] for x in rows)<1e-9
    assert rows[-1]['max_leakage']<rows[-2]['max_leakage']<rows[1]['max_leakage']
    assert rows[-1]['corrected_spectral_error']<rows[-1]['bare_spectral_error']
    return dict(modulus=3,sites=3,gauss_rank=r,physical_dimension=3**(18-r),states_enumerated=len(E),zero_flux_dimension=dim,extended_dimension=3*dim,invariant_moves=len(moves),invariant_projector_commutator=comm(Hbase),invariant_max_leakage=invariant_leakage,breaking_projector_commutator=comm(V),bare_eigenvalues=e0.tolist(),invariant_interaction_eigenvalues=ev.tolist(),time=times.tolist(),gap_results=rows)
if __name__=='__main__':
    result={'experiment':'QG-014','python':sys.version.split()[0],'numpy':np.__version__,'scipy':scipy.__version__,'seed':1414,'linear':mode_lab(),'quantum':quantum_lab()}
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print('QG-014 assertions PASS');print('linear:',json.dumps({k:v for k,v in result['linear'].items() if k!='momentum_rows'},indent=2));print('quantum:',json.dumps({k:v for k,v in result['quantum'].items() if k not in ['time','gap_results']},indent=2));print('gap:',json.dumps([{k:v for k,v in x.items() if k!='leakage'} for x in result['quantum']['gap_results']],indent=2))
