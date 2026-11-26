"""QG-020: quantized physical modes of the staggered linearized ADM reference.
Static commuting probe stresses, not dynamical relativistic quantum matter or full GR.
Units hbar=c=a=1; overall source coupling absorbs gravitational normalization.
"""
import json, platform
from pathlib import Path
import numpy as np
import scipy
from scipy.linalg import null_space
from scipy.sparse import diags, eye, kron, block_diag
from scipy.sparse.linalg import expm_multiply
from qg015_reference import pairs,t,lap,Kh,Dreal,curvature,symbols
OUT=Path(__file__).resolve().parent
SIG=np.array([[1,1],[1,-1],[-1,1],[-1,-1]])
OFF=np.array([np.zeros(3) if i==j else (np.eye(3)[i]+np.eye(3)[j])/2 for i,j in pairs])
TOL=2e-10

def maxabs(x):return float(np.max(np.abs(x)))
def neg(r):
    e=np.linalg.eigvalsh(r.reshape(2,2,2,2).transpose(0,3,2,1).reshape(4,4))
    return float(np.maximum(-e,0).sum())

def local_stress(L):
    seed=np.zeros((6,L,L,L));seed[0,0,0,0]=1;seed[1,0,0,0]=-1
    K=Kh(seed);S=(Kh(K)-np.stack([lap(x) for x in K]))/2
    S/=np.linalg.norm(S)
    return S,seed

def lattice(L, explicit=True):
    assert L%2==1 # Avoid special self-conjugate Nyquist sectors; zero mode excluded.
    k=2*np.pi*np.fft.fftfreq(L); V=L**3; S,seed=local_stress(L)
    F=np.fft.fftn(S,axes=(1,2,3))/np.sqrt(V)
    R=[];w=[];couplings=[];coords=[];basis_err=[];rows=[]
    for ix in np.ndindex(L,L,L):
        partner=tuple((-i)%L for i in ix)
        if ix==(0,0,0) or ix>partner:continue
        kk=np.array([k[i] for i in ix]);q=2*np.sin(kk/2);omega=np.linalg.norm(q)
        D,c,K,C=symbols(q);W=null_space(np.vstack([D,t]))
        P=W@W.T;basis_err.append(max(maxabs(W.T@W-np.eye(2)),maxabs(D@W),maxabs(t@W),maxabs(K@W-omega**2*W),maxabs(P-P.T)))
        z=W.T@(F[(slice(None),)+ix]*np.exp(-1j*(OFF@kk)))
        # Two real cosine/sine functions for each of two polarizations.
        for j in range(2):
            for sine in (False,True):
                w.append(omega); coords.append((ix,j,sine));R.append((kk,W[:,j],sine));couplings.append(np.sqrt(2)*(-z[j].imag if sine else z[j].real))
        rows.append(dict(k=kk.tolist(),omega=float(omega),rank=int(np.linalg.matrix_rank(C)),physical_pairs=W.shape[1]))
    w=np.array(w);couplings=np.array(couplings)
    def translate(dx):
        F2=np.fft.fftn(np.roll(S,dx,axis=1),axes=(1,2,3))/np.sqrt(V);res=[]
        for ix,j,sine in coords:
            kk=np.array([k[i] for i in ix]);q=2*np.sin(kk/2);D,_,_,_=symbols(q);W=null_space(np.vstack([D,t]));z=(W.T@(F2[(slice(None),)+ix]*np.exp(-1j*(OFF@kk))))[j]
            res.append(np.sqrt(2)*(-z.imag if sine else z.real))
        return np.array(res)
    if not explicit:return w,couplings,translate,dict(L=L,modes=len(w),mode_residual=max(basis_err))
    # Explicit real-space basis for independently checking Fourier reconstruction.
    X=np.stack(np.meshgrid(*[np.arange(L)]*3,indexing='ij'));U=[]
    for kk,p,sine in R:
        phase=np.einsum('i,ixyz->xyz',kk,X)[None]+(OFF@kk)[:,None,None,None]
        U.append((np.sqrt(2/V)*p[:,None,None,None]*(np.sin(phase) if sine else np.cos(phase))).ravel())
    U=np.stack(U,axis=1); recon=U@couplings
    dcheck=dict(L=L,modes=len(w),expected_modes=2*(V-1),min_omega=float(w.min()),max_omega=float(w.max()),source_divergence=maxabs(Dreal(S)),source_trace=maxabs(t@S.reshape(6,-1)),source_mean=maxabs(S.mean(axis=(1,2,3))),bad_source_divergence=maxabs(Dreal(seed)),source_reconstruction=maxabs(recon-S.ravel()),source_parseval=abs(float(couplings@couplings)-1),mode_residual=max(basis_err),rank_ok=all(r['rank']==4 and r['physical_pairs']==2 for r in rows),orthogonality=maxabs(U.T@U-np.eye(len(w))),local_support=int(np.sum(np.max(np.abs(S),axis=0)>1e-12)))
    # Check real-space TT constraint and equal-time commutator kernel by sampled columns.
    rng=np.random.default_rng(20026+L); idx=rng.choice(len(w),size=8,replace=False); dcheck['real_mode_divergence']=max(maxabs(Dreal(U[:,j].reshape(S.shape))) for j in idx);dcheck['real_mode_curvature']=max(maxabs(curvature(U[:,j].reshape(S.shape))) for j in idx)
    return w,couplings,translate,dcheck

def analytic(w,f,ts,T):
    # Exact forced harmonic-oscillator solution for product thermal state.
    # f shape [4 branches, modes], H_s = sum w a†a + f_s(a+a†).
    ds=f[:,None,:]-f[None,:,:]; sq=f[:,None,:]**2-f[None,:,:]**2
    coth=np.ones_like(w) if T==0 else 1/np.tanh(w/(2*T))
    decay=np.einsum('ijm,tm->tij',ds**2/w**2,(1-np.cos(ts[:,None]*w))*coth)
    phase=np.einsum('ijm,tm->tij',sq/w**2,ts[:,None]*w-np.sin(ts[:,None]*w))
    rho=np.exp(-decay+1j*phase)/4; classical=np.exp(-decay)/4
    alpha=f[None,:,:]/w*(np.exp(-1j*ts[:,None,None]*w)-1)
    energy=np.sum(w*abs(alpha)**2+2*f[None,:,:]*alpha.real,axis=-1)
    return rho,classical,alpha,energy

def sweep():
    grids=[];cases=[];ts=np.linspace(0,40,161)
    for L in (3,5,7,9):
        w,cA,translate,d=lattice(L);grids.append(d)
        for dx in sorted(set([0,1,L//2])):
            cB=translate(dx)
            S,_=local_stress(L); S2=np.roll(S,dx,axis=1)
            overlap=int(np.sum((np.max(abs(S),axis=0)>1e-12)&(np.max(abs(S2),axis=0)>1e-12)))
            static_rate=float(np.sum(cA*cB/w**2)); bracket_error=abs(float(cA@cB-np.vdot(S,S2).real))
            for g in (.2,.5,1.):
                for T in (0.,.5,2.):
                    f=SIG@np.stack([g*cA,g*cB])/np.sqrt(2*w)
                    r,rc,alpha,en=analytic(w,f,ts,T)
                    ns=[neg(x) for x in r];cs=[neg(x) for x in rc]
                    maxidx=int(np.argmax(ns)); ps=np.einsum('tij,tji->t',r,r).real
                    cases.append(dict(L=L,dx=dx,g=g,temperature=T,support_overlap=overlap,static_cross_rate=static_rate,smeared_canonical_bracket_error=bracket_error,max_neg=float(ns[maxidx]),peak_t=float(ts[maxidx]),final_neg=ns[-1],max_classical_neg=max(cs),max_energy_error=maxabs(en),mean_displacement_error=maxabs(alpha.mean(axis=1)),norm_error=maxabs(np.trace(r,axis1=1,axis2=2)-1),min_eig=float(np.min(np.linalg.eigvalsh(r))),coherence_magnitude_error=maxabs(abs(r)-abs(rc)),series=np.column_stack([ts,ns,cs,ps]).tolist()))
        # single-probe and no-coupling controls
        for g in (0.,.5):
            f=SIG@np.stack([g*cA,0*cA])/np.sqrt(2*w);r,rc,_,_=analytic(w,f,ts,0.)
            cases.append(dict(L=L,dx=0,g=g,temperature=0.,control='one_probe' if g else 'zero',max_neg=max(neg(x) for x in r)))
    return grids,cases

def direct():
    # Independent sparse full two-probe / two-oscillator calculation, using two
    # modes of actual L=5 field with greatest joint source weight; vacuum only.
    w,cA,translate,_=lattice(5);cB=translate(1);idx=np.argsort(cA*cA+cB*cB)[-2:];w=w[idx]
    f=SIG@np.stack([.5*cA[idx],.5*cB[idx]])/np.sqrt(2*w)
    ts=np.linspace(0,10,41);r,_,_,_=analytic(w,f,ts,0);rows=[]
    for N in (6,10,16):
        a=diags(np.sqrt(np.arange(1,N)),1,shape=(N,N),format='csr');n=diags(np.arange(N),format='csr');I=eye(N,format='csr');X=a+a.T
        H0=w[0]*kron(n,I)+w[1]*kron(I,n);X1=kron(X,I);X2=kron(I,X)
        H=block_diag([H0+ff[0]*X1+ff[1]*X2 for ff in f],format='csr');v=np.zeros(4*N*N,complex);v[::N*N]=.5
        traj=expm_multiply(-1j*H,v,start=0,stop=10,num=len(ts));R=traj.reshape(len(ts),4,N*N);red=R@R.conj().transpose(0,2,1)
        en=np.array([np.vdot(z,H@z).real for z in traj]);rows.append(dict(N=N,dimension=H.shape[0],max_density_error=maxabs(red-r),norm_error=maxabs(np.sum(abs(traj)**2,axis=1)-1),energy_error=maxabs(en),max_neg=max(neg(x) for x in red)))
    return dict(mode_indices=idx.tolist(),frequencies=w.tolist(),couplings=f.tolist(),samples=len(ts),runs=rows)

def disjoint():
    L=11;w,A,translate,d=lattice(L,explicit=False);S,_=local_stress(L);rows=[];ts=np.linspace(0,100,401)
    for dx in (4,5):
        B=translate(dx);S2=np.roll(S,dx,axis=1);overlap=int(np.sum((np.max(abs(S),axis=0)>1e-12)&(np.max(abs(S2),axis=0)>1e-12)))
        rate=float(np.sum(A*B/w**2));bracket=abs(float(A@B-np.vdot(S,S2).real))
        for g in (.5,1.,2.):
            f=SIG@np.stack([g*A,g*B])/np.sqrt(2*w);r,rc,_,_=analytic(w,f,ts,0.)
            ns=[neg(x) for x in r];rows.append(dict(L=L,dx=dx,g=g,support_overlap=overlap,static_cross_rate=rate,smeared_canonical_bracket_error=bracket,max_neg=max(ns),peak_t=float(ts[np.argmax(ns)]),samples=len(ts),series=np.column_stack([ts,ns]).tolist()))
    return rows

def main():
    grids,cases=sweep();dr=direct();dj=disjoint();valid=[x for x in cases if 'control' not in x];controls=[x for x in cases if 'control' in x];fin=dr['runs'][-1]
    keys=['source_divergence','source_trace','source_mean','source_reconstruction','source_parseval','mode_residual','orthogonality','real_mode_divergence','real_mode_curvature'];summary={k:max(x[k] for x in grids) for k in keys};summary.update({k:max(x[k] for x in valid) for k in ['max_classical_neg','max_energy_error','mean_displacement_error','norm_error','coherence_magnitude_error']});summary['min_eig']=min(x['min_eig'] for x in valid);summary['max_neg']=max(x['max_neg'] for x in valid);summary['max_separated_neg']=max(x['max_neg'] for x in valid if x['dx']>0);summary['max_control_neg']=max(x['max_neg'] for x in controls)
    checks=dict(lattice_constraints=max(summary[k] for k in keys)<TOL,mode_count=all(x['modes']==x['expected_modes'] and x['rank_ok'] for x in grids),bad_source_detected=min(x['bad_source_divergence'] for x in grids)>1e-3,unit_trace=summary['norm_error']<TOL,positive=summary['min_eig']>-TOL,energy=summary['max_energy_error']<TOL,zero_mean=summary['mean_displacement_error']<TOL,classical_separable=summary['max_classical_neg']<TOL,matched_coherence_magnitude=summary['coherence_magnitude_error']<TOL,null_controls=summary['max_control_neg']<TOL,entanglement=summary['max_neg']>.05,offset_center_entanglement=summary['max_separated_neg']>.01,direct_density=fin['max_density_error']<1e-7,direct_norm=fin['norm_error']<TOL,direct_energy=fin['energy_error']<TOL,smeared_canonical_bracket=max(x['smeared_canonical_bracket_error'] for x in valid+dj)<TOL,disjoint_control=all(x['support_overlap']==0 for x in dj if x['dx']==5))
    out=dict(status='PASS_BOUNDED_QUANTIZED_LINEAR_TENSOR_REFERENCE' if all(checks.values()) else 'FAIL_OR_UNRESOLVED',environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),summary=summary,checks=checks,grids=grids,cases=cases,direct=dr,disjoint=dj)
    (OUT/'results.json').write_text(json.dumps(out,separators=(',',':'))+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ('cases','disjoint')},indent=2));print('cases',len(valid),'controls',len(controls),'samples',len(valid)*161);print('peak separated', {k:v for k,v in max((x for x in valid if x['dx']>0),key=lambda x:x['max_neg']).items() if k!='series'});print('large-grid disjoint controls',json.dumps([{k:v for k,v in x.items() if k!='series'} for x in dj]));assert all(checks.values()),checks
if __name__=='__main__':main()
