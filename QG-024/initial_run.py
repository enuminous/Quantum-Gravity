"""QG-024: free linearized curvature microcausality, normalized units."""
import json, sys, platform
from pathlib import Path
import numpy as np
import scipy
from scipy.special import spherical_jn
from scipy.integrate import simpson
ROOT=Path(__file__).resolve().parent
L=16.; A=1.; DISTS=[2.5,3.,4.]; CUTS=[16,32,48,64,96]; TIMES=np.linspace(0,7,281)
ALPHA=BETA=.5

def ball(x):
    out=np.ones_like(x); mask=x!=0
    out[mask]=3*spherical_jn(1,x[mask])/x[mask]
    return out

def shells(N):
    # Spherical Fourier cutoff, summed over all integer vectors. k=0 excluded.
    a=np.arange(-N,N+1); xx,yy=np.meshgrid(a,a,indexing='ij'); xy=xx*xx+yy*yy
    r=np.arange(N*N+1); k=2*np.pi/L*np.sqrt(r); F=ball(k*A/4)**4
    f=np.zeros_like(k); f[1:]=F[1:]**2/k[1:]/L**3
    out=np.zeros((len(DISTS)+1,2,len(k)))
    count=0
    for z in range(N+1):
        r2=xy+z*z; mask=(r2<=N*N)&(r2>0); ids=r2[mask]; nx=xx[mask]
        mult=1 if z==0 else 2; count+=mult*len(ids)
        # Lambda_xxxx = (1 - n_x^2)^2 / 2 for orthonormal TT polarizations.
        w=mult*.5*(1-nx*nx/ids)**2*f[ids]
        h=np.bincount(ids,weights=w,minlength=len(k)); e=h*k**4/4
        phase=np.array([1]+[np.cos(2*np.pi*z*d/L) for d in DISTS])
        out[:,0,:]+=phase[:,None]*e;out[:,1,:]+=phase[:,None]*h
    return k,out,count

def records(N):
    k,w,count=shells(N); V=w[0].sum(axis=1)/2
    sin=np.sin(k[:,None]*TIMES); cos=np.cos(k[:,None]*TIMES)
    rows=[]
    for i,d in enumerate(DISTS,1):
        entry={'distance':d,'first_possible_arrival':d-2*A,'last_direct_arrival':d+2*A}
        for j,label in enumerate(['curvature','bare_TT']):
            c=w[i,j]@sin/V[j]; cov=w[i,j]@cos/(2*V[j]); signal=np.exp(-BETA**2/2)*np.abs(np.sin(ALPHA*BETA*c))
            space=TIMES<=d-2*A-.1; active=(TIMES>=d-2*A)&(TIMES<=d+2*A)
            after=TIMES>=d+2*A+.1
            entry[label]={'max_spacelike_commutator':float(np.max(np.abs(c[space]))), 'max_spacelike_signal':float(np.max(signal[space])), 'max_direct_signal':float(np.max(signal[active])), 'max_postpulse_commutator':float(np.max(np.abs(c[after]))), 'equal_time_covariance':float(cov[0]),'commutator':c.tolist(),'covariance':cov.tolist(),'signal':signal.tolist()}
        rows.append(entry)
    return {'cutoff':N,'wavevectors':count,'vacuum_variance':V.tolist(),'rows':rows}

def infinite_space():
    # Independent angular integral of TT projector for separation along z.
    # phi-average (1-kx^2/k^2)^2 = 3/8 + u^2/4 + 3u^4/8.
    # Integral against cos(kr u) = 2*(8/15 j0 - 8/21 j2 + 3/35 j4).
    k=np.linspace(0,160,32001); F=ball(k*A/4)**4
    w=k**5*F**2/(16*np.pi**2) # k^2/(2pi^2) * k^3/4 * 1/2
    V=simpson(w*(8/15),x=k)/2
    out=[]
    for d in DISTS:
        angular=8/15*spherical_jn(0,k*d)-8/21*spherical_jn(2,k*d)+3/35*spherical_jn(4,k*d)
        values=[]
        for t in [0.,.25,.5,1.,2.,3.,4.,5.,6.]:
            c=simpson(w*angular*np.sin(k*t),x=k)/V; cov=simpson(w*angular*np.cos(k*t),x=k)/(2*V)
            values.append({'time':t,'commutator':float(c),'covariance':float(cov)})
        out.append({'distance':d,'samples':values})
    return {'variance':float(V),'rows':out}

def projector_tests():
    rng=np.random.default_rng(24001); residual=[]; ward=[]; trace=[]; rank=[]; polynomial=[]
    for k in rng.normal(size=(128,3)):
        k2=k@k; P=np.eye(3)-np.outer(k,k)/k2
        B=np.einsum('ik,jl->ijkl',P,P)/2+np.einsum('il,jk->ijkl',P,P)/2-np.einsum('ij,kl->ijkl',P,P)/2
        b=B.reshape(9,9);residual.append(np.max(np.abs(b@b-b))); rank.append(np.linalg.matrix_rank(b,tol=1e-10)); ward.append(np.max(np.abs(np.einsum('i,ijkl->jkl',k,B))));trace.append(np.max(np.abs(np.einsum('iikl->kl',B))))
        polynomial.append(abs(k2*k2*B[0,0,0,0]-.5*(k2-k[0]**2)**2))
    return {'count':len(rank),'ranks':sorted(set(rank)),'max_idempotence':float(max(residual)),'max_transverse':float(max(ward)),'max_trace':float(max(trace)),'max_polynomial':float(max(polynomial))}

def main():
    data=[records(n) for n in CUTS]; inf=infinite_space(); p=projector_tests(); fine=data[-1]
    # Independent infinite-volume commutator compared away from periodic images.
    cmp=[]; infspace=[]
    for row,ref in zip(fine['rows'],inf['rows']):
        for v in ref['samples']:
            idx=int(round(v['time']/(TIMES[1]-TIMES[0])));cmp.append(abs(row['curvature']['commutator'][idx]-v['commutator']))
            if v['time']<row['first_possible_arrival']-.05: infspace.append(abs(v['commutator']))
    maxspace=max(r['curvature']['max_spacelike_commutator'] for r in fine['rows']);maxsignal=max(r['curvature']['max_spacelike_signal'] for r in fine['rows']);maxafter=max(r['curvature']['max_postpulse_commutator'] for r in fine['rows']); minactive=min(r['curvature']['max_direct_signal'] for r in fine['rows']); mincov=min(abs(r['curvature']['equal_time_covariance']) for r in fine['rows']); minbad=min(r['bare_TT']['max_spacelike_signal'] for r in fine['rows']); maxcoarse=max(r['curvature']['max_spacelike_commutator'] for r in data[0]['rows']);
    checks={'TT_rank_two':p['ranks']==[2], 'TT_projector':p['max_idempotence']<1e-12 and p['max_transverse']<1e-12 and p['max_trace']<1e-12, 'local_polynomial':p['max_polynomial']<1e-10, 'spacelike_commutator':maxspace<1e-7, 'spacelike_signal':maxsignal<1e-7, 'cutoff_refinement':maxspace<maxcoarse/100, 'causal_positive_control':minactive>1e-5, 'nonzero_spacelike_covariance':mincov>1e-7, 'nonlocal_TT_control':minbad>1e-4, 'postpulse_Huygens':maxafter<1e-7, 'independent_integral_spacelike':max(infspace)<1e-8, 'independent_integral_agreement':max(cmp)<1e-6}
    summary={'max_spacelike_commutator':maxspace,'max_spacelike_signal':maxsignal,'max_postpulse_commutator':maxafter,'min_direct_signal':minactive,'min_equal_time_covariance_abs':mincov,'min_nonlocal_control_signal':minbad,'coarse_spacelike_commutator':maxcoarse,'max_independent_integral_difference':max(cmp),'max_independent_spacelike':max(infspace)}
    result={'scope':'Free linearized gravity; continuum periodic box with spherical Fourier regulator; no nonlinear constraints or dynamical matter.', 'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},'parameters':{'box_length':L,'support_radius':A,'smearing':'four-fold convolution of normalized balls radius A/4','alpha':ALPHA,'beta':BETA,'times':TIMES.tolist()},'projector':p,'sweep':data,'independent_infinite_volume':inf,'summary':summary,'checks':{k:bool(v) for k,v in checks.items()}}
    (ROOT/'results.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
    print(json.dumps({'summary':summary,'checks':result['checks'],'refinement':[{'cutoff':s['cutoff'],'wavevectors':s['wavevectors'],'max_spacelike':max(r['curvature']['max_spacelike_commutator'] for r in s['rows'])} for s in data],'finest':[{'distance':r['distance'],**{j:{k:v for k,v in r[j].items() if not isinstance(v,list)} for j in ['curvature','bare_TT']}} for r in fine['rows']]},indent=2))
    if not all(checks.values()):sys.exit(1)
if __name__=='__main__':main()
