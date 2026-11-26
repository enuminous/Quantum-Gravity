"""QG-016: conserved, dynamical test matter on QG-015's staggered linear lattice.
One-way linearized forcing; not an interacting Einstein-matter or quantum theory.
"""
import json, platform
from pathlib import Path
import numpy as np
import scipy
from scipy.linalg import eigh
from qg015_reference import (B,t,pairs,lap,grad,div,tr,lift,Dreal,Dadj,curvature,Kh,Kp,symbols,maxabs)
OUT=Path(__file__).resolve().parent
SEED=16026
TOL=2e-10
# Source strength absorbs 16 pi G. Units c=1, lattice spacing=1.
KAPPA=.1
M_EL=np.diag([.5,.4,.36,.2,.24,.16]);M_EL[:3,:3]=[[.5,.12,.18],[.12,.4,.04],[.18,.04,.36]]
M_AC=.25*np.outer(t,t)
MODELS={'elastic':M_EL,'acoustic':M_AC}
OFF=np.array([np.zeros(3) if i==j else (np.eye(3)[i]+np.eye(3)[j])/2 for i,j in pairs])
EOFF=np.eye(3)/2

def stress(M,u):return -np.einsum('ab,bxyz->axyz',M,Dadj(u),optimize=False)
# All mixed entries in M are vertex-centered normal components; shear blocks are diagonal.
def rhs(z,M,kappa,arm='conserved'):
    h,p,u,j,rho=z
    S=stress(M,u)
    SG=S if arm!='omit_gravity_stress' else 0*S
    dr=div(j) if arm!='freeze_density' else 0*rho
    return (Kp(p),-Kh(h)+kappa*SG,j,Dreal(S),dr)

def add(z,d,s):return tuple(a+s*b for a,b in zip(z,d))
def step(z,dt,M,kappa,arm):
    a=rhs(z,M,kappa,arm);b=rhs(add(z,a,dt/2),M,kappa,arm);c=rhs(add(z,b,dt/2),M,kappa,arm);d=rhs(add(z,c,dt),M,kappa,arm)
    return tuple(x+dt*(a0+2*b0+2*c0+d0)/6 for x,a0,b0,c0,d0 in zip(z,a,b,c,d))

def constraints(z,kappa):
    h,p,u,j,rho=z
    return curvature(h)-kappa*rho,Dreal(p)-kappa*j

def energy(z,M):
    _,_,u,j,_=z
    return float((np.vdot(j,j)-np.vdot(Dadj(u),stress(M,u))).real/2)

def tt(q):
    n=q/np.linalg.norm(q);axis=np.eye(3)[np.argmin(abs(n))];e=axis-n*(n@axis);e/=np.linalg.norm(e);v=np.cross(n,e)
    pol=np.stack([(np.outer(e,e)-np.outer(v,v))/np.sqrt(2),(np.outer(e,v)+np.outer(v,e))/np.sqrt(2)])
    return np.einsum('aij,bij->ab',B,pol)

def plane(L,d,M,amplitude=1e-3):
    k=2*np.pi*np.array(d)/L;q=2*np.sin(k/2);w=np.linalg.norm(q);D,_,_,_=symbols(q)
    vals,vec=eigh(D@M@D.T);i=int(np.argmax(abs(vec.T@q)));v=vec[:,i];v*=np.sign(v@q);om=np.sqrt(vals[i]);a=amplitude/w
    X=np.stack(np.meshgrid(*[np.arange(L)]*3,indexing='ij'));phase=np.einsum('i,ixyz->xyz',k,X)
    u=a*v[:,None,None,None]*np.sin(phase[None]+(EOFF@k)[:,None,None,None]);j=np.zeros_like(u);rho=div(u)
    # For h=a_metric I, R^(1)=-2 lap(a_metric); rho has only the chosen Fourier mode.
    h=lift(KAPPA*rho/(2*w*w));p=np.zeros_like(h)
    W=tt(q);src=W.T@(M@D.T@v*a)
    def pol(x):
        F=np.fft.fftn(x,axes=(1,2,3))[(slice(None),)+tuple(d)]*np.exp(-1j*(OFF@k))*2/L**3
        return (W.T@F).real
    def exact(time,kappa=KAPPA):
        # Zero initial TT metric/momentum; forced harmonic solution.
        return kappa*src*(np.cos(om*time)-np.cos(w*time))/(w*w-om*om)
    return (h,p,u,j,rho),dict(k=k.tolist(),q=q.tolist(),omega_g=w,omega_m=float(om),wave_speed=float(om/w),matter_vector=v.tolist(),TT_stress=src.tolist(),initial_density_amplitude=float(a*(q@v))),pol,exact

def propagate(L,d,model,dt,arm='conserved',gauge=False,kappa=KAPPA,amp=1e-3):
    M=MODELS[model];z,meta,pol,exact=plane(L,d,M,amp)
    if kappa!=KAPPA:z=(z[0]*kappa/KAPPA,*z[1:])
    if gauge:
        rng=np.random.default_rng(SEED+L);v=1e-5*rng.normal(size=(3,L,L,L));a=1e-5*rng.normal(size=(L,L,L));z=(z[0]+Dadj(v),z[1]-Dadj(grad(a))-lift(lap(a)),*z[2:])
    tmax=8.;N=round(tmax/dt);E0=energy(z,M);sc=max(maxabs(kappa*z[4]),1e-30)
    mx={k:0. for k in ['scalar_constraint','momentum_constraint','scalar_identity','momentum_identity','matter_energy_relative','matter_density_displacement','tt_error','tt_amplitude','metric_amplitude','density_mean','matter_continuity','matter_momentum_identity']};rows=[]
    for i in range(N+1):
        time=i*dt;h,p,u,j,rho=z;C,V=constraints(z,kappa);der=rhs(z,M,kappa,arm);S=stress(M,u);obs=pol(h);ex=exact(time,kappa)
        # Identity defects agree with the independently measured conservation/coupling defects.
        cs=curvature(der[0])-kappa*der[4]-div(V);vs=Dreal(der[1])-kappa*der[3]
        vals=dict(scalar_constraint=maxabs(C),momentum_constraint=maxabs(V),scalar_identity=maxabs(cs),momentum_identity=maxabs(vs),matter_energy_relative=abs(energy(z,M)/E0-1),matter_density_displacement=maxabs(rho-div(u)),tt_error=maxabs(obs-ex),tt_amplitude=maxabs(obs),metric_amplitude=maxabs(h),density_mean=abs(float(rho.mean())),matter_continuity=maxabs(der[4]-div(j)),matter_momentum_identity=maxabs(der[3]-Dreal(S)))
        for key,val in vals.items():mx[key]=max(mx[key],val)
        rows.append(dict(t=round(time,12),tt=obs.tolist(),exact=ex.tolist(),scalar=vals['scalar_constraint'],momentum=vals['momentum_constraint'],energy_error=vals['matter_energy_relative'],density_displacement=vals['matter_density_displacement']))
        if i<N:z=step(z,dt,M,kappa,arm)
    return dict(L=L,d=list(d),model=model,dt=dt,arm=arm,gauge=gauge,kappa=kappa,amplitude=amp,steps=N,source_scale=sc,meta=meta,max=mx,trajectory=rows)

def offshell():
    rng=np.random.default_rng(SEED);rows=[]
    for L in [4,6,8,12,16]:
        for seed in range(4):
            h=rng.normal(size=(6,L,L,L));p=rng.normal(size=h.shape);u=rng.normal(size=(3,L,L,L));j=rng.normal(size=u.shape);rho=rng.normal(size=(L,L,L));z=(h,p,u,j,rho)
            C,V=constraints(z,KAPPA);der=rhs(z,M_EL,KAPPA);S=stress(M_EL,u)
            ed=-Dadj(j);err=abs(float(np.vdot(j,der[3])+np.vdot(ed,S)))/max(1.,float(np.linalg.norm(j)*np.linalg.norm(der[3])))
            rows.append(dict(L=L,seed=seed,scalar_identity=maxabs(curvature(der[0])-KAPPA*der[4]-div(V)),momentum_identity=maxabs(Dreal(der[1])-KAPPA*der[3]),matter_energy_identity=err,mean_density_rate=abs(float(der[4].mean())),mean_momentum_rate=maxabs(der[3].mean(axis=(1,2,3)))))
    return rows

def speed_scan():
    rng=np.random.default_rng(SEED+1);rows=[]
    for i in range(128):
        n=rng.normal(size=3);n/=np.linalg.norm(n);D,_,_,_=symbols(n)
        rows.append(dict(n=n.tolist(),elastic_speeds=np.sqrt(np.maximum(eigh(D@M_EL@D.T,eigvals_only=True),0)).tolist(),acoustic_speeds=np.sqrt(np.maximum(eigh(D@M_AC@D.T,eigvals_only=True),0)).tolist()))
    return rows

def main():
    off=offshell();speeds=speed_scan();runs=[]
    # Freeze tests/thresholds before running. Fine-grid and gauge checks use same time step.
    for dt in [.2,.1,.05]:runs.append(propagate(8,(1,1,1),'elastic',dt))
    for L in [8,12,16]:runs.append(propagate(L,(0,0,1),'elastic',.1))
    runs.append(propagate(8,(1,1,1),'acoustic',.1))
    runs.append(propagate(8,(1,1,1),'elastic',.1,gauge=True))
    for arm in ['freeze_density','omit_gravity_stress']:runs.append(propagate(8,(1,1,1),'elastic',.1,arm))
    runs.append(propagate(8,(1,1,1),'elastic',.1,kappa=0.))
    runs.append(propagate(8,(1,1,1),'elastic',.1,amp=2e-3))
    good=[r for r in runs if r['arm']=='conserved'];bad=[r for r in runs if r['arm']!='conserved'];ref=runs[1];gauge=runs[7];double=runs[11]
    arr=lambda r:np.array([x['tt'] for x in r['trajectory']]);gerr=maxabs(arr(ref)-arr(gauge));linerr=maxabs(arr(double)-2*arr(ref));blind=maxabs(arr(ref)-arr(runs[8]))
    refine=[r['max']['tt_error'] for r in runs[:3]];ratios=[a/b for a,b in zip(refine,refine[1:])]
    assert max(max(x[k] for k in ['scalar_identity','momentum_identity','matter_energy_identity','mean_density_rate','mean_momentum_rate']) for x in off)<TOL
    assert max(max(r['max'][k] for k in ['scalar_constraint','momentum_constraint','scalar_identity','momentum_identity','matter_density_displacement','matter_continuity','matter_momentum_identity']) for r in good)<TOL
    assert all(12 < a < 20 for a in ratios),ratios
    assert max(refine)<1e-6 and max(r['max']['matter_energy_relative'] for r in good)<1e-3
    assert max(r['max']['tt_amplitude'] for r in good[:6])>1e-5
    assert min(abs(x) for x in ref['meta']['TT_stress'])>1e-6,ref['meta']
    assert runs[6]['max']['tt_amplitude']<TOL and runs[10]['max']['tt_amplitude']<TOL
    assert gerr<TOL and linerr<TOL and blind<TOL
    assert bad[0]['max']['scalar_constraint']>1e-5 and bad[1]['max']['momentum_constraint']>1e-5
    assert max(max(x['elastic_speeds']) for x in speeds)<1.
    result=dict(status='PASS_BOUNDED_LINEAR_SOURCE_REFERENCE',python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,seed=SEED,tolerance=TOL,kappa=KAPPA,elastic_M=M_EL.tolist(),acoustic_M=M_AC.tolist(),offshell=off,speed_scan=speeds,runs=runs,gauge_tt_difference=gerr,source_linearity_error=linerr,freeze_density_tt_difference=blind,refinement_ratios=ratios,scope='One-way linearized classical source forcing. Mean-zero density perturbations on prescribed periodic flat background. No self-gravity of matter or quantum evolution.')
    (OUT/'results.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['offshell','speed_scan','runs']},indent=2));print('off-shell max:',{k:max(r[k] for r in off) for k in off[0] if k not in ['L','seed']});print('sampled elastic speed max:',max(max(x['elastic_speeds']) for x in speeds))
    for r in runs:print(json.dumps({k:v for k,v in r.items() if k!='trajectory'}))
    print('All assertions passed.')
if __name__=='__main__':main()
