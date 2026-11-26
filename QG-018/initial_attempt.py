"""QG-018: constrained spherical Einstein--massless scalar evolution."""
import numpy as np, scipy, json, sys, platform
from pathlib import Path
from scipy.integrate import cumulative_trapezoid
from time import perf_counter
ROOT=Path(__file__).resolve().parent

def integ(y,dr): return cumulative_trapezoid(y,dx=dr,initial=0)
def deriv(y,dr): return np.gradient(y,dr,edge_order=2)
def geom(r,y,dr):
    phi,pi=y; S=phi*phi+pi*pi
    I=integ(r*S,dr); m=np.exp(-I)*integ(.5*r*r*S*np.exp(I),dr)
    compact=np.divide(2*m,r,out=np.zeros_like(m),where=r>0)
    if not np.all(np.isfinite(compact)) or compact.max()>=.85: raise RuntimeError('polar-areal domain limit')
    a=1/np.sqrt(1-compact); alpha=np.exp(I-I[-1])/a
    return m,a,alpha

def initial(r,amp,direction):
    z=(r-6)/2; b=np.zeros_like(r); Phi=b.copy(); mask=np.abs(z)<1
    b[mask]=amp*np.exp(1-1/(1-z[mask]**2))
    Phi[mask]=-b[mask]*z[mask]/(1-z[mask]**2)**2
    Pi=direction*(Phi+np.divide(b,r,out=np.zeros_like(b),where=r>0))
    return np.array([Phi,Pi])

def run(n,amp,direction,arm,T=4,record=40):
    r=np.linspace(0,20,n+1); dr=r[1]; steps=int(np.ceil(T/(.2*dr)/record))*record; dt=T/steps
    y=initial(r,amp,direction); g0=geom(r,y,dr)
    def metric(y):
        if arm=='full': return geom(r,y,dr)
        if arm=='frozen': return g0
        if arm=='flat': return np.zeros_like(r),np.ones_like(r),np.ones_like(r)
        raise ValueError(arm)
    def rhs(y):
        _,a,alpha=metric(y); v=alpha/a; F=v*y[0]; J=v*y[1]
        dPhi=deriv(J,dr); dPi=deriv(F,dr); dPi[1:]+=2*F[1:]/r[1:]; dPi[0]=3*F[1]/dr; dPhi[0]=0
        dPhi[-1]=dPi[-1]=0
        return np.array([dPhi,dPi])
    def diagnostic(t,y):
        m,a,alpha=metric(y); phi,pi=y; S=phi**2+pi**2
        expected=.5*r*r*S/a**2
        masserr=deriv(m,dr)-expected; lapseerr=deriv(np.log(alpha*a),dr)-r*S
        # Centered directional derivative of radial solve; flux equation not enforced.
        dydt=rhs(y); eps=1e-5
        mdot=(metric(y+eps*dydt)[0]-metric(y-eps*dydt)[0])/(2*eps)
        flux=r*r*alpha/a**3*phi*pi
        scale=max(np.max(abs(expected)),1e-14); fluxscale=max(np.max(abs(flux)),1e-14)
        return dict(t=t,mass=float(m[-1]),compactness=float(np.max(np.divide(2*m,r,out=np.zeros_like(m),where=r>0))),central_lapse=float(alpha[0]),ham_abs=float(np.max(abs(masserr[1:-1]))),ham_rel=float(np.max(abs(masserr[1:-1]))/scale),lapse_abs=float(np.max(abs(lapseerr[1:-1]))),flux_abs=float(np.max(abs(mdot[1:-1]-flux[1:-1]))),flux_rel=float(np.max(abs(mdot[1:-1]-flux[1:-1]))/fluxscale),outer_matter=float(np.max(abs(y[:,-10:]))))
    rows=[diagnostic(0,y)]; snapshots=[np.stack([*y,*metric(y)])]
    for j in range(steps):
        k1=rhs(y); k2=rhs(y+dt*k1/2); k3=rhs(y+dt*k2/2); k4=rhs(y+dt*k3)
        y+=dt*(k1+2*k2+2*k3+k4)/6
        if (j+1)%(steps//record)==0:
            rows.append(diagnostic((j+1)*dt,y)); snapshots.append(np.stack([*y,*metric(y)]))
    summary={k:max(row[k] for row in rows) for k in ['ham_rel','ham_abs','lapse_abs','flux_rel','flux_abs','compactness','outer_matter']}
    M=rows[0]['mass'];summary['mass_drift']=max(abs(row['mass']-M) for row in rows)/max(M,1e-14)
    summary['lapse_min']=min(row['central_lapse'] for row in rows)
    return dict(n=n,amp=amp,direction=direction,arm=arm,steps=steps,dt=dt,summary=summary,rows=rows),np.array(snapshots),r

def main():
    tic=perf_counter(); results=[]; data={}; grids={}
    for a in [.02,.05,.08]:
      for d in [1,-1]:
        for n,arm in [(200,'full'),(400,'full'),(800,'full'),(800,'frozen'),(800,'flat')]:
          out,s,r=run(n,a,d,arm); key=f'a{a}_d{d}_n{n}_{arm}';data[key]=s;grids[str(n)]=r;results.append(out)
          print(key, {k:round(v,9) for k,v in out['summary'].items()},flush=True)
    refinement=[]; reciprocal=[]
    for a in [.02,.05,.08]:
      for d in [1,-1]:
        k=lambda n,arm:f'a{a}_d{d}_n{n}_{arm}'
        coarse=data[k(200,'full')][-1,:2];mid=data[k(400,'full')][-1,:2];fine=data[k(800,'full')][-1,:2]
        e1=float(np.linalg.norm(coarse-mid[:,::2])/np.sqrt(coarse.size)); e2=float(np.linalg.norm(mid-fine[:,::2])/np.sqrt(mid.size)); refinement.append(dict(amp=a,direction=d,e200_400=e1,e400_800=e2,ratio=e1/e2))
        F=data[k(800,'full')]; B=data[k(800,'frozen')]; L=data[k(800,'flat')]; reciprocal.append(dict(amp=a,direction=d,full_frozen_final_matter_rel=float(np.linalg.norm(F[-1,:2]-B[-1,:2])/np.linalg.norm(F[-1,:2])),full_flat_final_matter_rel=float(np.linalg.norm(F[-1,:2]-L[-1,:2])/np.linalg.norm(F[-1,:2])),full_geometry_change=float(np.max(abs(F[:,3:]-F[0,3:])))))
    vac,s,r=run(200,0,1,'full'); assert np.max(abs(s[:,:3]))==0 and np.max(abs(s[:,3:]-1))==0
    finest=[x for x in results if x['n']==800 and x['arm']=='full'];frozen=[x for x in results if x['arm']=='frozen'];flat=[x for x in results if x['arm']=='flat']
    checks=dict(vacuum=True,refinement=all(x['ratio']>2.5 for x in refinement),positive_metric=all(np.min(x[:,3:])>0 for x in data.values()),full_flux_residual=all(x['summary']['flux_rel']<.03 for x in finest),frozen_flux_detected=all(x['summary']['flux_rel']>.99 for x in frozen),flat_constraint_detected=all(x['summary']['ham_rel']>.99 for x in flat),full_mass_drift=all(x['summary']['mass_drift']<.005 for x in finest),reciprocity=all(x['full_frozen_final_matter_rel']>1e-5 and x['full_geometry_change']>1e-5 for x in reciprocal),boundary_clear=all(x['summary']['outer_matter']<1e-9 for x in results))
    payload=dict(spec=dict(domain=[0,20],T=4,grid_intervals=[200,400,800],amplitudes=[.02,.05,.08],directions={'1':'initial inward flat-space profile','-1':'initial outward flat-space profile'},arms=['full','frozen','flat'],samples_per_run=41,thresholds='see checks in run.py; exploratory'),environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform()),runs=results,refinement=refinement,reciprocal=reciprocal,vacuum=vac,checks=checks)
    (ROOT/'results.json').write_text(json.dumps(payload,indent=2)+'\n');np.savez_compressed(ROOT/'snapshots.npz',**data,**{'r'+k:v for k,v in grids.items()}); print('REFINEMENT',refinement);print('RECIPROCAL',reciprocal);print('CHECKS',checks);print('SECONDS',perf_counter()-tic);assert all(checks.values())
if __name__=='__main__':main()
