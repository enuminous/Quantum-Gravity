"""QG-019 exploratory oscillator mediator and static Poisson constraint. Units hbar=omega=1."""
import json, sys, platform
from pathlib import Path
import numpy as np
import scipy
from scipy.linalg import eigh

ROOT=Path(__file__).resolve().parent
Z=np.array([1.,-1.]); signs=np.array([(a,b) for a in Z for b in Z]); r0=np.ones((4,4))/4
T=2*np.pi
pairs=[(0.,0.),(.25,0.),(.1,.1),(.25,.25),(.35,.35),(.2,.3)]
nbars=[0.,.5,2.]

def neg(r):
    pt=r.reshape(2,2,2,2).transpose(0,3,2,1).reshape(4,4)
    return float(np.maximum(-np.linalg.eigvalsh(pt),0).sum())
def purity(r): return float(np.trace(r@r).real)
def analytic(f,t,nbar):
    d=f[:,None]-f[None,:]; phase=(f[:,None]**2-f[None,:]**2)*(t-np.sin(t))
    decay=np.exp(-(2*nbar+1)*d*d*(1-np.cos(t)))
    return r0*decay*np.exp(1j*phase),r0*decay

def thermal(N,nbar):
    ratio=nbar/(1+nbar); w=ratio**np.arange(N); return w/w.sum()

def diagonalize(N,f):
    a=np.diag(np.sqrt(np.arange(1,N)),1); X=a+a.T; H0=np.diag(np.arange(N))
    return a,X,H0,[eigh(H0+v*X) for v in f]

def numeric(N,f,nbar,ts):
    a,X,H0,blocks=diagonalize(N,f); w=thermal(N,nbar); sqrtw=np.sqrt(w)
    energy0=float(w@np.arange(N)); out=[]
    for t in ts:
        # U_i sqrt(rho_osc), exact finite-matrix propagation for each spin branch.
        V=np.array([(U*np.exp(-1j*t*e))@U.T*sqrtw for e,U in blocks]); flat=V.reshape(4,-1)
        r=(flat@flat.conj().T)/4
        branch_x=np.einsum('bik,ij,bjk->b',V.conj(),X,V).real
        energy=np.mean([np.vdot(v,(H0+ff*X)@v).real for v,ff in zip(V,f)])
        rq,rc=analytic(f,t,nbar)
        # oscillator/probe pure-state entanglement only for initial vacuum
        out.append(dict(t=float(t),analytic_error=float(np.max(np.abs(r-rq))),norm_error=float(abs(np.trace(r)-1)),energy_error=float(abs(energy-energy0)),min_eigenvalue=float(np.linalg.eigvalsh(r)[0]),branch_x_error=float(np.max(np.abs(branch_x+2*f*(1-np.cos(t))))),negativity=neg(r),purity=purity(r)))
    return out

def poisson(N):
    x=np.arange(N)/N; d=lambda c: (x-c+.5)%1-.5
    densities=np.array([np.exp(-d(c)**2/(2*.045**2)) for c in (.25,.75)])
    densities/=densities.mean(axis=1)[:,None]; densities-=1
    k=2*np.pi*np.fft.fftfreq(N,d=1/N); eig=-4*N*N*np.sin(k/(2*N))**2; mask=k!=0
    def solve(r):
        v=np.fft.fft(r); v[...,mask]/=eig[mask]; v[...,~mask]=0; return np.fft.ifft(v).real
    def lap(v):return (np.roll(v,1,axis=-1)+np.roll(v,-1,axis=-1)-2*v)*N*N
    branch=solve(densities); cases=[]
    for p in (0.,.1,.25,.5,.75,.9,1.):
        w=np.array([p,1-p]); src=w@densities; mean=solve(src); meanr=lap(mean)-src
        residual=lap(mean)[None,:]-densities
        rms=np.sqrt(np.mean(w@(residual**2)))
        # variance of source after expectation-value subtraction
        predicted=np.sqrt(p*(1-p)*np.mean((densities[0]-densities[1])**2))
        cases.append(dict(p=p,mean_residual=float(np.max(np.abs(meanr))),operator_rms=float(rms),predicted=float(predicted),branch_residual=float(np.max(np.abs(lap(branch)-densities))),potential_mean_error=float(np.max(np.abs(mean-w@branch))),potential_rms_fluctuation=float(np.sqrt(np.mean(w@((branch-mean)**2))))))
    return dict(N=N,cases=cases,profile=dict(x=x[::max(1,N//64)].tolist(),rho_a=densities[0,::max(1,N//64)].tolist(),rho_b=densities[1,::max(1,N//64)].tolist(),phi_a=branch[0,::max(1,N//64)].tolist(),phi_b=branch[1,::max(1,N//64)].tolist()))

def main():
    rows=[]; numeric_rows=[]
    ts=np.linspace(0,T,129); nts=np.linspace(0,T,33)
    for ga,gb in pairs:
        f=signs@np.array([ga,gb]);
        for n in nbars:
            series=[]
            for t in ts:
                rq,rc=analytic(f,t,n)
                series.append([float(t),neg(rq),neg(rc),purity(rq),float(abs(rq[0,3])),float(abs(rc[0,3]))])
            end=series[-1]; theta=4*np.pi*ga*gb
            rows.append(dict(ga=ga,gb=gb,nbar=n,columns=['t','quantum_neg','classical_noise_neg','quantum_probe_purity','quantum_coherence_abs_03','classical_coherence_abs_03'],series=series,revival_neg=end[1],predicted_rev_neg=float(abs(np.sin(2*theta))/2),midpoint_neg=series[64][1],midpoint_purity=series[64][3],max_neg=max(r[1] for r in series)))
            for N in (24,48,80):
                data=numeric(N,f,n,nts)
                numeric_rows.append(dict(ga=ga,gb=gb,nbar=n,N=N,samples=data))
    ps=[poisson(N) for N in (64,128,256)]
    fine=[d for r in numeric_rows if r['N']==80 for d in r['samples']]; allsamples=[d for r in numeric_rows for d in r['samples']]
    summary=dict(analytic_cases=len(rows),numeric_cases=len(numeric_rows),numeric_samples=len(allsamples),analytic_samples=len(rows)*len(ts),fine_analytic_error=max(d['analytic_error'] for d in fine),fine_energy_error=max(d['energy_error'] for d in fine),fine_norm_error=max(d['norm_error'] for d in fine),fine_branch_error=max(d['branch_x_error'] for d in fine),fine_min_eigenvalue=min(d['min_eigenvalue'] for d in fine),max_classical_neg=max(r[2] for c in rows for r in c['series']),revival_error=max(abs(c['revival_neg']-c['predicted_rev_neg']) for c in rows),poisson_mean_error=max(c['mean_residual'] for p in ps for c in p['cases']),poisson_branch_error=max(c['branch_residual'] for p in ps for c in p['cases']),poisson_rms_error=max(abs(c['operator_rms']-c['predicted']) for p in ps for c in p['cases']))
    summary['cutoff_errors']={str(N):max(d['analytic_error'] for r in numeric_rows if r['N']==N for d in r['samples']) for N in (24,48,80)}
    target=[r for r in rows if r['ga']==r['gb']==.25]; controls=[r for r in rows if r['gb']==0]
    checks=dict(fine_analytic=summary['fine_analytic_error']<1e-8,norm=summary['fine_norm_error']<1e-10,energy=summary['fine_energy_error']<1e-9,positivity=summary['fine_min_eigenvalue']>-1e-10,branch=summary['fine_branch_error']<1e-7,classical_separability=summary['max_classical_neg']<1e-12,revival_formula=summary['revival_error']<1e-12,target_entanglement=all(r['revival_neg']>.499999 for r in target),one_probe_control=all(r['max_neg']<1e-12 for r in controls),poisson_mean=summary['poisson_mean_error']<1e-8,poisson_branch=summary['poisson_branch_error']<1e-8,poisson_variance=summary['poisson_rms_error']<1e-9)
    out=dict(environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__),summary=summary,checks=checks,analytic=rows,numeric=numeric_rows,poisson=ps)
    (ROOT/'results.json').write_text(json.dumps(out,separators=(',',':'))+'\n')
    print(json.dumps(dict(summary=summary,checks=checks,target=[{k:v for k,v in r.items() if k not in ('series','columns')} for r in target],poisson_mid=[p['cases'][3] for p in ps]),indent=2)); assert all(checks.values()),checks
if __name__=='__main__':main()
