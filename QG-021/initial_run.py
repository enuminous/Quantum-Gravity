"""QG-021: reduced Newtonian two-body quantum model, not full quantum GR."""
import json, platform
from pathlib import Path
import numpy as np
import scipy
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import expm_multiply
from scipy.integrate import solve_ivp

OUT=Path(__file__).resolve().parent
M=5.; WIDTH=2.; TIMES=np.linspace(0,4,81)

def model(n,d,g,profile='double',kind='full'):
    dx=WIDTH/(n+1); x=-WIDTH/2+dx*np.arange(1,n+1)
    T=diags([-np.ones(n-1),2*np.ones(n),-np.ones(n-1)],[-1,0,1],format='csr')/(2*M*dx**2)
    r=d+x[None,:]-x[:,None]; V=-g/r
    if kind=='additive': V=-g/d+g*(x[None,:]-x[:,None])/d**2+np.zeros_like(r)
    if profile=='double': a=np.exp(-((x-.42)/.18)**2/2)+np.exp(-((x+.42)/.18)**2/2)
    else: a=np.exp(-(x/.22)**2/2)
    a=a/np.linalg.norm(a); b=a.copy(); psi=np.outer(a,b)
    H=(kron(T,eye(n))+kron(eye(n),T)+diags(V.ravel())).tocsr()
    return x,T,r,V,a,b,psi,H

def run(n,d,g,profile='double',kind='full',mean=True):
    x,T,r,V,a,b,psi,H=model(n,d,g,profile,kind)
    states=expm_multiply(-1j*H,psi.ravel().astype(complex),start=0,stop=4,num=len(TIMES),traceA=-1j*H.diagonal().sum()).reshape(-1,n,n)
    def obs(z):
        p=abs(z)**2; pa=p.sum(1); pb=p.sum(0); s=np.linalg.svd(z,compute_uv=False); purity=np.sum(s**4)
        Fa=np.sum(p*g/r**2) if kind=='full' else g/d**2
        return [float(x@pa),float(x@pb),float(np.sum(p*r)),float(1-purity),float(((s.sum())**2-1)/2),float(Fa),float(p.sum()),float(np.vdot(z.ravel(),H@z.ravel()).real)]
    q=np.array([obs(z) for z in states]); m=None; mf_norm=None; mf_energy=None
    if mean:
        def rhs(t,z):
            A,B=z[:n],z[n:]; va=V@abs(B)**2; vb=V.T@abs(A)**2
            return -1j*np.r_[T@A+va*A,T@B+vb*B]
        sol=solve_ivp(rhs,(0,4),np.r_[a,b].astype(complex),t_eval=TIMES,method='DOP853',rtol=2e-10,atol=2e-12)
        if not sol.success: raise RuntimeError(sol.message)
        m=np.array([obs(np.outer(z[:n],z[n:])) for z in sol.y.T]); mf_norm=float(np.max(abs(m[:,6]-1))); mf_energy=float(np.max(abs(m[:,7]-m[0,7])))
    p0=abs(psi)**2; pa=p0.sum(1); pb=p0.sum(0)
    vc=V-(V@pb)[:,None]-(V.T@pa)[None,:]+pa@V@pb
    predicted=2*np.sum(p0*vc**2)
    dt=.002; short=expm_multiply(-1j*dt*H,psi.ravel().astype(complex)).reshape(n,n)
    observed=obs(short)[3]/dt**2
    def commut(A,B): return A@B-B@A
    X=diags(x); acc=-commut(T,commut(T,X))
    Za=-commut(H,commut(H,kron(X,eye(n))))
    Zb=-commut(H,commut(H,kron(eye(n),X)))
    z=psi.ravel(); aa=float(np.vdot(z,Za@z).real); ab=float(np.vdot(z,Zb@z).real)
    free=float(np.vdot(a,acc@a).real)
    summary={'n':n,'D':d,'coupling':g,'profile':profile,'kind':kind,'support_gap':d-WIDTH,'peak_entropy':float(max(q[:,3])),'final_entropy':float(q[-1,3]),'peak_negativity':float(max(q[:,4])),'final_separation':float(q[-1,2]),'early_separation':float(q[2,2]),'q_norm_error':float(max(abs(q[:,6]-1))),'q_energy_error':float(max(abs(q[:,7]-q[0,7]))),'mean_norm_error':mf_norm,'mean_energy_error':mf_energy,'mean_peak_entropy':None if m is None else float(max(abs(m[:,3]))),'mean_final_separation':None if m is None else float(m[-1,2]),'max_mean_separation_difference':None if m is None else float(max(abs(q[:,2]-m[:,2]))),'short_entropy_coefficient_predicted':float(predicted),'short_entropy_coefficient_measured':float(observed),'initial_accel_A':aa,'initial_accel_B':ab,'free_initial_accel':free,'initial_force_A':float(q[0,5]),'initial_force_B':float(-q[0,5]),'initial_accel_force_relative_error':float(abs(aa-free-q[0,5]/M)/max(abs(q[0,5]/M),1e-10)) if g else 0.}
    return summary,q,m

def main():
    configs=[(47,d,g,'double','full',True) for d in (3.,4.,6.) for g in (0.,2.,8.)]
    configs += [(n,3.,8.,'double','full',True) for n in (23,95)]
    configs += [(47,3.,8.,'single','full',True),(47,3.,8.,'double','additive',True)]
    rows=[]; trajectories=[]
    for c in configs:
        s,q,m=run(*c); rows.append(s); trajectories.append({'config':list(c),'quantum':q.tolist(),'mean_field':None if m is None else m.tolist()})
        print('case',c,'entropy',s['peak_entropy'],'neg',s['peak_negativity'],'early sep',s['early_separation'],'accel err',s['initial_accel_force_relative_error'],flush=True)
    valid=[r for r in rows if r['coupling']>0 and r['kind']=='full']; null=[r for r in rows if r['coupling']==0 or r['kind']=='additive']
    coarse,mid,fine=rows[9],rows[2],rows[10]; e1=abs(coarse['peak_entropy']-mid['peak_entropy']); e2=abs(mid['peak_entropy']-fine['peak_entropy'])
    checks={'unitary_norm':max(r['q_norm_error'] for r in rows)<1e-9,'unitary_energy':max(r['q_energy_error'] for r in rows)<1e-8,'mean_norm':max(r['mean_norm_error'] for r in rows)<1e-7,'mean_energy':max(r['mean_energy_error'] for r in rows)<1e-7,'mean_product':max(r['mean_peak_entropy'] for r in rows)<1e-7,'null_entanglement':max(abs(r['peak_entropy']) for r in null)<1e-9,'disjoint_entanglement':min(r['peak_entropy'] for r in valid)>1e-5,'attraction':all(r['early_separation']<r['D'] for r in valid),'opposite_acceleration':all(r['initial_accel_A']>0 and r['initial_accel_B']<0 for r in valid),'short_time_coefficient':all(abs(r['short_entropy_coefficient_measured']/r['short_entropy_coefficient_predicted']-1)<.01 for r in valid),'spatial_refinement':e2<e1,'force_correspondence_fine':fine['initial_accel_force_relative_error']<.02}
    result={'runtime':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},'columns':['mean_x_A_relative','mean_x_B_relative','separation','linear_entropy','negativity','attractive_force_A','norm','energy'],'times':TIMES.tolist(),'summary':rows,'trajectories':trajectories,'refinement':{'coarse_to_medium':e1,'medium_to_fine':e2,'ratio':e1/e2},'checks':checks}
    (OUT/'results.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
    (OUT/'summary.json').write_text(json.dumps({k:v for k,v in result.items() if k!='trajectories' and k!='times'},indent=2)+'\n')
    print('checks',checks,'refinement',result['refinement'],flush=True)
    if not all(checks.values()): raise SystemExit('One or more checks failed; retain result.')
if __name__=='__main__': main()
