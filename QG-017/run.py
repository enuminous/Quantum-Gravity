"""QG-017: classical Einstein-scalar Bianchi I reciprocal dynamics.
Run python run.py. Units 8*pi*G=1, fiducial comoving volume=1.
"""
import json, platform
from pathlib import Path
import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
HERE=Path(__file__).resolve().parent
RTOL=2e-12; ATOL=2e-13
# state alpha, H, beta+, beta-, s+, s-, phi, v, W
# W integrates p*d(a^3)/dt, i.e. matter pressure work.
POTS={'massless':(0.,0.),'quadratic':(.7,0.),'quartic':(.35,.4)}

def potential(phi,pot):
    m,l=POTS[pot];return .5*m*m*phi**2+.25*l*phi**4,m*m*phi+l*phi**3

def rhs(t,y,pot,arm='full',N=1.):
    a,H,b,c,s,r,phi,v,W=y; V,dV=potential(phi,pot)
    hdot=-3*(s*s+r*r)-(0 if arm=='no_matter_to_geometry' else .5*v*v)
    vdot=-(0 if arm=='no_geometry_to_matter' else 3*H*v)-dV
    return N*np.array([H,hdot,s,r,-3*H*s,-3*H*r,v,vdot,3*H*np.exp(3*a)*(.5*v*v-V)])

def diag(y,pot):
    a,H,b,c,s,r,phi,v,W=y;V,_=potential(phi,pot); vol=np.exp(3*a);rho=.5*v*v+V;pressure=.5*v*v-V
    return 3*H*H-3*(s*s+r*r)-rho,vol*rho+W,vol*np.array([s,r]),rho,pressure

def init(pot,sign,seed):
    rng=np.random.default_rng(seed);phi=rng.uniform(.2,1.1);v=rng.uniform(.2,.8)*(-1 if seed%2 else 1)
    s,r=rng.uniform(-.15,.15,2) if seed else (0.,0.)
    V,_=potential(phi,pot);H=sign*np.sqrt(s*s+r*r+(.5*v*v+V)/3)
    return np.array([0.,H,.04,-.03,s,r,phi,v,0.])

def solve(y0,pot,t,arm='full',lapse=None):
    f=lambda x,y:rhs(x,y,pot,arm,1. if lapse is None else lapse(x))
    sol=solve_ivp(f,(0,float(t[-1])),y0,t_eval=t,rtol=RTOL,atol=ATOL,method='DOP853')
    assert sol.success,(pot,arm,sol.message)
    return sol.y

def canonical(y0,pot,t):
    a,H,b,c,s,r,phi,v,_=y0;vol=np.exp(3*a)
    z0=np.array([a,b,c,phi,-6*vol*H,6*vol*s,6*vol*r,vol*v])
    def f(t,z):
        a,b,c,phi,pa,pb,pc,pp=z;vol=np.exp(3*a);V,dV=potential(phi,pot)
        T=-pa*pa/12+(pb*pb+pc*pc)/12+pp*pp/2
        return np.array([-pa/(6*vol),pb/(6*vol),pc/(6*vol),pp/vol,3*T/vol-3*vol*V,0.,0.,-vol*dV])
    sol=solve_ivp(f,(0,float(t[-1])),z0,t_eval=t,method='DOP853',rtol=RTOL,atol=ATOL);assert sol.success
    a,b,c,phi,pa,pb,pc,pp=sol.y;vol=np.exp(3*a);V,_=potential(phi,pot)
    C=(-pa*pa/12+(pb*pb+pc*pc)/12+pp*pp/2)/vol+vol*V
    return np.array([a,-pa/(6*vol),b,c,pb/(6*vol),pc/(6*vol),phi,pp/vol]),C

def exact_massless(y0,t):
    a,H,b,c,s,r,phi,v,W=y0;f=1+3*H*t;ln=np.log(f)/(3*H)
    return np.array([a+np.log(f)/3,H/f,b+s*ln,c+r*ln,s/f,r/f,phi+v*ln,v/f])

def rk4(y0,pot,tend,n):
    dt=tend/n;y=y0.copy();out=[y.copy()]
    for i in range(n):
        k1=rhs(i*dt,y,pot);k2=rhs((i+.5)*dt,y+dt*k1/2,pot);k3=rhs((i+.5)*dt,y+dt*k2/2,pot);k4=rhs((i+1)*dt,y+dt*k3,pot)
        y+=dt*(k1+2*k2+2*k3+k4)/6;out.append(y.copy())
    return np.array(out).T

def maxabs(x):return float(np.max(np.abs(x)))
def relerr(a,b):return maxabs((a-b)/(1+np.abs(b)))

def main():
    records=[]; trajectories=[]; exacts=[]; canon=[]; lapses=[]; off=[]; convergence=[]; stress=[]; raw={}
    for pot in POTS:
      for sign in [1,-1]:
       for seed in range(4):
        y0=init(pot,sign,seed);T=(.6 if sign==1 else .17)/abs(y0[1]);t=np.linspace(0,T,101)
        base=solve(y0,pot,t);C0,energy0,shear0,*_=diag(y0,pot)
        ref,ham=canonical(y0,pot,t);canon.append({'pot':pot,'sign':sign,'seed':seed,'state_scaled_error':relerr(ref,base[:8]),'hamiltonian_scaled_residual':maxabs(ham/(1+np.exp(3*ref[0])*diag(base,pot)[3]))})
        if pot=='massless':exacts.append({'sign':sign,'seed':seed,'error':relerr(base[:8],exact_massless(y0,t))})
        # compare at the same proper times with variable lapse
        lapse=lambda x:1+.3*np.sin(x)
        times=np.array([brentq(lambda x:x+.3*(1-np.cos(x))-tau,0,max(tau/.7+1e-10,1e-10),xtol=1e-13) if tau else 0. for tau in t])
        varied=solve(y0,pot,times,lapse=lapse);lapses.append({'pot':pot,'sign':sign,'seed':seed,'scaled_error':relerr(varied,base)})
        for arm in ['full','no_matter_to_geometry','no_geometry_to_matter']:
          y=base if arm=='full' else solve(y0,pot,t,arm)
          C,ew,sh,rho,pressure=diag(y,pot);a,H,b,c,s,r,phi,v,W=y
          scale=3*H*H+3*(s*s+r*r)+rho+1e-15
          e_scale=1+abs(energy0);out={'pot':pot,'sign':sign,'seed':seed,'arm':arm,'duration':float(T),'constraint_abs':maxabs(C),'constraint_rel':maxabs(C/scale),'pressure_work_residual':maxabs(ew-energy0)/e_scale,'shear_momentum_residual':maxabs(sh-shear0[:,None]),'H_delta':maxabs(H-base[1]),'phi_delta':maxabs(phi-base[6]),'volume_final':float(np.exp(3*a[-1])),'rho_initial':float(rho[0]),'rho_final':float(rho[-1]),'work_final':float(W[-1]),'matter_energy_initial':float(energy0),'matter_energy_final':float(np.exp(3*a[-1])*rho[-1])}
          records.append(out)
          raw[f'{pot}_{sign}_{seed}_{arm}']=np.vstack([t,y])
          if seed==1:trajectories.append({'pot':pot,'sign':sign,'arm':arm,'t':t.tolist(),'H':H.tolist(),'phi':phi.tolist(),'rho':rho.tolist(),'volume':np.exp(3*a).tolist(),'constraint':C.tolist(),'work_balance':(ew-energy0).tolist()})
    # Long-window stress test: expanding starts; stop at large |H| or small volume.
    # The stopping surface is a numerical domain limit, not a singularity diagnosis.
    for pot in POTS:
      for seed in range(4):
        y0=init(pot,1,seed);T=2.5/abs(y0[1]);H0=abs(y0[1])
        def stop(t,y):return min(20-abs(y[1])/H0, y[0]-np.log(1e-3)/3)
        stop.terminal=True;stop.direction=-1
        for arm in ['full','no_matter_to_geometry','no_geometry_to_matter']:
          sol=solve_ivp(lambda t,y:rhs(t,y,pot,arm),(0,T),y0,method='DOP853',rtol=RTOL,atol=ATOL,events=stop)
          C,*_=diag(sol.y,pot)
          stress.append({'pot':pot,'seed':seed,'arm':arm,'requested_time':float(T),'final_time':float(sol.t[-1]),'status':int(sol.status),'success':bool(sol.success),'message':sol.message,'final_H':float(sol.y[1,-1]),'final_volume':float(np.exp(3*sol.y[0,-1])),'max_abs_constraint':maxabs(C)})
    # Exact off-shell derivative identity and control signatures. No constraint projection.
    rng=np.random.default_rng(172026)
    for i in range(128):
      y=rng.uniform(-.8,.8,9);pot=list(POTS)[i%3];a,H,b,c,s,r,phi,v,W=y;_,dV=potential(phi,pot)
      grad=np.array([0,6*H,0,0,-6*s,-6*r,-dV,-v,0])
      for arm,expected in [('full',0.),('no_matter_to_geometry',3*H*v*v),('no_geometry_to_matter',-3*H*v*v)]:
        observed=float(grad@rhs(0,y,pot,arm));off.append({'case':i,'pot':pot,'arm':arm,'observed':observed,'expected':float(expected),'error':abs(observed-expected)})
    for sign in [1,-1]:
      y0=init('quartic',sign,1);T=(.6 if sign==1 else .17)/abs(y0[1]);ref=solve(y0,'quartic',np.linspace(0,T,3))[:,-1]
      for n in [64,128,256,512]:
        y=rk4(y0,'quartic',T,n);C,*_=diag(y,'quartic');convergence.append({'sign':sign,'steps':n,'dt':T/n,'final_scaled_error':relerr(y[:,-1],ref),'constraint_abs':maxabs(C)})
    full=[r for r in records if r['arm']=='full'];ng=[r for r in records if r['arm']=='no_geometry_to_matter'];nm=[r for r in records if r['arm']=='no_matter_to_geometry']
    summary={'stress_runs':len(stress),'stress_stops_full':sum(r['status']==1 for r in stress if r['arm']=='full'),'stress_stops_ablation':sum(r['status']==1 for r in stress if r['arm']!='full'),'stress_solver_failures':sum(not r['success'] for r in stress),'full_runs':len(full),'control_runs':len(ng)+len(nm),'canonical_runs':len(canon),'lapse_runs':len(lapses),'rk4_runs':len(convergence),'sampled_primary_states':len(records)*101,'off_shell_checks':len(off),'full_max_constraint_rel':max(r['constraint_rel'] for r in full),'full_max_constraint_abs':max(r['constraint_abs'] for r in full),'full_max_work_residual':max(r['pressure_work_residual'] for r in full),'full_max_shear_residual':max(r['shear_momentum_residual'] for r in full),'canonical_max_scaled_error':max(r['state_scaled_error'] for r in canon),'canonical_max_scaled_hamiltonian':max(r['hamiltonian_scaled_residual'] for r in canon),'lapse_max_scaled_error':max(r['scaled_error'] for r in lapses),'massless_exact_max_scaled_error':max(r['error'] for r in exacts),'off_shell_max_identity_error':max(r['error'] for r in off),'no_matter_to_geometry_min_max_constraint_rel':min(r['constraint_rel'] for r in nm),'no_matter_to_geometry_max_constraint_rel':max(r['constraint_rel'] for r in nm),'no_geometry_to_matter_min_max_constraint_rel':min(r['constraint_rel'] for r in ng),'no_geometry_to_matter_max_constraint_rel':max(r['constraint_rel'] for r in ng),'max_H_delta_when_matter_removed':max(r['H_delta'] for r in nm),'max_phi_delta_when_geometry_removed':max(r['phi_delta'] for r in ng)}
    checks={'valid_constraints':summary['full_max_constraint_rel']<1e-8,'pressure_work':summary['full_max_work_residual']<1e-8,'shear_conservation':summary['full_max_shear_residual']<1e-8,'canonical_agreement':summary['canonical_max_scaled_error']<1e-8,'canonical_constraint':summary['canonical_max_scaled_hamiltonian']<1e-8,'lapse_agreement':summary['lapse_max_scaled_error']<1e-8,'analytic_massless':summary['massless_exact_max_scaled_error']<1e-8,'off_shell_identity':summary['off_shell_max_identity_error']<1e-12,'detect_each_ablation':min(summary['no_matter_to_geometry_min_max_constraint_rel'],summary['no_geometry_to_matter_min_max_constraint_rel'])>1e-3,'reciprocal_response':min(summary['max_H_delta_when_matter_removed'],summary['max_phi_delta_when_geometry_removed'])>1e-3,'rk4_refinement':all(group[-1]['final_scaled_error']<group[0]['final_scaled_error']/100 for group in ([r for r in convergence if r['sign']==s] for s in [1,-1]))}
    checks={k:bool(v) for k,v in checks.items()};result={'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},'config':{'seeds':list(range(4)),'potentials':POTS,'rtol':RTOL,'atol':ATOL,'samples_per_primary_run':101,'thresholds':'exploratory, specified in checks'},'summary':summary,'checks':checks,'runs':records,'canonical':canon,'lapse':lapses,'massless_exact':exacts,'off_shell':off,'convergence':convergence,'selected_trajectories':trajectories,'long_window_stress':stress}
    np.savez_compressed(HERE/'primary_trajectories.npz',**raw)
    (HERE/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'summary':summary,'checks':checks,'convergence':convergence},indent=2));assert all(checks.values()),checks
if __name__=='__main__':main()
