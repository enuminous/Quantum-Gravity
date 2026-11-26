"""QG-015: local staggered linearized ADM diagnostic, not emergent GR."""
import json, platform
from pathlib import Path
import numpy as np
from scipy.linalg import expm, null_space
import scipy
OUT=Path(__file__).resolve().parent
rng=np.random.default_rng(15026)
TOL=2e-10
pairs=[(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]
B=np.zeros((6,3,3))
for a,(i,j) in enumerate(pairs):
    B[a,i,j]=B[a,j,i]=1 if i==j else 1/np.sqrt(2)
t=np.array([1.,1.,1.,0.,0.,0.]); I=np.eye(6)
J=np.block([[np.zeros((6,6)),I],[-I,np.zeros((6,6))]])
Pkin=I-np.outer(t,t)/2

def symbols(q):
    s=q@q; D=np.einsum('aij,j->ia',B,q); c=q@D-s*t
    K=s*I-2*D.T@D+np.outer(D.T@q,t)+np.outer(t,D.T@q)-s*np.outer(t,t)
    C=np.block([[c[None,:],np.zeros((1,6))],[np.zeros((3,6)),D]])
    return D,c,K,C

def maxabs(a):return float(np.max(np.abs(a)))
def block(K,P=Pkin):return np.block([[K,np.zeros((6,6))],[np.zeros((6,6)),P]])
def mode(q,seed):
    D,c,K,C=symbols(q); s=q@q
    W=null_space(np.vstack((D,t))); Z=scipy.linalg.block_diag(W,W)
    H=block(K); F=J@H; R=np.zeros((4,4));R[0,1:]=q
    # Gauge shifts span J C.T; no projection inserted into H.
    gauge=J@C.T; z=Z@rng.normal(size=4); z2=z+gauge@rng.normal(size=4)
    errors=[]; constraint=[]; energy=[]; bareleak=[]
    for time in np.linspace(0,8/np.sqrt(s),17):
        E=expm(time*F); a=E@z; b=E@z2
        errors.append(maxabs(Z.T@(a-b)))
        constraint.append(maxabs(C@b));energy.append(abs(b@H@b-z2@H@z2)/2)
    N=null_space(C)
    for name,Hbad in [('identity_kinetic',block(K,I)),('scalar_stiffness',block(s*I)),('remove_cross',block(K-np.outer(D.T@q,t)-np.outer(t,D.T@q)))]:
        bareleak.append([name,float(np.linalg.norm(C@J@Hbad@N,2))])
    return dict(seed=seed,q=q.tolist(),rank=int(np.linalg.matrix_rank(C)),physical_pairs=W.shape[1],
        bracket=maxabs(C@J@C.T),closure=maxabs(C@F-R@C),reduced=maxabs(Z.T@H@Z-np.diag([s,s,1,1])),
        gauge=max(errors),trajectory_constraints=max(constraint),energy=float(max(energy)),negative_controls=dict(bareleak))
# Tensor h_ii at vertices, h_ij at ij face centers, vector v_i at i-edge centers.
# forward: vertex -> edge or edge -> face, backward: reverse; grid spacing = 1.
def f(a,j):return np.roll(a,-1,axis=j)-a
def b(a,j):return a-np.roll(a,1,axis=j)
def lap(a):return sum(b(f(a,j),j) for j in range(3))
def grad(a):return np.stack([f(a,j) for j in range(3)])
def div(v):return sum(b(v[j],j) for j in range(3))
def tr(h):return h[0]+h[1]+h[2]
def lift(a):return np.stack([a,a,a,np.zeros_like(a),np.zeros_like(a),np.zeros_like(a)])
def Dreal(h):
    v=np.stack([f(h[i],i) for i in range(3)])
    for a,(i,j) in enumerate(pairs[3:],3):
        v[i]+=b(h[a],j)/np.sqrt(2);v[j]+=b(h[a],i)/np.sqrt(2)
    return v
def Dadj(v):
    return np.stack([-b(v[i],i) if i==j else -(f(v[i],j)+f(v[j],i))/np.sqrt(2) for i,j in pairs])
def curvature(h):return div(Dreal(h))-lap(tr(h))
def Kh(h):
    return -np.stack([lap(x) for x in h])-2*Dadj(Dreal(h))+Dadj(grad(tr(h)))+lift(-div(Dreal(h))+lap(tr(h)))
def Kp(p):return p-lift(tr(p))/2

def realtest(L,seed):
    h=rng.normal(size=(6,L,L,L));p=rng.normal(size=h.shape);v=rng.normal(size=(3,L,L,L));a=rng.normal(size=(L,L,L))
    u=rng.normal(size=h.shape)
    # Adjoint and Hessian checks use arbitrary off-shell fields.
    norm=lambda x:max(1.,float(np.linalg.norm(x.ravel())))
    inner=lambda x,y:float(np.vdot(x,y).real)
    # Check against independent complex Fourier symbols, including staggered phases.
    k=2*np.pi*np.fft.fftfreq(L); ix=(1,2,3%L); kk=np.array([k[i] for i in ix]);q=2*np.sin(kk/2)
    offsets=np.array([np.zeros(3) if i==j else (np.eye(3)[i]+np.eye(3)[j])/2 for i,j in pairs])
    phase=np.exp(1j*(offsets@kk)); x=np.fft.fftn(h,axes=(1,2,3))[(slice(None),)+ix]/phase
    y=np.fft.fftn(Kh(h),axes=(1,2,3))[(slice(None),)+ix]/phase
    _,_,K,_=symbols(q)
    return dict(L=L,seed=seed,curvature_gauge=maxabs(curvature(Dadj(v))),momentum_closure=maxabs(Dreal(Kh(h))),
        scalar_closure=maxabs(curvature(Kp(p))-div(Dreal(p))),adjoint=abs(inner(Dreal(h),v)-inner(h,Dadj(v)))/(norm(h)*norm(v)),
        hessian=abs(inner(h,Kh(u))-inner(Kh(h),u))/(norm(h)*norm(u)),symbol=maxabs(y-K@x)/max(1.,maxabs(y)),
        broken_kinetic=maxabs(curvature(p)-div(Dreal(p))),broken_potential=maxabs(Dreal(-np.stack([lap(x) for x in h]))))

def dispersion(L):
    k=2*np.pi*np.fft.fftfreq(L); xx=np.meshgrid(k,k,k,indexing='ij'); sq=sum((2*np.sin(x/2))**2 for x in xx); centered=sum(np.sin(x)**2 for x in xx)
    rows=[]
    for d in [(1,0,0),(1,1,0),(1,1,1),(2,1,0)]:
        p=2*np.pi*np.array(d)/L; w=np.linalg.norm(2*np.sin(p/2)); rows.append(dict(direction=d,k=float(np.linalg.norm(p)),omega=float(w),relative_error=float(abs(w/np.linalg.norm(p)-1))))
    return dict(L=L,staggered_zeros=int(np.sum(sq<1e-24)),centered_zeros=int(np.sum(centered<1e-24)),first_shell=rows)

def source(L):
    X=np.meshgrid(*[2*np.pi*np.arange(L)/L]*3,indexing='ij'); rho=np.cos(X[0])+0.2*np.cos(X[1]+X[2]);rho-=rho.mean()
    k=2*np.pi*np.fft.fftfreq(L);q=np.meshgrid(2*np.sin(k/2),2*np.sin(k/2),2*np.sin(k/2),indexing='ij');s=sum(z*z for z in q);mask=s>0
    r=np.fft.fftn(rho);phi=np.zeros_like(r);phi[mask]=-4*np.pi*r[mask]/s[mask];phi=np.fft.ifftn(phi).real
    scale=1e-3/maxabs(phi); phi*=scale; rho*=scale
    h=-2*lift(phi);target=16*np.pi*rho
    return dict(L=L,relative_residual=maxabs(curvature(h)-target)/maxabs(target),nonzero_mean_source_unavoidable_mean_residual=float(16*np.pi),mean_curvature=float(curvature(h).mean()),max_abs_phi=maxabs(phi))

def propagate(L,dt):
    k=2*np.pi*np.array([1,1,1])/L;q=2*np.sin(k/2);w=np.linalg.norm(q)
    D,_,_,_=symbols(q);W=null_space(np.vstack((D,t)));pol=W[:,0]
    X=np.stack(np.meshgrid(*[np.arange(L)]*3,indexing='ij'))
    phase=np.einsum('i,ixyz->xyz',k,X)
    offsets=np.array([np.zeros(3) if i==j else (np.eye(3)[i]+np.eye(3)[j])/2 for i,j in pairs])
    h0=1e-3*pol[:,None,None,None]*np.cos(phase[None]+(offsets@k)[:,None,None,None]);p0=np.zeros_like(h0)
    # Add both independent classes of gauge shifts to the comparison trajectory.
    v=1e-4*rng.normal(size=(3,L,L,L));a=1e-4*rng.normal(size=(L,L,L))
    h1=h0+Dadj(v);p1=p0-Dadj(grad(a))-lift(lap(a))
    def amplitude(h):
        z=np.fft.fftn(h,axes=(1,2,3))[:,1,1,1]*np.exp(-1j*(offsets@k))*2/(L**3*1e-3)
        return W.T@z
    def energy(h,p):return float((np.vdot(p,Kp(p))+np.vdot(h,Kh(h))).real/2)
    tmax=8.;N=round(tmax/dt);maxc=0.;maxg=0.;maxerr=0.;maxen=0.;E0=energy(h0,p0)
    for step in range(N+1):
        exact=np.array([np.cos(w*step*dt),0.]);maxerr=max(maxerr,maxabs(amplitude(h0)-exact));maxg=max(maxg,maxabs(amplitude(h0)-amplitude(h1)))
        maxc=max(maxc,maxabs(curvature(h1)),maxabs(Dreal(p1)));maxen=max(maxen,abs(energy(h1,p1)-E0)/E0)
        if step==N:break
        p0-=dt/2*Kh(h0);h0+=dt*Kp(p0);p0-=dt/2*Kh(h0)
        p1-=dt/2*Kh(h1);h1+=dt*Kp(p1);p1-=dt/2*Kh(h1)
    return dict(L=L,dt=dt,steps=N,max_constraint=maxc,max_gauge_amplitude_difference=maxg,max_wave_error=maxerr,max_relative_energy_error=maxen)

def main():
    modes=[mode(2*np.sin(rng.uniform(-np.pi,np.pi,size=3)/2),i) for i in range(64)]
    real=[realtest(L,i) for L in [4,6,8,12,16] for i in range(4)]
    dis=[dispersion(L) for L in [8,16,32,64,128]]
    src=[source(L) for L in [8,16,32]]
    waves=[propagate(8,dt) for dt in [.1,.05,.025]]
    assert max(r["max_constraint"] for r in waves)<TOL and max(r["max_gauge_amplitude_difference"] for r in waves)<TOL
    assert all(3.8 < a["max_wave_error"]/b["max_wave_error"] < 4.2 for a,b in zip(waves,waves[1:]))
    modekeys=['bracket','closure','reduced','gauge','trajectory_constraints','energy'];realkeys=['curvature_gauge','momentum_closure','scalar_closure','adjoint','hessian','symbol']
    ms={k:max(r[k] for r in modes) for k in modekeys};rs={k:max(r[k] for r in real) for k in realkeys}
    controls={k:min(r['negative_controls'][k] for r in modes) for k in modes[0]['negative_controls']}
    assert all(r['rank']==4 and r['physical_pairs']==2 for r in modes)
    assert max(ms.values())<TOL and max(rs.values())<TOL
    assert min(controls.values())>1e-6 and min(r['broken_kinetic'] for r in real)>1e-6 and min(r['broken_potential'] for r in real)>1e-6
    assert all(r['staggered_zeros']==1 and r['centered_zeros']==8 for r in dis)
    assert max(r['relative_residual'] for r in src)<TOL
    result=dict(status='PASS_BOUNDED_REFERENCE',python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,seed=15026,tolerance=TOL,mode_max=ms,real_max=rs,negative_control_min=controls,modes=modes,real_space=real,dispersion=dis,static_sources=src,real_space_propagation=waves)
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','python','numpy','scipy','mode_max','real_max','negative_control_min','dispersion','static_sources','real_space_propagation']},indent=2)); print('All assertions passed.')
if __name__=='__main__':main()
