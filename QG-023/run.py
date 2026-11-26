"""QG-023: tree-level scalar Ward identities and classical retarded response.
Run: python run.py. All units c=hbar=1; metric signature (-+++).
"""
import json, platform
from pathlib import Path
import numpy as np
from numpy.polynomial import Polynomial

OUT=Path(__file__).resolve().parent
eta=np.diag([-1.,1.,1.,1.]); rng=np.random.default_rng(23001)
def dot(a,b): return float(a@eta@b)
def trace(A): return float(np.sum(eta*A))
def contract(A,B): return float(np.sum(A*(eta@B@eta)))
def vertex(p,r,m,broken=False):
    A=np.outer(r,p)+np.outer(p,r)
    return A if broken else A-eta*(dot(p,r)+m*m)
def div(q,A): return (eta@q)@A
def numerator(A,B): return contract(A,B)-trace(A)*trace(B)/2
def boost(v):
    v=np.asarray(v); b2=v@v; gamma=1/np.sqrt(1-b2)
    L=np.eye(4); L[0,0]=gamma; L[0,1:]=gamma*v;L[1:,0]=gamma*v
    if b2: L[1:,1:]+=(gamma-1)*np.outer(v,v)/b2
    return L
def normvec():
    v=rng.normal(size=3);return v/np.linalg.norm(v)
def pair(m1,m2,p,u,w):
    E1=np.sqrt(m1*m1+p*p); E2=np.sqrt(m2*m2+p*p)
    return np.r_[E1,p*u],np.r_[E1,p*w],np.r_[E2,-p*u],np.r_[E2,-p*w]
def decompose(A,B,q):
    n=q[1:]/np.linalg.norm(q[1:]); P=np.eye(3)-np.outer(n,n)
    SA=P@A[1:,1:]@P; SB=P@B[1:,1:]@P
    sA=np.trace(SA);sB=np.trace(SB)
    TA=SA-P*sA/2;TB=SB-P*sB/2
    return [float(A[0,0]*B[0,0]/2),float((A[0,0]*sB+B[0,0]*sA)/2),float(-2*A[0,1:]@P@B[0,1:]),float(np.sum(TA*TB))]
def sample_scattering():
    rows=[]
    for i in range(128):
        m1,m2=rng.uniform(.5,2.5,2); p=rng.uniform(.05,1.8);u,w=normvec(),normvec()
        if np.linalg.norm(u-w)<.15: w=-u
        a,b,c,d=pair(m1,m2,p,u,w);q=b-a
        A=vertex(a,b,m1);B=vertex(c,d,m2);N=numerator(A,B)
        v=rng.uniform(.05,.75)*normvec();L=boost(v)
        ap,bp,cp,dp=(L@z for z in (a,b,c,d)); qp=bp-ap
        Ap=vertex(ap,bp,m1); Bp=vertex(cp,dp,m2)
        xi=rng.normal(size=4);xi/=np.linalg.norm(xi)
        # pure gauge h_{mu nu} = q_mu xi_nu + q_nu xi_mu
        gauge=lambda T: abs(2*div(qp,T)@xi)/max(1,np.linalg.norm(qp)*np.linalg.norm(T))
        longitudinal=lambda X,Y: dot(div(qp,X),div(qp,Y))/dot(qp,qp)
        scale=max(1.,abs(N)); norm=max(1.,np.linalg.norm(qp)*max(np.linalg.norm(Ap),np.linalg.norm(Bp)))
        A_bad=vertex(ap,bp,m1,True);B_bad=vertex(cp,dp,m2,True)
        bo=bp.copy();bo[0]+=.12; do=dp.copy();do[0]-=.12;qo=bo-ap
        Ao=vertex(ap,bo,m1);Bo=vertex(cp,do,m2)
        predA=(dot(bo,bo)+m1*m1)*ap-(dot(ap,ap)+m1*m1)*bo
        predB=(dot(do,do)+m2*m2)*cp-(dot(cp,cp)+m2*m2)*do
        parts=decompose(A,B,q)
        rows.append(dict(i=i,m1=m1,m2=m2,p=p,boost=float(np.linalg.norm(v)),transfer_energy=qp[0],q2=dot(qp,qp),
           shell=max(abs(dot(z,z)+m*m) for z,m in ((ap,m1),(bp,m1),(cp,m2),(dp,m2))),
           conservation=float(max(np.linalg.norm(div(qp,Ap)),np.linalg.norm(div(qp,Bp)))/norm),
           gauge=max(gauge(Ap),gauge(Bp)),boost_amplitude=abs(numerator(Ap,Bp)-N)/scale,
           gauge_shift=abs(longitudinal(Ap,Bp))/scale, bad_gauge_shift=abs(longitudinal(A_bad,B_bad))/scale,
           bad_vertex_conservation=float(max(np.linalg.norm(div(qp,A_bad)),np.linalg.norm(div(qp,B_bad)))/norm),
           offshell_identity=float(max(np.linalg.norm(div(qo,Ao)-predA),np.linalg.norm(div(-qo,Bo)-predB))/norm),
           offshell_conservation=float(max(np.linalg.norm(div(qo,Ao)),np.linalg.norm(div(-qo,Bo)))/norm),
           decomposition=abs(sum(parts)-N)/scale,parts=parts,numerator=N))
    return rows

def static_limit():
    rows=[]
    for p in [.001,.003,.01,.03,.1,.3,.6]:
        for m1,m2 in [(1,1),(.7,1.8),(2,.8)]:
            for theta in [.4,1.1,2.4]:
                a,b,c,d=pair(m1,m2,p,np.array([1,0,0]),np.array([np.cos(theta),np.sin(theta),0]));q=b-a
                A=vertex(a,b,m1);B=vertex(c,d,m2); parts=decompose(A,B,q)
                denom=2*a[0]*c[0]*m1*m2
                rows.append(dict(p=p,m1=m1,m2=m2,theta=theta,ratio=numerator(A,B)/denom,
                    no_trace_ratio=contract(A,B)/denom,tt_only_ratio=parts[3]/denom,
                    double_newton_ratio=(numerator(A,B)+denom)/denom))
    return rows

# Classical retarded trace-reversed field of a distributional compact-in-time
# quadrupole source, F=f(t) delta^3(x). This is not physical positive-energy matter.
M=np.array([[1.,.35,0],[.35,-1.,0],[0,0,0]])
Tdur=2.; G=1e-6
poly=Polynomial([0,4/Tdur,-4/Tdur**2])**6
pd=[poly.deriv(i) for i in range(3)]
def pulse(u):
    if u<=0 or u>=Tdur: return 0.,0.,0.
    # evaluate in centered variable for stability near trailing boundary
    s=min(u,Tdur-u); f,fp,fpp=(float(p(s)) for p in pd)
    return f,fp if u<=Tdur/2 else -fp,fpp

def field(z,mode='retarded',ablation=False):
    t=z[0];x=z[1:];r=np.linalg.norm(x);n=x/r
    sign={'retarded':-1,'advanced':1,'instant':0}[mode]
    f,fp,fpp=pulse(t+sign*r)
    Ur=sign*fp/r-f/r**2
    Urr=sign**2*fpp/r-2*sign*fp/r**2+2*f/r**3
    Utr=sign*fpp/r-fp/r**2
    H=np.zeros((4,4));H[0,0]=np.sum(M*(Urr*np.outer(n,n)+Ur/r*(np.eye(3)-np.outer(n,n))))
    if not ablation: H[0,1:]=H[1:,0]=-M@n*Utr
    H[1:,1:]=M*fpp/r
    return 4*G*H

def metric(z,mode='retarded',ablation=False):
    H=field(z,mode,ablation);return eta@H@eta-eta*trace(H)/2

def d1(fun,z,i,h):
    e=np.eye(4)[i]*h;return (fun(z+e)-fun(z-e))/(2*h)
def d2(fun,z,i,j,h):
    e=np.eye(4)[i]*h;f=np.eye(4)[j]*h
    if i==j: return (fun(z+e)-2*fun(z)+fun(z-e))/h**2
    return (fun(z+e+f)-fun(z+e-f)-fun(z-e+f)+fun(z-e-f))/(4*h*h)
def ward_field(fun,z,h): return sum(d1(fun,z,i,h)[i,:] for i in range(4))
def wave(fun,z,h): return sum(eta[i,i]*d2(fun,z,i,i,h) for i in range(4))
def tidal(fun,z,h):
    return np.array([[.5*(d2(fun,z,0,i,h)[0,j]+d2(fun,z,j,0,h)[i,0]-d2(fun,z,j,i,h)[0,0]-d2(fun,z,0,0,h)[i,j]) for j in range(1,4)] for i in range(1,4)])
def pure_gauge(z):
    k=np.array([.7,-.3,.8,1.1]);v=np.array([.3,-.2,.5,.4]);return 4*G*(np.outer(k,v)+np.outer(v,k))*np.cos(k@z)

def causal():
    rows=[]; h=.002
    for r in [2.,4.,8.]:
        for t in np.linspace(-.5,11,231):
            z=np.array([t,0.,0.,r]); H=field(z);Ea=tidal(metric,z,h);Eadv=tidal(lambda a:metric(a,'advanced'),z,h);Einst=tidal(lambda a:metric(a,'instant'),z,h)
            rows.append(dict(r=r,t=float(t),metric=float(np.linalg.norm(H)/(4*G)),plus=float((Ea[0,0]-Ea[1,1])/(8*G)),cross=float(Ea[0,1]/(4*G)),tidal=float(np.linalg.norm(Ea)/(4*G)),advanced=float(np.linalg.norm(Eadv)/(4*G)),instant=float(np.linalg.norm(Einst)/(4*G))))
    return rows

def refinement():
    # fixed random exterior spacetime points strictly inside pulse window
    pts=[]
    for _ in range(24):
        r=rng.uniform(2,7);pts.append(np.r_[r+rng.uniform(.35,1.65),r*normvec()])
    rows=[]
    for h in [.04,.02,.01,.005]:
        rr=[]
        for z in pts:
            curv=tidal(metric,z,h); shifted=tidal(lambda x:metric(x)+pure_gauge(x),z,h)
            rr.append(dict(z=z.tolist(),ward=float(np.linalg.norm(ward_field(field,z,h))/(4*G)),wave=float(np.linalg.norm(wave(field,z,h))/(4*G)),
                bad_ward=float(np.linalg.norm(ward_field(lambda x:field(x,ablation=True),z,h))/(4*G)),gauge_curvature=float(np.linalg.norm(curv-shifted)/(4*G))))
        rows.append(dict(h=h,points=rr,**{k:max(p[k] for p in rr) for k in ['ward','wave','bad_ward','gauge_curvature']}))
    return rows

def source_identity():
    rows=[]
    for _ in range(64):
        om=rng.uniform(-3,3);k=rng.normal(size=3);q=np.r_[om,k]
        # Fourier convention exp(i k.x-i omega t): source derivatives as above
        S=np.zeros((4,4)); S[0,0]=-k@M@k;S[0,1:]=S[1:,0]=-om*(M@k);S[1:,1:]=-om**2*M
        B=S.copy();B[0,1:]=B[1:,0]=0
        rows.append(dict(q=q.tolist(),ward=float(np.linalg.norm(div(q,S))),bad=float(np.linalg.norm(div(q,B)))))
    return rows

def main():
    s=sample_scattering();n=static_limit();c=causal();r=refinement();src=source_identity()
    maxima={k:max(x[k] for x in s) for k in ['shell','conservation','gauge','boost_amplitude','gauge_shift','bad_gauge_shift','bad_vertex_conservation','offshell_identity','offshell_conservation','decomposition']}
    static=[]
    for p in sorted({x['p'] for x in n}):
        z=[x for x in n if x['p']==p];static.append(dict(p=p,relative_error=max(abs(x['ratio']-1) for x in z),no_trace_min=min(x['no_trace_ratio'] for x in z),tt_only_max=max(abs(x['tt_only_ratio']) for x in z),double_newton_min=min(x['double_newton_ratio'] for x in z)))
    # Tidal stencil probes +/- 2h at worst. Exclude a conservative 0.02 shell around the front.
    detector=[]
    for d in [2.,4.,8.]:
        z=[x for x in c if x['r']==d];pre=[x for x in z if x['t']<d-.02];active=[x for x in z if d+.02<x['t']<d+Tdur-.02];after=[x for x in z if x['t']>d+Tdur+.02]
        detector.append(dict(r=d,pre_metric=max(x['metric'] for x in pre),pre_tidal=max(x['tidal'] for x in pre),post_tidal=max(x['tidal'] for x in after),peak_tidal=max(x['tidal'] for x in active),peak_plus=max(abs(x['plus']) for x in active),peak_cross=max(abs(x['cross']) for x in active),instant_pre=max(x['instant'] for x in pre),advanced_pre=max(x['advanced'] for x in pre),max_metric=max(x['metric'] for x in z)*4*G))
    ratios={k:[a[k]/b[k] for a,b in zip(r,r[1:])] for k in ['ward','wave','gauge_curvature']}
    checks={
       'on_shell':maxima['shell']<1e-11,'vertex_ward':maxima['conservation']<1e-12,'pure_gauge_decoupling':maxima['gauge']<1e-12,'boost_invariance':maxima['boost_amplitude']<1e-11,
       'longitudinal_propagator_control':maxima['gauge_shift']<1e-22 and maxima['bad_gauge_shift']>1e-3,'missing_vertex_trace_detected':maxima['bad_vertex_conservation']>1e-3,
       'offshell_identity':maxima['offshell_identity']<1e-12 and maxima['offshell_conservation']>1e-3,'static_channel_decomposition':maxima['decomposition']<1e-11,
       'newton_limit':static[0]['relative_error']<1e-4 and static[0]['tt_only_max']<1e-8,'double_count_and_trace_controls':static[0]['double_newton_min']>1.99 and static[0]['no_trace_min']>1.99,
       'source_conservation':max(x['ward'] for x in src)<1e-12 and max(x['bad'] for x in src)>1,
       'retarded_support':all(x['pre_tidal']==x['pre_metric']==x['post_tidal']==0 and x['peak_plus']>0 and x['peak_cross']>0 for x in detector),
       'nonretarded_control':all(x['instant_pre']>0 for x in detector) and max(x['advanced_pre'] for x in detector)>0,
       'exterior_ward_convergence':min(ratios['ward'])>3.2,'exterior_wave_convergence':min(ratios['wave'])>3.2,'curvature_gauge_convergence':min(ratios['gauge_curvature'])>3.2,
       'missing_momentum_control':r[-1]['bad_ward']>100*r[-1]['ward'],'weak_metric':max(x['max_metric'] for x in detector)<.01}
    checks={k:bool(v) for k,v in checks.items()}
    result=dict(environment=dict(python=platform.python_version(),numpy=np.__version__),seed=23001,checks=checks,scattering=s,scattering_max=maxima,static=n,static_summary=static,source=src,detector_trajectories=c,detectors=detector,refinement=r,refinement_ratios=ratios)
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    summary=dict(environment=result['environment'],checks=checks,scattering_max=maxima,static_summary=static,detectors=detector,refinement=[{k:v for k,v in x.items() if k!='points'} for x in r],ratios=ratios,source_max={k:max(x[k] for x in src) for k in ['ward','bad']})
    print(json.dumps(summary,indent=2));assert all(checks.values()),[k for k,v in checks.items() if not v]
if __name__=='__main__':main()
