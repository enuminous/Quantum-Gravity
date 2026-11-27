import numpy as np
from itertools import permutations
eta=np.diag([1.,-1.,-1.,-1.])
def dot(a,b):return a@eta@b
def outer(a,b):return np.outer(eta@a,eta@b)
def rest(m,w,theta):
    c,s=np.cos(theta),np.sin(theta); wp=w/(1+w/m*(1-c)); p=np.array([m,0,0,0.]); k=w*np.array([1,0,0,1.]); l=wp*np.array([1,s,0,c]); P=p+k-l
    e=(np.array([0,1,0,0.])+1j*np.array([0,0,1,0.]))/np.sqrt(2); f=(np.array([0,c,0,-s])-1j*np.array([0,0,1,0.]))/np.sqrt(2)
    return p,P,k,l,e,f

def bracket(p,P,k,l,e,f):
    a,b=dot(p,k),dot(p,l)
    return dot(p,e)*dot(P,f)/a-dot(P,e)*dot(p,f)/b-dot(e,f)
def factorized(p,P,k,l,e,f):
    a,b,c=dot(p,k),dot(p,l),dot(k,l)
    return .5*a*b/c*bracket(p,P,k,l,e,f)**2

def scalar_vertex(p,P,h,m):
    return -p@h@P + .5*np.trace(eta@h)*(dot(p,P)-m*m)

def cubic(hs,qs):
    inv=[-eta@h@eta for h in hs]; dens=[inv[i]+.5*np.trace(eta@h)*eta for i,h in enumerate(hs)]
    Cs=[]
    for h,q in zip(hs,qs):
        d=1j*eta@q
        # C[d,m,n] = (partial_m h_dn + partial_n h_dm - partial_d h_mn)/2
        C=.5*(np.einsum('m,dn->dmn',d,h)+np.einsum('n,dm->dmn',d,h)-np.einsum('d,mn->dmn',d,h))
        Cs.append(C)
    G=[np.einsum('ad,dmn->amn',eta,C) for C in Cs]
    def pair(g,A,B):
        return np.einsum('mn,amb,bna->',g,A,B)-np.einsum('mn,amn,bab->',g,A,B)
    val=0j
    for a,b,c in permutations(range(3)):
        G2=np.einsum('ad,dmn->amn',inv[a],Cs[b])
        val+=pair(dens[a],G[b],G[c])+pair(eta,G2,G[c])+pair(eta,G[c],G2)
    return -2*val

def diagrams_tensor(p,P,k,l,h1,h2,m):
    a,b,c=dot(p,k),dot(p,l),dot(k,l)
    sv=-scalar_vertex(p,p+k,h1,m)*scalar_vertex(p+k,P,h2,m)/(2*a)
    uv=-scalar_vertex(p,p-l,h2,m)*scalar_vertex(p-l,P,h1,m)/(-2*b)
    # determinant and densitized inverse cross coefficient for arbitrary tensors
    tr1,tr2=np.trace(eta@h1),np.trace(eta@h2)
    det=.25*tr1*tr2-.5*np.trace(eta@h1@eta@h2)
    C=eta@h1@eta@h2@eta+eta@h2@eta@h1@eta-.5*tr1*eta@h2@eta-.5*tr2*eta@h1@eta+det*eta
    contact=(eta@p)@C@(eta@P)-det*m*m
    V=-.5*(np.outer(p,P)+np.outer(P,p)-eta*(dot(p,P)-m*m))
    h=eta@V@eta-.5*eta*np.trace(eta@V)
    tv=-cubic([h1,h2,h],[k,-l,l-k])/(-2*c)
    return np.array([sv,uv,contact,tv])

def diagrams(p,P,k,l,e,f,m):
    return diagrams_tensor(p,P,k,l,outer(e,e),outer(f,f),m)

def field_strength(p,k,l,e,f):
    a,b,c=dot(p,k),dot(p,l),dot(k,l)
    F=np.outer(k,eta@e)-np.outer(e,eta@k)
    H=np.outer(l,eta@f)-np.outer(f,eta@l)
    return (p@eta@F@H@p)**2/(2*a*b*c)

def boost(v):
    v=np.asarray(v); b2=v@v; g=1/np.sqrt(1-b2)
    L=np.eye(4); L[0,0]=g; L[0,1:]=g*v; L[1:,0]=g*v
    L[1:,1:]+=(g-1)*np.outer(v,v)/b2
    return L

def gauge(k,xi): return outer(k,xi)+outer(xi,k)
def error(a,b,scale):return float(abs(a-b)/scale)
def C(z):return [float(z.real),float(z.imag)]

if __name__=='__main__':
    import sys,json,time
    from pathlib import Path
    import scipy
    start=time.monotonic(); root=Path(__file__).parent; rng=np.random.default_rng(25001); rows=[]
    def uniform_direction():
        v=rng.normal(size=3);return v/np.linalg.norm(v)
    for case in range(128):
        m=rng.uniform(.5,3); ratio=np.exp(rng.uniform(np.log(.03),np.log(2))); w=m*ratio; theta=rng.uniform(.2,np.pi-.2)
        p,P,k,l,e0,f0=rest(m,w,theta); L=boost(uniform_direction()*rng.uniform(.1,.8)); pp,PP,kk,ll=(L@v for v in (p,P,k,l))
        shell=max(abs(dot(pp,pp)-m*m),abs(dot(PP,PP)-m*m),abs(dot(kk,kk)),abs(dot(ll,ll)),abs(pp+kk-PP-ll).max(),abs(dot(pp,kk)-dot(pp,ll)-dot(kk,ll)))
        for hi in [1,-1]:
            for hf in [1,-1]:
                e=e0 if hi==1 else e0.conj(); f=f0 if hf==1 else f0.conj(); ee,ff=L@e,L@f
                z1=(rng.normal()+1j*rng.normal())*.5;z2=(rng.normal()+1j*rng.normal())*.5
                # Dimensionless gauge shifts; normalize by positive incoming energies in boosted frame.
                es=ee+z1*kk/kk[0]; fs=ff+z2*ll/ll[0]
                d0=diagrams(p,P,k,l,e,f,m);d=diagrams(pp,PP,kk,ll,ee,ff,m);ds=diagrams(pp,PP,kk,ll,es,fs,m)
                ref=factorized(p,P,k,l,e,f);scale=max(1,abs(ref)); M=d.sum()
                xi1=rng.normal(size=4)+1j*rng.normal(size=4);xi2=rng.normal(size=4)+1j*rng.normal(size=4)
                gi=diagrams_tensor(pp,PP,kk,ll,gauge(kk/kk[0],xi1),outer(ff,ff),m).sum()
                gf=diagrams_tensor(pp,PP,kk,ll,outer(ee,ee),gauge(ll/ll[0],xi2),m).sum()
                rev=diagrams(P,p,l,k,f.conj(),e.conj(),m).sum()
                angle=(np.cos(theta)+hi*hf)/2
                exact=m*m/(2*(1-np.cos(theta)))*angle**2
                errs=dict(diagram_factorization=error(M,ref,scale),field_strength=error(field_strength(pp,kk,ll,ee,ff),ref,scale),lorentz=error(M,d0.sum(),scale),gauge_shift=error(ds.sum(),M,scale),ward_in=float(abs(gi)/scale),ward_out=float(abs(gf)/scale),reverse=error(rev,d0.sum().conj(),scale),rest_helicity=error(d0.sum(),exact,scale),shell=float(shell),polarization=float(max(abs(dot(kk,ee)),abs(dot(ll,ff)),abs(dot(ee,ee)),abs(dot(ff,ff)))))
                # Normalize all leave-one-out gauge differences to the same full-amplitude scale.
                ablate={name:error(ds.sum()-ds[j],M-d[j],scale) for j,name in enumerate(['s','u','contact','t'])}
                rows.append(dict(case=case,m=float(m),omega_over_m=float(ratio),theta=float(theta),hi=hi,hf=hf,errors=errs,ablations=ablate,rest_diagrams=[C(x) for x in d0],boosted_diagrams=[C(x) for x in d],shifted_diagrams=[C(x) for x in ds],reference=C(ref)))
    # Cross sections in units G^2*m^2, derived independently from the complete diagram amplitude and phase space.
    def cross(theta,x):
        m=1.0; p,P,k,l,e,f=rest(m,x,theta); wp=l[0]; vals=[]
        for ee in (e,e.conj()):
            for ff in (f,f.conj()): vals.append(abs(diagrams(p,P,k,l,ee,ff,m).sum())**2)
        # kappa=1, G=1/(32*pi). Divide by G^2*m^2.
        num=(wp/x)**2*.5*sum(vals)*16/m**4
        expected=(wp/x)**2*(np.cos(theta/2)**8+np.sin(theta/2)**8)/np.sin(theta/2)**4
        return float(num),float(expected)
    crossrows=[]
    for x in [1e-4,.01,.1,1,2]:
        for t in np.deg2rad([5,15,30,45,60,90,120,150,175]):
            num,expected=cross(t,x); low=(np.cos(t/2)**8+np.sin(t/2)**8)/np.sin(t/2)**4
            crossrows.append(dict(omega_over_m=x,angle_deg=float(np.degrees(t)),computed=num,expected=expected,relative_error=abs(num/expected-1),low_energy_error=abs(num/low-1)))
    fw=[]
    for t in [.2,.1,.05,.02,.01,.005]:
        num,exp=cross(t,1e-4); fw.append(dict(theta=t,scaled=num*t**4/16,error=abs(num*t**4/16-1)))
    maxima={key:max(r['errors'][key] for r in rows) for key in rows[0]['errors']}
    ablations={key:dict(max=max(r['ablations'][key] for r in rows),median=float(np.median([r['ablations'][key] for r in rows])),detected=sum(r['ablations'][key]>1e-3 for r in rows)) for key in rows[0]['ablations']}
    cs=max(r['relative_error'] for r in crossrows);low=max(r['low_energy_error'] for r in crossrows if r['omega_over_m']==1e-4)
    checks={k:bool(v<(1e-10 if k=='shell' else 1e-9)) for k,v in maxima.items()}
    checks.update({f'detect_missing_{key}':v['max']>1e-3 for key,v in ablations.items()});checks.update(cross_section=cs<1e-9,low_energy=low<5e-4,forward=fw[-1]['error']<1e-4)
    checks={k:bool(v) for k,v in checks.items()}
    summary=dict(environment=dict(python=sys.version.split()[0],numpy=np.__version__,scipy=scipy.__version__),events=128,helicity_cases=len(rows),max_errors=maxima,ablations=ablations,cross_section_max_relative_error=cs,low_energy_max_relative_error=low,forward=fw,checks=checks,cross_sections=crossrows)
    (root/'results.json').write_text(json.dumps(dict(summary=summary,rows=rows),indent=2)+'\n');(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='cross_sections'},indent=2));print('elapsed seconds',time.monotonic()-start);assert all(checks.values()),checks
