"""Reduced Newtonian interaction plus two quantized TT modes; not full GR."""
import json, platform
from pathlib import Path
import numpy as np
import scipy
from scipy.sparse import diags,eye,kron
from scipy.sparse.linalg import expm_multiply
from scipy.linalg import eigh
from numpy.polynomial.legendre import leggauss
from qg015_reference import Kh,lap,Dreal,curvature,tr,symbols
OUT=Path(__file__).resolve().parent
M=5.; L=13; K=2*np.pi*6/L; W=2*np.sin(K/2); TIMES=np.linspace(0,12,61)
def mx(x):return float(np.max(np.abs(x)))
def field():
    def filt(seed):
        b=Kh(seed);v=(Kh(b)-np.stack([lap(x) for x in b]))/2
        return v/np.linalg.norm(v)
    a=np.zeros((6,L,L,L));b=a.copy();a[0,0,0,0]=1;a[1,0,0,0]=-1;a[3,0,0,0]=.5
    b[0,6,0,0]=.6;b[1,6,0,0]=-.6;b[3,6,0,0]=-1
    sa,sb=filt(a),filt(b);z=np.arange(L)[None,None,None,:]
    plus=np.array([1,-1,0,0,0,0])/np.sqrt(2);cross=np.array([0,0,0,1,0,0])
    modes=np.array([np.broadcast_to(np.sqrt(2/L**3)*p[:,None,None,None]*np.cos(K*z),sa.shape) for p in (plus,cross)])
    flat=modes.reshape(2,-1);c=np.stack([flat@sa.ravel(),flat@sb.ravel()]);scale=mx(c);c/=scale
    d,cur,km,cons=symbols(np.array([0,0,W]));pol=np.stack([plus,cross],axis=1)
    return c,dict(L=L,k=K,omega=W,rank=int(np.linalg.matrix_rank(cons)),couplings=c.tolist(),raw_projection_scale=scale,overlap=int(np.sum((np.max(abs(sa),axis=0)>1e-12)&(np.max(abs(sb),axis=0)>1e-12))),source_divergence=max(mx(Dreal(sa)),mx(Dreal(sb))),source_trace=max(mx(tr(sa)),mx(tr(sb))),bad_source_divergence=mx(Dreal(a)),mode_divergence=max(mx(Dreal(u)) for u in modes),mode_curvature=max(mx(curvature(u)) for u in modes),mode_orthogonality=mx(flat@flat.T-np.eye(2)),mode_stiffness=mx(km@pol-W**2*pol))
C,FD=field()
def model(n,cut,D,g,lam,kind,prep):
    dx=2/(n+1);x=-1+dx*np.arange(1,n+1)
    T=diags([-np.ones(n-1),2*np.ones(n),-np.ones(n-1)],[-1,0,1],format='csr')/(2*M*dx**2)
    ev,U=eigh(T.toarray());q=x*x-np.sum(U[:,0]**2*x*x)
    A=U[:,2] if prep=='excited' else (U[:,0]+U[:,1])/np.sqrt(2);B=U[:,0]
    r=D+x[None,:]-x[:,None];v=-g/r;factor=2 if kind=='double_scalar' else 1
    if kind=='force_only':v=np.broadcast_to(-g/D+g*(x[None,:]-x[:,None])/D**2,r.shape).copy()
    h0=(kron(T,eye(n))+kron(eye(n),T)+diags((factor*v).ravel())).tocsr()
    f=lam*(q[:,None,None]*C[0]+q[None,:,None]*C[1]);f=f.reshape(n*n,2)
    if kind=='double_TT':h0=h0+diags(-np.sum(f*f,axis=1)/W)
    a=diags(np.sqrt(np.arange(1,cut)),1,shape=(cut,cut),format='csr');num=diags(np.arange(cut));bops=[kron(a,eye(cut)),kron(eye(cut),a)]
    hm=kron(h0,eye(cut*cut)).tocsr();hg=kron(eye(n*n),W*(kron(num,eye(cut))+kron(eye(cut),num))).tocsr()
    hi=sum(kron(diags(f[:,i]),bops[i]+bops[i].T) for i in range(2)).tocsr();H=hm+hg+hi
    psi=np.zeros((n,n,cut,cut),complex);psi[:,:,0,0]=np.outer(A,B)
    return x,T,r,v,f,H,hm,hg,hi,psi

def run(conf):
    n,cut,D,g,lam,kind,prep=conf;x,T,r,v,f,H,hm,hg,hi,psi=model(*conf);z0=psi.ravel()
    states=expm_multiply(-1j*H,z0,start=0,stop=12,num=len(TIMES),traceA=-1j*H.diagonal().sum())
    out=[];tval=T[0,1];null=np.zeros(n+1);cont=power=tail=0.
    for z in states:
        w=z.reshape(psi.shape);p=abs(w)**2;zz=w.reshape(n*n,cut*cut);rho=zz@zz.conj().T
        rt=rho.reshape(n,n,n,n).transpose(0,3,2,1).reshape(n*n,n*n)
        neg=float(np.maximum(-np.linalg.eigvalsh(rt),0).sum());rf=zz.conj().T@zz;purity=float(np.trace(rf@rf).real)
        tail=max(tail,float(p[:,:,-1,:].sum()),float(p[:,:,:,-1].sum()));hz=H@z;dp=2*np.real(w.conj()*(-1j*hz).reshape(psi.shape))
        ja=null.copy();jb=null.copy();ja[1:-1]=-2*np.imag(tval*np.sum(w[:-1].conj()*w[1:],axis=(1,2,3)));jb[1:-1]=-2*np.imag(tval*np.sum(w[:,:-1].conj()*w[:,1:],axis=(0,2,3)))
        cont=max(cont,mx(dp.sum((1,2,3))-(ja[:-1]-ja[1:])),mx(dp.sum((0,2,3))-(jb[:-1]-jb[1:])))
        os=[O@z for O in (hm,hg,hi)];es=np.array([np.vdot(z,o).real for o in os]);rates=np.array([2*np.imag(np.vdot(o,hz)) for o in os]);power=max(power,abs(float(rates.sum())))
        occ=[float(np.sum(p*np.arange(cut)[None,None,:,None])),float(np.sum(p*np.arange(cut)[None,None,None,:]))]
        out.append([float(np.sum(p.sum((2,3))*r)),neg,1-purity,*map(float,es),float(es.sum()),float(p.sum()),*occ,*map(float,rates)])
    a=np.array(out);dt=TIMES[1]-TIMES[0]
    return dict(config=list(conf),dimension=H.shape[0],support_gap=D-2,peak_AB_negativity=float(a[:,1].max()),peak_field_linear_entropy=float(a[:,2].max()),peak_field_energy=float(a[:,4].max()),min_matter_energy_change=float(np.min(a[:,3]-a[0,3])),peak_occupation=[float(a[:,8].max()),float(a[:,9].max())],final_separation=float(a[-1,0]),early_separation=float(a[1,0]),norm_error=mx(a[:,7]-1),energy_error=mx(a[:,6]-a[0,6]),number_continuity=cont,power_balance=power,rate_finite_difference=mx((a[2:,3:6]-a[:-2,3:6])/(2*dt)-a[1:-1,10:13]),cutoff_tail=tail),a

def static_checks():
    u,w=leggauss(20);rad=(u+1)*.1;wr=w*.1;mu,wm=leggauss(20);rows=[];g=2.
    for d in (1.1,2.,3.,4.,6.):
        cross=-g*1.5/.2**3*np.sum(wr[:,None]*rad[:,None]**2*wm[None,:]/np.sqrt(d*d+rad[:,None]**2-2*d*rad[:,None]*mu[None,:]));rows.append([d,float(cross),-g/d,float(2*cross)])
    a=diags(np.sqrt(np.arange(1,32)),1,shape=(32,32)).toarray();num=np.diag(np.arange(32));shifts=[]
    for f in ((.1,.2),(.5,-.3),(1.,.7)):
        e=sum(np.linalg.eigvalsh(W*num+s*(a+a.T))[0] for s in f);pred=-sum(s*s for s in f)/W;shifts.append([list(f),float(e),float(pred),float(e+pred)])
    return dict(scalar=rows,scalar_error=max(abs(x[1]-x[2]) for x in rows),TT=shifts,TT_error=max(abs(x[1]-x[2]) for x in shifts))
def dense_check():
    c=(3,3,3.,2.,.5,'full','excited');*_,H,hm,hg,hi,psi=model(*c);z=psi.ravel();e,U=eigh(H.toarray());t=1.3
    return mx(U@(np.exp(-1j*t*e)*(U.conj().T@z))-expm_multiply(-1j*t*H,z))
def main():
    cfg=[(11,6,3.,g,l,'full','excited') for g,l in ((0.,0.),(2.,0.),(0.,.8),(2.,.4),(2.,.8),(2.,1.2),(8.,.8))]
    cfg += [(11,6,d,2.,.8,'full','excited') for d in (4.,6.)]
    cfg += [(11,6,3.,2.,.8,k,'excited') for k in ('double_scalar','double_TT','force_only')]
    cfg += [(11,c,3.,2.,.8,'full','excited') for c in (4,8)]+[(n,6,3.,2.,.8,'full','excited') for n in (7,15)]+[(11,6,3.,2.,.8,'full','superposition')]
    rows=[];series=[]
    for c in cfg:
        r,a=run(c);rows.append(r);series.append(a.tolist());print(c,'neg',r['peak_AB_negativity'],'field E',r['peak_field_energy'],'matter min',r['min_matter_energy_change'],'tail',r['cutoff_tail'],flush=True)
    static=static_checks();dense=dense_check();base=rows[4];diff4=mx(np.array(series[12])[:,:6]-np.array(series[4])[:,:6]);diff8=mx(np.array(series[13])[:,:6]-np.array(series[4])[:,:6])
    checks=dict(scalar_once=static['scalar_error']<1e-12,TT_static_shift=static['TT_error']<1e-12,TT_source=FD['source_divergence']<1e-12 and FD['source_trace']<1e-12 and FD['bad_source_divergence']>1e-3,disjoint_source_envelopes=FD['overlap']==0,TT_modes=max(FD['mode_divergence'],FD['mode_curvature'],FD['mode_stiffness'],FD['mode_orthogonality'])<1e-12 and FD['rank']==4,unitarity=max(r['norm_error'] for r in rows)<1e-9,energy=max(r['energy_error'] for r in rows)<1e-9,continuity=max(r['number_continuity'] for r in rows)<1e-12,power=max(r['power_balance'] for r in rows)<1e-12,null=rows[0]['peak_AB_negativity']<1e-10 and rows[0]['peak_field_energy']<1e-12,scalar_only=rows[1]['peak_AB_negativity']>1e-5 and rows[1]['peak_field_energy']<1e-12,TT_only=rows[2]['peak_field_energy']>1e-3,combined=base['peak_field_energy']>1e-3 and base['peak_AB_negativity']>1e-5 and base['peak_field_linear_entropy']>1e-4,cutoff_refinement=diff8<diff4,dense=dense<1e-10)
    result=dict(runtime=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),times=TIMES.tolist(),columns=['separation','AB_negativity','field_linear_entropy','matter_energy','field_energy','interaction_energy','total_energy','norm','plus_occupation','cross_occupation','matter_power','field_power','interaction_power'],field=FD,rows=rows,series=series,static=static,dense_error=dense,cutoff=dict(diff4_6=diff4,diff6_8=diff8),checks=checks)
    (OUT/'results.json').write_text(json.dumps(result,separators=(',',':'))+'\n');print('CHECKS',checks,'static',static,'dense',dense,'cutoff',result['cutoff'],flush=True)
    if not all(checks.values()):raise SystemExit('FAIL: retain results')
if __name__=='__main__':main()
