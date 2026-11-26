"""Independent dense spectral evolution of a small two-body lattice."""
import json
import numpy as np
from scipy.linalg import eigh
from run import model, run, TIMES, OUT
x,T,r,V,a,b,psi,H=model(15,3.,8.)
e,U=eigh(H.toarray()); c=U.T@psi.ravel(); z=(U@(c[:,None]*np.exp(-1j*e[:,None]*TIMES))).T.reshape(-1,15,15)
_,q,_=run(15,3.,8.,mean=False)
p=np.abs(z)**2
sep=np.sum(p*r,axis=(1,2))
S=np.linalg.svd(z,compute_uv=False); entropy=1-np.sum(S**4,axis=1)
res={'n':15,'dimension':225,'samples':len(TIMES),'separation_error':float(np.max(abs(sep-q[:,2]))),'entropy_error':float(np.max(abs(entropy-q[:,3]))),'eigen_residual':float(np.max(abs(H@U-U*e)))}
res['pass']=max(res['separation_error'],res['entropy_error'],res['eigen_residual'])<1e-10
(OUT/'crosscheck.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res));assert res['pass']
