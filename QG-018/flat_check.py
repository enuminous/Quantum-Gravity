import numpy as np
from pathlib import Path
s=np.load(Path(__file__).with_name('snapshots.npz')); r=s['r800']
for d in [1,-1]:
    t=4; z=(r+d*t-6)/2; mask=np.abs(z)<1; b=np.zeros_like(r); bp=b.copy()
    b[mask]=.08*np.exp(1-1/(1-z[mask]**2));bp[mask]=-b[mask]*z[mask]/(1-z[mask]**2)**2
    F=(r+d*t)*b; Fp=b+(r+d*t)*bp
    p=np.divide(Fp,r,out=np.zeros_like(r),where=r>0)-np.divide(F,r*r,out=np.zeros_like(r),where=r>0)
    q=d*np.divide(Fp,r,out=np.zeros_like(r),where=r>0)
    target=np.array([p,q]);obs=s[f'a0.08_d{d}_n800_flat'][-1,:2]
    print(d,'relative L2',np.linalg.norm(obs-target)/np.linalg.norm(target),'max absolute',np.max(abs(obs-target)))
