"""QG-017-R additional seed and integrator test, not preregistered independently.
Uses the exact published run.py. Does not modify the original evidence.
"""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
import run as q
HERE=Path(__file__).parent
SEEDS=list(range(100,112))
THRESHOLDS={'full_relative_constraint':1e-8,'work':1e-8,'shear':1e-8,'cross_solver':1e-8,'ablation_relative_constraint_min':1e-3}

def diagnostics(y,pot,y0):
    c,e,s,rho,_=q.diag(y,pot);_,e0,s0,*_=q.diag(y0,pot)
    scale=3*y[1]**2+3*(y[4]**2+y[5]**2)+rho+1e-15
    return dict(constraint=q.maxabs(c/scale),work=q.maxabs(e-e0)/(1+abs(e0)),shear=q.maxabs(s-s0[:,None]))

def main():
    rows=[]
    for pot in q.POTS:
        for sign in (1,-1):
            for seed in SEEDS:
                y0=q.init(pot,sign,seed); T=(.6 if sign==1 else .17)/abs(y0[1]); t=np.linspace(0,T,101)
                y=q.solve(y0,pot,t); canon,ham=q.canonical(y0,pot,t)
                ref=solve_ivp(lambda t,y:q.rhs(t,y,pot),(0,T),y0,t_eval=t,method='Radau',rtol=q.RTOL,atol=q.ATOL)
                if not ref.success: raise RuntimeError(ref.message)
                a=diagnostics(y,pot,y0); b=diagnostics(ref.y,pot,y0)
                row=dict(pot=pot,sign=sign,seed=seed,dop853=a,radau=b,cross_solver=q.relerr(y,ref.y),canonical=q.relerr(y[:8],canon))
                for arm in ('no_matter_to_geometry','no_geometry_to_matter'):
                    z=q.solve(y0,pot,t,arm);row[arm]=diagnostics(z,pot,y0)
                rows.append(row)
    summary={'full_cases':len(rows),'ablation_cases':2*len(rows),'primary_sampled_states':len(rows)*3*101,'radau_sampled_states':len(rows)*101,'dop853_max_constraint':max(r['dop853']['constraint'] for r in rows),'radau_max_constraint':max(r['radau']['constraint'] for r in rows),'max_work':max(r[m]['work'] for r in rows for m in ('dop853','radau')),'max_shear':max(r[m]['shear'] for r in rows for m in ('dop853','radau')),'max_cross_solver_error':max(r['cross_solver'] for r in rows),'max_canonical_error':max(r['canonical'] for r in rows)}
    for arm in ('no_matter_to_geometry','no_geometry_to_matter'):
        summary[arm+'_min']=min(r[arm]['constraint'] for r in rows);summary[arm+'_max']=max(r[arm]['constraint'] for r in rows);summary[arm+'_detected']=sum(r[arm]['constraint']>THRESHOLDS['ablation_relative_constraint_min'] for r in rows)
    checks=dict(valid_constraints=max(summary['dop853_max_constraint'],summary['radau_max_constraint'])<THRESHOLDS['full_relative_constraint'],work=summary['max_work']<THRESHOLDS['work'],shear=summary['max_shear']<THRESHOLDS['shear'],cross_solver=summary['max_cross_solver_error']<THRESHOLDS['cross_solver'],canonical=summary['max_canonical_error']<THRESHOLDS['cross_solver'],all_ablations_detected=all(summary[a+'_detected']==len(rows) for a in ('no_matter_to_geometry','no_geometry_to_matter')))
    out=dict(seeds=SEEDS,thresholds=THRESHOLDS,summary=summary,checks={k:bool(v) for k,v in checks.items()},cases=rows)
    (HERE/'holdout_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ('summary','checks')},indent=2));assert all(checks.values())
if __name__=='__main__':main()
