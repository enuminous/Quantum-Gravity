"""Additional exploratory matter-grid and finite-difference power checks."""
import json,numpy as np
from run import run,mx,OUT
r=json.loads((OUT/'results.json').read_text());conf=(23,6,3.,2.,.8,'full','excited');q,a=run(conf)
old=[np.array(r['series'][i]) for i in (14,4,15)]+[a];errors=[mx(old[i+1][:,:6]-old[i][:,:6]) for i in range(3)];base=old[1];fd=[]
for skip in (2,1):
    b=base[::skip];dt=.2*skip;fd.append(mx((b[2:,3:6]-b[:-2,3:6])/(2*dt)-b[1:-1,10:13]))
obj={'fine_case':q,'fine_series':a.tolist(),'spatial_observable_difference_7_11_15_23':errors,'finite_difference_power_error_dt_0p4_0p2':fd,'peak_field_energy_by_n':[r['rows'][i]['peak_field_energy'] for i in (14,4,15)]+[q['peak_field_energy']],'peak_AB_negativity_by_n':[r['rows'][i]['peak_AB_negativity'] for i in (14,4,15)]+[q['peak_AB_negativity']]}
(OUT/'refinement.json').write_text(json.dumps(obj,separators=(',',':'))+'\n');print({k:v for k,v in obj.items() if k!='fine_series'})
