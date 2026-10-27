#!/usr/bin/env python3
"""QST-001: dimensionless homogeneous scalar + optional FLRW backreaction.

This implements only the classical scalar/FLRW truncation of the proposed
low-energy action. It does NOT quantize the metric or simulate the SM.
All quantities are rescaled dimensionless units; NOT SI or Stoney scales.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Params:
    mode: str = "flrw"  # flrw or minkowski
    mass: float = 1.0
    phi_initial: float = 0.55
    pi_initial: float = 0.0
    mplanck: float = 2.5
    vacuum_density: float = 0.03
    dt: float = 0.01
    duration: float = 60.0
    pressure_law: str = "stress" # stress or assumed_linear
    pressure_offset: float = 0.0
    pressure_coupling: float = 0.8
    threshold: float = 0.0
    gamma_on: float = 2.0
    gamma_off: float = 4.0

    def validate(self):
        if self.mode not in ("flrw", "minkowski"):
            raise ValueError("mode must be flrw or minkowski")
        if self.pressure_law not in ("stress", "assumed_linear"):
            raise ValueError("pressure_law must be stress or assumed_linear")
        if min(self.mass, self.mplanck, self.dt, self.duration) <= 0:
            raise ValueError("mass, mplanck, dt, duration must be positive")
        if self.vacuum_density < 0 or min(self.gamma_on, self.gamma_off) < 0:
            raise ValueError("vacuum_density and rates must be nonnegative")
        if self.duration / self.dt > 2_000_000:
            raise ValueError("too many time steps")


def scalar_density_pressure(phi: float, pi: float, mass: float):
    kinetic = 0.5 * pi*pi
    potential = 0.5 * mass*mass*phi*phi
    return kinetic+potential, kinetic-potential


def rhs(y, p: Params):
    phi, pi, hubble, log_a = y
    if p.mode == "minkowski":
        return (pi, -p.mass*p.mass*phi, 0.0, 0.0)
    return (
        pi,
        -3*hubble*pi-p.mass*p.mass*phi,
        -pi*pi/(2*p.mplanck*p.mplanck),
        hubble,
    )


def rk4(y, dt, p):
    k1 = rhs(y, p)
    k2 = rhs(tuple(a+dt*b/2 for a,b in zip(y,k1)),p)
    k3 = rhs(tuple(a+dt*b/2 for a,b in zip(y,k2)),p)
    k4 = rhs(tuple(a+dt*b for a,b in zip(y,k3)),p)
    return tuple(a+dt*(b+2*c+2*d+e)/6 for a,b,c,d,e in zip(y,k1,k2,k3,k4))


def occupancy_step(n, cavitation_allowed, dt, gamma_on, gamma_off):
    if cavitation_allowed:
        return 1-(1-n)*math.exp(-gamma_on*dt)
    return n*math.exp(-gamma_off*dt)


def simulate(p: Params):
    p.validate()
    rho0, _ = scalar_density_pressure(p.phi_initial, p.pi_initial,p.mass)
    H0 = math.sqrt((rho0+p.vacuum_density)/(3*p.mplanck*p.mplanck)) if p.mode=="flrw" else 0.0
    y = (p.phi_initial, p.pi_initial, H0, 0.0)
    n = 0.0
    steps=math.ceil(p.duration/p.dt)
    data=[]
    for k in range(steps+1):
        t = min(k*p.dt, p.duration)
        phi, pi, H, log_a = y
        rho, p_scalar = scalar_density_pressure(phi,pi,p.mass)
        p_linear = p.pressure_offset+p.pressure_coupling*phi
        p_gate = p_scalar if p.pressure_law=="stress" else p_linear
        c = int(p_gate < p.threshold)
        friedmann_residual = H*H-(rho+p.vacuum_density)/(3*p.mplanck*p.mplanck) if p.mode=="flrw" else None
        data.append({"t":t,"phi":phi,"pi":pi,"rho_scalar":rho,"pressure_stress":p_scalar,
          "pressure_linear_assumption":p_linear,"cavitation_permission":c,"occupancy":n,
          "hubble":H,"scale_factor":math.exp(log_a),"friedmann_residual":friedmann_residual})
        if k < steps:
            dt=min(p.dt,p.duration-k*p.dt)
            if dt<=0: break
            y = rk4(y,dt,p)
            n=occupancy_step(n,c,dt,p.gamma_on,p.gamma_off)
    errors=[abs(x["friedmann_residual"]) for x in data if x["friedmann_residual"] is not None]
    max_err=max(errors) if errors else None
    return {"metadata":{
      "model":"QST-001 classical homogeneous truncation",
      "physical_status":"illustrative, not measured, no quantum spacetime",
      "pressure_warning":"stress-energy pressure is not experimentally established substrate pressure",
      "units":"dimensionless rescaled toy units; no SI/Stoney calibration",
      "params":vars(p),"max_abs_friedmann_residual":max_err}, "rows":data}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode",choices=["flrw","minkowski"],default="flrw")
    ap.add_argument("--mass",type=float,default=1.0)
    ap.add_argument("--amplitude",type=float,default=0.55)
    ap.add_argument("--mplanck",type=float,default=2.5)
    ap.add_argument("--vacuum-density",type=float,default=0.03)
    ap.add_argument("--dt",type=float,default=0.01)
    ap.add_argument("--duration",type=float,default=60)
    ap.add_argument("--pressure-law",choices=["stress","assumed_linear"],default="stress")
    ap.add_argument("--threshold",type=float,default=0)
    ap.add_argument("--out",type=Path,default=Path("qst001-results.json"))
    ap.add_argument("--format",choices=["json","csv"],default="json")
    args=ap.parse_args()
    p=Params(mode=args.mode,mass=args.mass,phi_initial=args.amplitude,mplanck=args.mplanck,
             vacuum_density=args.vacuum_density,dt=args.dt,duration=args.duration,
             pressure_law=args.pressure_law,threshold=args.threshold)
    result=simulate(p)
    if args.format=="json":
        args.out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    else:
        with args.out.open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(result["rows"][0]))
            writer.writeheader();writer.writerows(result["rows"])
    print(json.dumps({"file":str(args.out),"samples":len(result["rows"]),
        "max_abs_friedmann_residual":result["metadata"]["max_abs_friedmann_residual"],
        "final_scale_factor":result["rows"][-1]["scale_factor"]},indent=2))

if __name__ == "__main__":
    main()
