#!/usr/bin/env python3
"""Unitary 888-level cyclic toy Hamiltonian. NOT an emergent spacetime theory.

The 888 basis labels are arbitrary graph-site labels. There is no proven
mapping from this finite cyclic model to 3+1 D gravity or the Standard Model.
"""
from __future__ import annotations
import argparse
import cmath
import json
import math

N_STATES=888

def eigenvalues(j=1.0):
    """Periodic nearest-neighbor hopping eigenvalues for 888-level ring."""
    return [2*j*math.cos(2*math.pi*k/N_STATES) for k in range(N_STATES)]


def amplitudes_in_momentum_basis(t:float,j:float=1.0):
    """Delta-localized initial state -> uniform Fourier amplitudes with phases."""
    root=math.sqrt(N_STATES)
    return [cmath.exp(-1j*e*t)/root for e in eigenvalues(j)]


def norm_squared(amplitudes):
    return sum(abs(a)**2 for a in amplitudes)


def return_probability(t:float,j:float=1.0):
    """Probability at original graph label, computed via inverse DFT at site 0."""
    phases=amplitudes_in_momentum_basis(t,j)
    a0=sum(phases)/math.sqrt(N_STATES)
    return abs(a0)**2


def report(t=2.5,j=1.0):
    amplitudes=amplitudes_in_momentum_basis(t,j)
    return {
      "label":"Base888 finite Hilbert-space illustration, NOT quantum spacetime",
      "dimension":N_STATES,
      "geometry":"one-dimensional cyclic graph defined by assumption",
      "Hamiltonian":"H=J sum_j (|j><j+1| + h.c.)",
      "coupling":j,"time":t,"norm_squared":norm_squared(amplitudes),
      "return_probability":return_probability(t,j),
      "status":"unitarity follows from Hermitian Hamiltonian; no 3+1D/SM/graviton emergence"
    }

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--time",type=float,default=2.5)
    p.add_argument("--hopping",type=float,default=1.0)
    args=p.parse_args();print(json.dumps(report(args.time,args.hopping),indent=2))

if __name__=="__main__":main()
