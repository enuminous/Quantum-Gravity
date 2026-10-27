#!/usr/bin/env python3
"""No third-party packages: python3 -m unittest -v test_qst001.py"""
import math
import unittest
from gravity_lab import Params, simulate, scalar_density_pressure, occupancy_step
from base888_toy import N_STATES, amplitudes_in_momentum_basis, norm_squared, return_probability

class QST001Tests(unittest.TestCase):
    def test_scalar_stress_pressure_double_frequency(self):
        mass=1.3; A=0.6
        for t in [0,0.3,0.8,1.3,2.1]:
            phi=A*math.cos(mass*t)
            pi=-mass*A*math.sin(mass*t)
            rho,pressure=scalar_density_pressure(phi,pi,mass)
            self.assertAlmostEqual(rho,0.5*mass**2*A**2,places=12)
            self.assertAlmostEqual(pressure,-0.5*mass**2*A**2*math.cos(2*mass*t),places=12)

    def test_minkowski_oscillator_and_energy(self):
        p=Params(mode="minkowski",dt=0.01,duration=12.0)
        rows=simulate(p)["rows"]
        E0=rows[0]["rho_scalar"]
        self.assertLess(max(abs(r["rho_scalar"]-E0) for r in rows),2e-10)
        self.assertAlmostEqual(rows[-1]["phi"],p.phi_initial*math.cos(p.mass*p.duration),delta=2e-9)

    def test_flrw_constraint_and_friction(self):
        p=Params(dt=.01,duration=40)
        result=simulate(p)
        self.assertLess(result["metadata"]["max_abs_friedmann_residual"],1e-7)
        rows=result["rows"]
        self.assertGreater(rows[-1]["scale_factor"],1.0)
        self.assertLess(rows[-1]["hubble"],rows[0]["hubble"])
        self.assertLess(rows[-1]["rho_scalar"],rows[0]["rho_scalar"])

    def test_occupancy_invariant_interval(self):
        n=0.27
        for i in range(1000):
            n=occupancy_step(n,(i%9)<3,0.1,1.5,3.1)
            self.assertGreaterEqual(n,0)
            self.assertLessEqual(n,1)

    def test_888_unitary_norm(self):
        self.assertEqual(N_STATES,888)
        for t in [0,0.3,3.0,28.0]:
            self.assertAlmostEqual(norm_squared(amplitudes_in_momentum_basis(t)),1.0,places=11)
        self.assertAlmostEqual(return_probability(0),1.0,places=11)
        self.assertGreaterEqual(return_probability(2.5),0.0)
        self.assertLessEqual(return_probability(2.5),1.0+1e-10)

    def test_linear_pressure_explicitly_distinct(self):
        p=Params(mode="minkowski",dt=0.01,duration=2,pressure_law="assumed_linear")
        row=simulate(p)["rows"][0]
        self.assertNotAlmostEqual(row["pressure_stress"],row["pressure_linear_assumption"])

if __name__=='__main__': unittest.main(verbosity=2)
