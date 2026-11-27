# QG-025 — Interacting quantum gravity: gravitational Compton scattering
27 November 2026 · Monolithic LLC · exploratory EFMW reference laboratory · not peer reviewed.

**PASS_BOUNDED: explicit four-diagram tree-level Einstein–scalar reference.** This evaluates a standard scattering amplitude at order kappa^2, including the cubic graviton vertex. It is not a new physical interaction, a loop calculation, a quantum constraint-closure theorem, an EFMW microscopic derivation, or empirical validation. QG-024 tested free curvature causality; QG-025 tests a distinct on-shell interacting S-matrix observable, not a real-time interacting extension of that causality simulation.

## 1. Specification and two independent representations
Use c=hbar=1, eta=diag(+1,-1,-1,-1), kappa^2=32 pi G, and set kappa=1 for dimensionless amplitudes. Let p,k be incoming scalar and graviton, P,l outgoing scalar and graviton. p^2=P^2=m^2, k^2=l^2=0, p+k=P+l. Let a=p.k, b=p.l, c=k.l, so a-b=c and the propagator denominators are 2a,-2b,-2c. Exclude exact forward and backward endpoint degeneracies. Helicity polarization vectors e,f are complex null, transverse to k,l respectively; f denotes the conjugated outgoing polarization. Physical metric polarization tensors are products of the lowered vectors. Products of complex four-vectors are bilinear with eta, not Hermitian.

The independently coded factorized reference is

    B=(p.e)(P.f)/a - (P.e)(p.f)/b - e.f
    M_ref = (kappa^2/2) (ab/c) B^2.

This conventional Einstein–scalar amplitude is the product of scalar QED amplitudes with a kinematic factor [1,2]. For F=k⊗e_lower-e⊗k_lower and H=l⊗f_lower-f⊗l_lower, the on-shell transverse identity p.F.H.p=ab B gives a third expression kappa^2(p.F.H.p)^2/(2abc). This expression is algebraically related to factorization; it is not another independent theory or diagram derivation.

Direct implementation starts from L_m=sqrt(-g)[g^{mu nu} partial_mu phi partial_nu phi/2-m^2 phi^2/2] and g=eta+kappa h. With the curvature convention giving a positive free TT kinetic term, the Einstein–Hilbert action after a boundary term is -2/kappa^2 times sqrt(-g)g^{mn}(Gamma^a_mb Gamma^b_na-Gamma^a_mn Gamma^b_ab). The direct cubic calculation uses plane waves with all momenta incoming (k,-l,l-k); each derivative is i q_lower. It extracts the multilinear coefficient in three distinct metric perturbations by summing all six permutations. In detail let C_h[d,m,n]=(partial_m h_dn+partial_n h_dm-partial_d h_mn)/2, G1[h]=eta^{-1} C_h, G2[h_i,h_j]=-eta^{-1}h_i eta^{-1} C_hj, and Q1[h]=(tr(eta^{-1}h)/2)eta^{-1}-eta^{-1}h eta^{-1}. Sum Q1 G1 G1 + eta^{-1} G2 G1 + eta^{-1} G1 G2 under the same Gamma-Gamma contraction. This retains all terms of the cubic Einstein vertex; no factorized amplitude is used inside the direct diagram routine.

At kappa=1, scalar vertex contracted with h is V(p,P;h)=-p^T h P+tr(eta h)(p.P-m^2)/2. The s and u scalar propagator contributions are -V1 V2/(2a) and -V1 V2/(-2b), using the appropriate intermediate scalar momentum. Contact term: expand sqrt(-g)g^{-1} and sqrt(-g) to the cross term of two perturbations. With h1,h2 the external tensors, det12=tr(eta h1)tr(eta h2)/4-tr(eta h1 eta h2)/2. The densitized inverse cross coefficient is eta h1 eta h2 eta+eta h2 eta h1 eta-tr(eta h1)eta h2 eta/2-tr(eta h2)eta h1 eta/2+det12 eta. Contract it with p_lower and P_lower and subtract det12 m^2. In the t-channel, the scalar vertex tensor V^{mn}=-[p^m P^n+p^n P^m-eta^{mn}(p.P-m^2)]/2 is trace reversed with the de Donder propagator numerator. The direct contribution is minus the cubic vertex with that internal tensor divided by -2c. The convention is a stripped invariant amplitude without the overall S-matrix i. All four contributions are added before taking absolute squares. They are gauge dependent individually; the ablations refer to this particular representation, not uniquely invariant diagram assignments.

## 2. Numerical tests
A pilot at three scattering angles and two outgoing helicities agreed with the reference. The subsequent exploratory protocol was recorded before the 128-event sweep. Generate m in [.5,3], omega/m log-uniform [.03,2], theta in [.2,pi-.2], random boost speeds in [.1,.8], seed 25001. Evaluate all four helicity pairs per event (512 cases). Boost all external momenta and polarizations together. Add independent complex multiples of the external null momenta to both helicity vectors (scaled by their boosted energies); evaluate the entire diagram sum again. Independently insert arbitrary symmetric pure-gauge tensors k_lower xi_lower+xi_lower k_lower (with complex random xi and k scaled by its energy) on one leg at a time with the other physical. Reverse incoming/outgoing momenta and conjugate polarization factors. For all amplitude comparisons normalize by max(1,|M_ref|), not by an arbitrarily tiny amplitude; shell and polarization errors are absolute. No independent sampling of the four helicities is claimed.

| Check | Max residual |
|---|---:|
| Complete diagram vs factorization | 2.421112366670723e-13 |
| Field-strength form vs factorization | 1.3138140579367177e-14 |
| Lorentz boost | 2.43887348993921e-13 |
| Both vector gauge shifts | 7.594940545044341e-13 |
| Incoming pure gauge tensor | 1.0528118650674276e-12 |
| Outgoing pure gauge tensor | 1.4732024295982845e-12 |
| Reversed scattering | 1.4124357246619969e-15 |
| Rest helicity reference | 2.9917041066696015e-14 |
| Momentum/mass-shell identity | 2.042810365310288e-14 |
| Null/transverse polarization | 5.581300881723771e-16 |

## 3. Mechanism-removal controls
Remove a single contribution from both gauge-related amplitudes, and measure the difference with the same normalization as the complete result. Detection threshold 1e-3. The rest-frame radiation gauge makes the scalar pole terms zero, so testing only that gauge would not expose all missing contributions. The random gauge shifts address this blind spot.

| Removed diagram | Median residual | Max residual | Detected cases / 512 |
|---|---:|---:|---:|
| s-channel scalar | .497243 | 154.365558 | 500 |
| u-channel scalar | .173499 | 112.192862 | 492 |
| scalar contact | .673151 | 64.901766 | 511 |
| t-channel graviton | .443347 | 26.115935 | 511 |

All four ablations are rejected by the specified tests. This does not prove a uniqueness theorem for interactions; gauge-invariant higher-derivative EFT operators can alter amplitudes without violating the Ward identities.

## 4. Cross sections and infrared correspondence
In the target rest frame, omega'=omega/[1+(omega/m)(1-cos theta)]. Use circular polarizations such that B=(cos theta+h_i h_f)/2, with h_i,h_f=+/-1 labeling the vector helicities (graviton helicities +/-2). The reference amplitude at kappa=1 is m^2 B^2/[2(1-cos theta)]. The lab differential cross section is (omega'/omega)^2 times the initial-polarization-averaged and final-polarization-summed |M|^2/(64 pi^2 m^2). Thus

    (d sigma/d Omega)/(G^2 m^2) = (omega'/omega)^2 [cos^8(theta/2)+sin^8(theta/2)] / sin^4(theta/2).

This is the conventional massive spin-zero result [2]. The direct diagram evaluation reproduces it across 45 grid points (five omega/m values and nine angles); maximum relative discrepancy 7.09432512735475e-14. In the low-energy limit the recoil factor tends to one; for omega/m=1e-4 the maximum relative deviation from the no-recoil formula across the grid is .000399119427630934. In the forward direction the low-energy normalized differential cross section approaches 16/theta^4. The scaled quantity theta^4(d sigma/d Omega)/(16 G^2 m^2) is .9671497539, .9916963047, .9979183346, .9996666758, .9999166597, .9999791643 at theta=.2,.1,.05,.02,.01,.005 rad respectively. At the last point error 2.0835653673e-5. This long-range divergence is not a prediction of a finite total cross section; no IR regulator or inclusive radiation observable is attempted.

## 5. Verdict and next gate
All 17 programmed checks pass. Bounded PASS for the complete on-shell tree amplitude, direct action-based cubic vertex, gauge Ward identities, Lorentz and reversal comparisons, known cross section and rejection of the four incomplete amplitudes. No new EFMW physics is implied: the input action and the reference scattering laws are standard. The next meaningful gate is QG-026: soft-graviton emission from scalar scattering, with the leading universal soft factor and its gauge cancellation across all external legs, followed by an independent computation of the full finite-soft-energy five-point amplitude if feasible. Distinguish the soft-factor approximation from a complete five-point calculation. The microscopic EFMW-to-Einstein derivation, local nonlinear quantum constraint algebra, loops and renormalization, physical Hilbert space, UV completion, and an experimentally discriminating prediction remain unresolved.

## 6. Replay and provenance
Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0 (SciPy only used to report its version). `OPENBLAS_NUM_THREADS=1 python run.py > replay.log` regenerates results.json and summary.json. The original pilot and its log are retained. The first full sweep reached output serialization but failed on a NumPy bool; the initial_attempt.py and initial_failure.log preserve this. Final repair converts diagnostic booleans to Python bool only. pre_execution.sha256 records the pre-repair source and protocol: compare its run.py digest against initial_attempt.py, not the final source. MANIFEST.sha256 checks the final packaged originals. The raw result records all diagnostic rows and individual diagram amplitudes; initial vectors and boosts are regenerated from the specified RNG seed, not separately stored. No independent external replication; thresholds are exploratory and selected after the pilot. No repository modifications are made by the local package.

## References
1. B. R. Holstein, On-shell calculation of mixed electromagnetic and gravitational scattering, Phys. Rev. D 108, 013004 (2023), https://doi.org/10.1103/PhysRevD.108.013004 .
2. N. E. J. Bjerrum-Bohr, B. R. Holstein, L. Planté and P. Vanhove, Graviton-photon scattering, Phys. Rev. D 91, 064008 (2015), eq. (4.48) scalar cross section, https://doi.org/10.1103/PhysRevD.91.064008 .
3. QG-024 free curvature causality reference, https://github.com/enuminous/Quantum-Gravity/tree/ebe3638fed3ccad7d3f931e1dc8df6565f0c3916/QG-024 .
