from functools import reduce
import numpy as np
import scipy

import pyscf
from pyscf import fci
from pyscf import gto, scf, ao2mo, lo, tdscf, cc


def tda_denisty_matrix(td, state_id):
    '''
    Taking the TDA amplitudes as the CIS coefficients, calculate the density
    matrix (in AO basis) of the excited states
    '''
    cis_t1 = td.xy[state_id][0]
    dm_oo =-np.einsum('ia,ka->ik', cis_t1.conj(), cis_t1)
    dm_vv = np.einsum('ia,ic->ac', cis_t1, cis_t1.conj())

    # The ground state density matrix in mo_basis
    mf = td._scf
    dm = np.diag(mf.mo_occ)

    # Add CIS contribution
    nocc = cis_t1.shape[0]
    # Note that dm_oo and dm_vv correspond to spin-up contribution. "*2" to
    # include the spin-down contribution
    dm[:nocc,:nocc] += dm_oo * 2
    dm[nocc:,nocc:] += dm_vv * 2

    # Transform density matrix to AO basis
    mo = mf.mo_coeff
    dm = np.einsum('pi,ij,qj->pq', mo, dm, mo.conj())
    return dm

mol = gto.Mole()
mol.atom = '''
H       -3.4261000000     -2.2404000000      5.4884000000                 
H       -5.6274000000     -1.0770000000      5.2147000000                 
C       -3.6535000000     -1.7327000000      4.5516000000                 
H       -1.7671000000     -2.2370000000      3.6639000000                 
C       -4.9073000000     -1.0688000000      4.3947000000                 
H       -6.1631000000      0.0964000000      3.1014000000                 
C       -2.7258000000     -1.7321000000      3.5406000000                 
H       -0.3003000000      1.0832000000     -5.2357000000                 
C       -5.2098000000     -0.4190000000      3.2249000000                 
C       -2.9961000000     -1.0636000000      2.3073000000                 
H       -1.1030000000     -1.5329000000      1.3977000000                 
H       -0.4270000000     -0.8029000000     -0.8566000000                 
H        0.2361000000     -0.0979000000     -3.1273000000                 
C       -1.0193000000      1.0730000000     -4.4150000000                 
H       -2.4988000000      2.2519000000     -5.5034000000                 
C       -4.2740000000     -0.3924000000      2.1445000000                 
H       -5.5015000000      0.7944000000      0.8310000000                 
C       -2.0613000000     -1.0272000000      1.2718000000                 
C       -1.3820000000     -0.2895000000     -0.9772000000                 
C       -0.7171000000      0.4180000000     -3.2476000000                 
C       -2.2720000000      1.7395000000     -4.5690000000                 
H       -4.1576000000      2.2412000000     -3.6787000000                 
C       -4.5463000000      0.2817000000      0.9534000000                 
C       -2.3243000000     -0.3402000000      0.0704000000                 
C       -1.6528000000      0.3874000000     -2.1670000000                 
C       -3.1998000000      1.7341000000     -3.5584000000                 
C       -3.6044000000      0.3309000000     -0.0943000000                 
C       -2.9302000000      1.0591000000     -2.3292000000                 
C       -3.8665000000      1.0187000000     -1.2955000000                 
H       -4.8243000000      1.5256000000     -1.4217000000                 
'''

mol.basis = '6-31g*'
mol.spin = 0
mol.build()

#mf = scf.ROHF(mol).x2c()
mf = scf.RHF(mol)
mf.verbose = 4
mf.get_init_guess(mol, key='minao')
mf.conv_tol = 1e-9
#mf.level_shift = .1
#mf.diis_start_cycle = 4
#mf.diis_space = 10
mf.run(max_cycle=200)


n_triplets = 4
n_singlets = 4

avg_rdm1 = mf.make_rdm1()


# compute singlets
mytd = tdscf.TDA(mf)
mytd.singlet = True 
mytd = mytd.run(nstates=n_singlets)
mytd.analyze()
for i in range(mytd.nroots):
    avg_rdm1 += tda_denisty_matrix(mytd, i)

# compute triplets 
mytd = tdscf.TDA(mf)
mytd.singlet = False 
mytd = mytd.run(nstates=n_triplets)
mytd.analyze()
for i in range(mytd.nroots):
    avg_rdm1 += tda_denisty_matrix(mytd, i)

# normalize
avg_rdm1 = avg_rdm1 / (n_singlets + n_triplets + 1)


S = mf.get_ovlp()
np.save("overlap_mat_single", S)
np.save("density_mat_single", mf.make_rdm1())
np.save("rhf_mo_coeffs_single", mf.mo_coeff)
np.save("cis_sa_density_mat_single", avg_rdm1)


