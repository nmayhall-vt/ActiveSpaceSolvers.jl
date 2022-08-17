# ActiveSpaceSolvers

<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://nmayhall-vt.github.io/ActiveSpaceSolvers.jl/stable/) -->
<!-- [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://nmayhall-vt.github.io/ActiveSpaceSolvers.jl/dev/) -->
[![Build Status](https://github.com/nmayhall-vt/ActiveSpaceSolvers.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/nmayhall-vt/ActiveSpaceSolvers.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/nmayhall-vt/ActiveSpaceSolvers.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/nmayhall-vt/ActiveSpaceSolvers.jl)

## Description of workflow

### Ansatz 
An `Ansatz` abstract type contains all the metadata that defines a wavefunction approximation.
For example, if we create a `FCIAnsatz` subtype, then this contains the metadata
needed to diagonalize the Hamiltonian in the FCI basis.  
Diagonalization of H in a RASCI determinant basis would then require a different 
subtype, one that specified orbital spaces and such.
Since an `Ansatz` essentially defines a Slater determinant basis, 
the combination of an `Ansatz`, and an `InCoreInts` object can fully define 
the action of the operator (defined by the integrals) on a trial state
(defined by the Ansatz). This is simply a `LinearMap`, provided by the LinearMaps packages.

### SolverSettings
A `LinearMap` simply implements the action of our Hamiltonian on a trial state 
defined by the `Ansatz`. By pairing this `LinearMap` with a `Solver` concrete
subtype, we can then generate our solution, which is a `Solution{P,T}` type. 

### Solution
A `Solution{P,T}` is then essentially a set of eigenstates for Ansatz, `P`, of
datatype `T`. This can then be used for constructed RDMs and operator matrices, 
as we wish to do in FermiCG.

----

solve(`Ansatz` + `InCoreInts` + `SolverSettings`) --> `Solution` --> RDMs and Operators

----

1. `Ansatz`
	- `FCIAnsatz`
	- `RASCIAnsatz`
	- ...
	

## Example
```julia

# get h0, h1, h2 from pyscf or elsewhere and create ints
ints = InCoreInts(h0, h1, h2)	

# to use FCI, we simply need to define the number of orbitals and electrons
ansatz = FCIAnsatz(norb, n_elec_a, n_elec_b)

# We define some solver settings - default uses Arpack.jl
solver = SolverSettings(nroots=3, tol=1e-6, maxiter=100)

# we can now solve our Ansatz and get energies and vectors from solution
solution = eigs(ints, ansatz, solver)
 
display(solution)

e = solution.energies
v = solution.vectors

# This solution can then be used to compute the 1RDM
rdm1a, rdm1b = compute_1rdm(solution, root=2)
```
