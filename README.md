# ActiveSpaceSolvers

<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://nmayhall-vt.github.io/ActiveSpaceSolvers.jl/stable/) -->
<!-- [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://nmayhall-vt.github.io/ActiveSpaceSolvers.jl/dev/) -->
[![Build Status](https://github.com/nmayhall-vt/ActiveSpaceSolvers.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/nmayhall-vt/ActiveSpaceSolvers.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/nmayhall-vt/ActiveSpaceSolvers.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/nmayhall-vt/ActiveSpaceSolvers.jl)


This abstract type contains all the metadata that defines a CI basis
for example, if we subtype a `FCIProblem`, then this contains the metadata
to diagonalize the Hamiltonian in the FCI basis.  
Diagonalization of H in a RASCI determinant basis would then be a different 
subtype.

Since a Problem essentially defines a Slater determinant basis, 
the combination of a `Problem`, and an `InCoreInts` can define the operator
acting on a trial state, which is a `LinearMap`. This type is then 
extended from the LinearMaps packages.

A `LinearMap` simply implements the action of our Hamiltnoian on a trial state 
defined by the `Problem`. By pairing this `LinearMap` with a `Solver` concrete
subtype, we can then generate our solution, which is a `Solution{P,T}` type. 

A `Solution{P,T}` is then essentially a set of eigenstates for problem, `P`, of
datatype `T`. This can be used for constructed RDMs and operator matrices.  

----

1. `Problem` + `InCoreInts` --> `LinearMap`
2. `LinearMap` + `Solver` --> `Solution`

----

1. `Problem`
	- `FCIProblem`
	- `RASCIProblem`
	- ...
	
1. `Solver`
	- `ArpackSolver`
	- `Davidson`
	- ...
