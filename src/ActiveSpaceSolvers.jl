module ActiveSpaceSolvers
using LinearMaps
using InCoreIntegrals 

"""
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
subtype, we can then generate our solution, which is a `CIStates{P,T}` type. 

A `Solution{P,T}` is then essentially a set of eigenstates for problem, `P`, of
datatype `T`. This can be used for constructed RDMs and operator matrices.  
"""
abstract type Problem end



"""
Not yet sure about this one...
"""
abstract type Solver end

# includes
include("type_Solutions.jl");
include("type_ArpackSolver.jl");
include("StringCI/StringCI.jl");

# import stuff so we can extend and export
import .StringCI.FCIProblem
import .StringCI.build_H_matrix
import LinearMaps: LinearMap


# Exports
export Problem
export Solver 
export Solution 
export FCIProblem
export build_H_matrix 
export LinearMap 
export ArpackSolver 
export solve 

end
