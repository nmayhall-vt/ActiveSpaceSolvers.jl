module ActiveSpaceSolvers
using LinearMaps
using InCoreIntegrals 
using Printf

# includes
include("type_Ansatz.jl");
include("type_FCIAnsatz.jl");
include("type_Solutions.jl");
include("type_SolverSettings.jl");

# import stuff so we can extend and export
#import .StringCI: FCIAnsatz, LinOp, build_H_matrix
#import .StringCI: FCIAnsatz, build_H_matrix
import LinearMaps: LinearMap


# Exports
export Ansatz
export FCIAnsatz

export Solution 
export build_H_matrix 
export LinearMap 
export SolverSettings
export solve 

end
