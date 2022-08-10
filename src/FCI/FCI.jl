module FCI 
using ActiveSpaceSolvers
using LinearMaps
using InCoreIntegrals 
using Printf

# includes
include("type_FCIAnsatz.jl");
include("Helpers.jl");
include("type_DeterminantString.jl");
include("inner.jl");
include("outer.jl");

# import stuff so we can extend and export
import LinearMaps: LinearMap


# Exports
export FCIAnsatz
export build_H_matrix 
export LinearMap 

end
