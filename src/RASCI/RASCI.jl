module RASCI
using ActiveSpaceSolvers
using LinearMaps
using InCoreIntegrals 
using Printf

# includes
include("interface.jl");
include("type_RASCI_OlsenGraph.jl");
include("TDMs.jl");
include("inner.jl");

# import stuff so we can extend and export
import LinearMaps: LinearMap


# Exports
export RASCIAnsatz
export build_H_matrix 
export LinearMap 

end
