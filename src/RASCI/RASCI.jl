module RASCI
using ActiveSpaceSolvers
using LinearMaps
using InCoreIntegrals 
using Printf

# includes
include("interface.jl");
include("type_RASCI_OlsenGraph.jl");
include("generate_ras_spaces.jl");
include("TDMs.jl");
include("inner.jl");
include("single_index_lookup.jl");

# import stuff so we can extend and export
import LinearMaps: LinearMap


# Exports
export RASCIAnsatz
export build_H_matrix 
export LinearMap 
export Spin_Categories

end
