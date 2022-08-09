module ActiveSpaceSolvers
using LinearMaps
using InCoreIntegrals 

abstract type Problem end

abstract type Solver end

# includes
include("StringCI/StringCI.jl");

# import stuff so we can extend and export
import .StringCI.FCIProblem
import .StringCI.build_H_matrix
import LinearMaps: LinearMap


# Exports
export FCIProblem
export build_H_matrix 
export LinearMap 

end
