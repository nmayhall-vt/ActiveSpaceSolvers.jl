module ActiveSpaceSolvers
using LinearMaps
using InCoreIntegrals 
using Printf


# Interface Requirements
#   These types/methods should be subtyped/extended by each submodule 
#   that implements a new ansatz
abstract type Ansatz end        
function build_H_matrix end     
#function LinearMap end


# Includes from Interface Layer
include("type_Solutions.jl");
include("type_SolverSettings.jl");


# Exports from Interface Layer
export Ansatz
export Solution 
export build_H_matrix 
export SolverSettings
export solve 


# include sub-modules and import Ansatz sub-types
include("FCI/FCI.jl");
import .FCI: FCIAnsatz

# Exports from Sub-modules 
export FCIAnsatz
export build_H_matrix 
export LinearMap 
export SolverSettings
export solve 

#import .StringCI: FCIAnsatz, build_H_matrix
#import LinearMaps: LinearMap


end
