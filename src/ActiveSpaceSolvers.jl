module ActiveSpaceSolvers
using LinearMaps
using InCoreIntegrals 
using Printf


# Interface Types
abstract type Ansatz end        
include("type_Solutions.jl");
include("type_SolverSettings.jl");


# Interface Methods: extend each for a new `Ansatz`
function build_H_matrix end     
function solve end     
function compute_1rdm end     
function compute_2rdm end    
# operator functions
function compute_operator_ca_aa end     
function compute_operator_ca_bb end     
function compute_operator_ca_ab end     
# methods for getting info from Ansatze


# Exports from Interface Layer
# types
export Ansatz
export Solution 
export SolverSettings
# methods 
export LinearMap 
export build_H_matrix 
export solve 
export compute_1rdm
export compute_2rdm
export compute_operator_ca_aa
export compute_operator_ca_bb
export compute_operator_ca_ab

export n_orbs
export n_elec
export n_elec_a
export n_elec_b
export dim

# include sub-modules and import/export Ansatz sub-types
include("FCI/FCI.jl");
import .FCI: FCIAnsatz
export FCIAnsatz


# some methods
n_orbs(a::Ansatz) = a.no 
n_elec(a::Ansatz) = a.na + a.nb 
n_elec_a(a::Ansatz) = a.na     
n_elec_b(a::Ansatz) = a.nb     
dim(a::Ansatz) = a.dim 

n_orbs(a::Solution) = n_orbs(a.ansatz)
n_elec(a::Solution) = n_elec(a.ansatz)
n_elec_a(a::Solution) = n_elec_a(a.ansatz)
n_elec_b(a::Solution) = n_elec_b(a.ansatz)
dim(a::Solution) = dim(a.ansatz)

end
