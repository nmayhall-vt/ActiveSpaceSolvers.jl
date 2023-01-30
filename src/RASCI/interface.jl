#using InCoreIntegrals
using ActiveSpaceSolvers
using QCBase
import LinearMaps
using OrderedCollections
using BlockDavidson
using StaticArrays
using LinearAlgebra
using Printf
using TimerOutputs
using StatProfilerHTML

"""
Type containing all the metadata needed to define a RASCI problem 

    no::Int  # number of orbitals
    na::Int  # number of alpha
    nb::Int  # number of beta
    dima::Int 
    dimb::Int 
    dim::Int
    converged::Bool
    restarted::Bool
    iteration::Int
    algorithm::String   #  options: direct/davidson
    n_roots::Int
    fock::SVector{3, Int}   #fock sector orbitals (RAS1, RAS2, RAS3)
    ras1_min::Int       # min electrons in RAS1
    ras3_max::Int       # max electrons in RAS3
    xalpha::Array{Int}
    xbeta::Array{Int}
"""
struct RASCIAnsatz <: Ansatz
    no::Int  # number of orbitals
    na::Int  # number of alpha
    nb::Int  # number of beta
    dima::Int 
    dimb::Int 
    dim::Int
    fock::SVector{3, Int}   # Number of orbitals in each ras space (RAS1, RAS2, RAS3)
    ras1_min::Int       # Minimum number of electrons in RAS1
    ras3_max::Int       # Max number of electrons in RAS3
    xalpha::Array{Int}
    xbeta::Array{Int}
    converged::Bool
    restarted::Bool
    iteration::Int
    algorithm::String   #  options: direct/davidson
    n_roots::Int
end

"""
    RASCIAnsatz(no, na, nb, fock::Any, ras1_min=1, ras3_max=2)
Constructor
# Arguments
- `no`: Number of spatial orbitals
- `na`: Number of α electrons
- `nb`: Number of β electrons
- `fock`: Number of orbitals in each (RAS1, RAS2, RAS3)
- `ras1_min`: Minimum number of electrons in RAS1
- `ras3_max`: Max number of electrons in RAS3
"""
function RASCIAnsatz(no::Int, na::Int, nb::Int, fock::Any, ras1_min=1, ras3_max=2)
    na <= no || throw(DimensionMismatch)
    nb <= no || throw(DimensionMismatch)
    ras3_max <= fock[3] || throw(DimensionMismatch)
    fock = convert(SVector{3,Int},collect(fock))
    println(fock)
    #adding extra line
    dima, xalpha = ras_calc_ndets(no, na, fock, ras1_min, ras3_max)
    dimb, xbeta = ras_calc_ndets(no, nb, fock, ras1_min, ras3_max)
    return RASCIAnsatz(no, na, nb, dima, dimb, dima*dimb, fock, ras1_min, ras3_max, xalpha, xbeta, false, false, 1, "direct", 1)
end

function Base.display(p::RASCIAnsatz)
    @printf(" RASCIAnsatz:: #Orbs = %-3i #α = %-2i #β = %-2i Fock Spaces: (%i, %i, %i) Dimension: %-3i RAS1 min: %i RAS3 max: %i\n",p.no,p.na,p.nb,p.fock[1], p.fock[2], p.fock[3], p.dim,  p.ras1_min, p.ras3_max)
end

function Base.print(p::RASCIAnsatz)
    @printf(" RASCIAnsatz:: #Orbs = %-3i #α = %-2i #β = %-2i Fock Spaces: (%i, %i, %i) Dimension: %-3i RAS1 min: %i RAS3 max: %i\n",p.no,p.na,p.nb,p.fock[1], p.fock[2], p.fock[3], p.dim,  p.ras1_min, p.ras3_max)
end

"""
    LinearMap(ints, prb::RASCIAnsatz)

Get LinearMap with takes a vector and returns action of H on that vector

# Arguments
- ints: `InCoreInts` object
- prb:  `RASCIAnsatz` object
"""
function LinearMaps.LinearMap(ints::InCoreInts, prb::RASCIAnsatz)
    @printf(" %-50s", "Compute alpha configs: ")
    flush(stdout)
    @time a_configs = ActiveSpaceSolvers.RASCI.compute_configs(prb)[1]
    @printf(" %-50s", "Compute beta configs: ")
    flush(stdout)
    @time b_configs = ActiveSpaceSolvers.RASCI.compute_configs(prb)[2]
    
    #fill single excitation lookup tables
    @printf(" %-50s", "Fill single excitation lookup for alpha: ")
    flush(stdout)
    @time a_lookup = ActiveSpaceSolvers.RASCI.fill_lookup(prb, a_configs, prb.dima)
    @printf(" %-50s", "Fill single excitation lookup for beta: ")
    flush(stdout)
    @time b_lookup = ActiveSpaceSolvers.RASCI.fill_lookup(prb, b_configs, prb.dimb)
    
    function mymatvec(v)
        @printf(" %-50s", "Compute sigma 1: ")
        flush(stdout)
        @time sigma1 = vec(ActiveSpaceSolvers.RASCI.compute_sigma_one(b_configs, b_lookup, v, ints, prb))
        @printf(" %-50s", "Compute sigma 2: ")
        flush(stdout)
        @time sigma2 = vec(ActiveSpaceSolvers.RASCI.compute_sigma_two(a_configs, a_lookup, v, ints, prb))
        #@profilehtml sigma2 = vec(ActiveSpaceSolvers.RASCI.compute_sigma_two(a_configs, a_lookup, v, ints, prb))
        @printf(" %-50s", "Compute sigma 3: ")
        flush(stdout)
        @time sigma3 = vec(ActiveSpaceSolvers.RASCI.compute_sigma_three(a_configs, b_configs, a_lookup, b_lookup, v, ints, prb))
        @profilehtml sigma3 = vec(ActiveSpaceSolvers.RASCI.compute_sigma_three(a_configs, b_configs, a_lookup, b_lookup, v, ints, prb))
        error("here")
        sigma = sigma1 + sigma2 + sigma3
        return sigma
    end
    return LinearMap(mymatvec, prb.dim, prb.dim, issymmetric=true, ismutating=false, ishermitian=true)
end

function ras_calc_ndets(no, nelec, fock, ras1_min, ras3_max)
    x = ActiveSpaceSolvers.RASCI.make_ras_x(no, nelec, fock, ras1_min, ras3_max)
    dim_x = findmax(x)[1]
    #dim_x = no 
    return dim_x, x
end

"""
"""
function ActiveSpaceSolvers.apply_sminus(v::Matrix, ansatz::RASCIAnsatz)
end

"""
"""
function ActiveSpaceSolvers.apply_splus(v::Matrix, ansatz::RASCIAnsatz)
end

"""
    build_S2_matrix(P::RASCIAnsatz)

Build the S2 matrix in the Slater Determinant Basis  specified by `P`
"""
function ActiveSpaceSolvers.apply_S2_matrix(P::RASCIAnsatz, v::AbstractArray{T}) where T
end



"""
    build_H_matrix(ints, P::RASCIAnsatz)

Build the Hamiltonian defined by `ints` in the Slater Determinant Basis  specified by `P`
"""
function ActiveSpaceSolvers.build_H_matrix(ints::InCoreInts, p::RASCIAnsatz)
    return ActiveSpaceSolvers.RASCI.build_H_matrix(ints::InCoreInts, p::RASCIAnsatz)
end

"""
    compute_operator_a_a(bra::Solution{RASCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a operator between states `bra_v` and `ket_v` for alpha
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_a_a(bra::Solution{RASCIAnsatz,T}, 
                                                 ket::Solution{RASCIAnsatz,T}) where {T}
    return ActiveSpacesolvers.RASCI.compute_operator_a_a(bra::Solution{RASCIAnsatz,T},                                                       ket::Solution{RASCIAnsatz,T}) where {T}
end

"""
    compute_operator_a_b(bra::Solution{RASCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a operator between states `bra_v` and `ket_v` for beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_a_b(bra::Solution{RASCIAnsatz,T}, 
                                                 ket::Solution{RASCIAnsatz,T}) where {T}
    return ActiveSpaceSolvers.RASCI.compute_operator_a_b(bra::Solution{RASCIAnsatz,T}, 
                                                 ket::Solution{RASCIAnsatz,T}) where {T}
end

"""
    compute_operator_ca_aa(bra::Solution{RASCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a'a operators between states `bra_v` and `ket_v` for alpha-alpha
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_ca_aa(bra::Solution{RASCIAnsatz,T}, 
                                                   ket::Solution{RASCIAnsatz,T}) where {T}
    return ActiveSpaceSolvers.RASCI.compute_operator_ca_aa(bra::Solution{RASCIAnsatz,T}, 
                                                 ket::Solution{RASCIAnsatz,T}) where {T}
end

"""
    compute_operator_ca_bb(bra::Solution{RASCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a'a operators between states `bra_v` and `ket_v` for beta-beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_ca_bb(bra::Solution{RASCIAnsatz,T}, 
                                                   ket::Solution{RASCIAnsatz,T}) where {T}
    return ActiveSpaceSolvers.RASCI.compute_operator_ca_bb(bra::Solution{RASCIAnsatz,T}, 
                                                 ket::Solution{RASCIAnsatz,T}) where {T}
end


"""
    compute_operator_ca_ab(bra::Solution{RASCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a'a operators between states `bra_v` and `ket_v` for alpha-beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_ca_ab(bra::Solution{RASCIAnsatz,T}, 
                                                   ket::Solution{RASCIAnsatz,T}) where {T}
    return ActiveSpaceSolvers.RASCI.compute_operator_ca_ab(bra::Solution{RASCIAnsatz,T}, 
                                                 ket::Solution{RASCIAnsatz,T}) where {T}
end


"""
    compute_operator_cc_aa(bra::Solution{RASCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a'a' operators between states `bra_v` and `ket_v` for beta-beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_cc_bb(bra::Solution{RASCIAnsatz,T}, 
                                                   ket::Solution{RASCIAnsatz,T}) where {T}
    return ActiveSpaceSolvers.RASCI.compute_operator_cc_bb(bra::Solution{RASCIAnsatz,T}, 
                                                 ket::Solution{RASCIAnsatz,T}) where {T}
end


"""
    compute_operator_cc_aa(bra::Solution{RASCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a'a' operators between states `bra_v` and `ket_v` for alpha-alpha 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_cc_aa(bra::Solution{RASCIAnsatz,T}, 
                                                   ket::Solution{RASCIAnsatz,T}) where {T}
    return ActiveSpaceSolvers.RASCI.compute_operator_cc_aa(bra::Solution{RASCIAnsatz,T}, 
                                                 ket::Solution{RASCIAnsatz,T}) where {T}
end


"""
    compute_operator_cc_ab(bra::Solution{RASCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a'a' operators between states `bra_v` and `ket_v` for alpha-beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_cc_ab(bra::Solution{RASCIAnsatz,T}, 
                                                   ket::Solution{RASCIAnsatz,T}) where {T}
    return ActiveSpaceSolvers.RASCI.compute_operator_cc_ab(bra::Solution{RASCIAnsatz,T}, 
                                                 ket::Solution{RASCIAnsatz,T}) where {T}
end


"""
    compute_operator_cca_aaa(bra::Solution{RASCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a'a'a operators between states `bra_v` and `ket_v` for alpha-alpha-alpha 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_cca_aaa(bra::Solution{RASCIAnsatz,T}, 
                                                   ket::Solution{RASCIAnsatz,T}) where {T}
    return ActiveSpaceSolvers.RASCI.compute_operator_cca_aaa(bra::Solution{RASCIAnsatz,T}, 
                                                 ket::Solution{RASCIAnsatz,T}) where {T}
end


"""
    compute_operator_cca_bbb(bra::Solution{RASCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a'a'a operators between states `bra_v` and `ket_v` for beta-beta-beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_cca_bbb(bra::Solution{RASCIAnsatz,T}, 
                                                   ket::Solution{RASCIAnsatz,T}) where {T}
    return ActiveSpaceSolvers.RASCI.compute_operator_cca_bbb(bra::Solution{RASCIAnsatz,T}, 
                                                 ket::Solution{RASCIAnsatz,T}) where {T}
end


"""
    compute_operator_cca_aba(bra::Solution{RASCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a'a'a operators between states `bra_v` and `ket_v` for alpha-beta-alpha 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_cca_aba(bra::Solution{RASCIAnsatz,T}, 
                                                     ket::Solution{RASCIAnsatz,T}) where {T}
    return ActiveSpaceSolvers.RASCI.compute_operator_cca_aba(bra::Solution{RASCIAnsatz,T}, 
                                                 ket::Solution{RASCIAnsatz,T}) where {T}
end


"""
    compute_operator_cca_abb(bra::Solution{RASCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a'a'a operators between states `bra_v` and `ket_v` for alpha-beta-beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_cca_abb(bra::Solution{RASCIAnsatz,T}, 
                                                     ket::Solution{RASCIAnsatz,T}) where {T}
    return ActiveSpaceSolvers.RASCI.compute_operator_cca_abb(bra::Solution{RASCIAnsatz,T}, 
                                                 ket::Solution{RASCIAnsatz,T}) where {T}
end


"""
    compute_1rdm(sol::Solution{RASCIAnsatz,T}; root=1) where {T}
"""
function ActiveSpaceSolvers.compute_1rdm(sol::Solution{RASCIAnsatz,T}; root=1) where {T}
    return ActiveSpaceSolvers.RASCI.compute_1rdm(sol.ansatz, sol.vectors[:,root])
end

"""
    compute_1rdm_2rdm(sol::Solution{A,T}; root=1) where {A,T}
"""
function ActiveSpaceSolvers.compute_1rdm_2rdm(sol::Solution{RASCIAnsatz,T}; root=1) where {T}
    return ActiveSpaceSolvers.RASCI.compute_1rdm_2rdm(sol.ansatz, sol.vectors[:,root])
end
