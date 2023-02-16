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
function RASCIAnsatz(no::Int, na::Int, nb::Int, fock::Any, ras1_min=0, ras3_max=fock[3])
    na <= no || throw(DimensionMismatch)
    nb <= no || throw(DimensionMismatch)
    ras1_min <= fock[1] || throw(DimensionMismatch)
    ras3_max <= fock[3] || throw(DimensionMismatch)
    sum(fock) == no || throw(DimensionMismatch)
    fock = convert(SVector{3,Int},collect(fock))
    if fock[1] == 0 && fock[3] == 0
        #FCI problem
        dima = calc_ndets(no, na)
        dimb = calc_ndets(no, nb)
        xalpha = ras_calc_ndets(no, na, fock, ras1_min, ras3_max)[2]
        xbeta = ras_calc_ndets(no, nb, fock, ras1_min, ras3_max)[2]
    else
        dima, xalpha = ras_calc_ndets(no, na, fock, ras1_min, ras3_max)
        dimb, xbeta = ras_calc_ndets(no, nb, fock, ras1_min, ras3_max)
    end
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
    a_configs = ActiveSpaceSolvers.RASCI.compute_configs(prb)[1]
    b_configs = ActiveSpaceSolvers.RASCI.compute_configs(prb)[2]
    
    #fill single excitation lookup tables
    a_lookup = ActiveSpaceSolvers.RASCI.fill_lookup(prb, a_configs, prb.dima)
    b_lookup = ActiveSpaceSolvers.RASCI.fill_lookup(prb, b_configs, prb.dimb)
    #iters = 0
    function mymatvec(v)
        #iters += 1
        #@printf(" Iter: %4i\n", iters)
        #@printf(" %-50s", "Compute sigma 1: ")
        #flush(stdout)
        
        nr = 0
        if length(size(v)) == 1
            nr = 1
            v = reshape(v,prb.dim, nr)
        else 
            nr = size(v)[2]
        end
        v = reshape(v, prb.dima, prb.dimb, nr)
        
        sigma1 = ActiveSpaceSolvers.RASCI.compute_sigma_one(b_configs, b_lookup, v, ints, prb)
        sigma2 = ActiveSpaceSolvers.RASCI.compute_sigma_two(a_configs, a_lookup, v, ints, prb)
        sigma3 = ActiveSpaceSolvers.RASCI.compute_sigma_three(a_configs, b_configs, a_lookup, b_lookup, v, ints, prb)
        
        sig = sigma1 + sigma2 + sigma3
        
        v = reshape(v,prb.dim, nr)
        sig = reshape(sig, prb.dim, nr)
        sig .+= ints.h0*v
        return sig
    end
    return LinearMap(mymatvec, prb.dim, prb.dim, issymmetric=true, ismutating=false, ishermitian=true)
end

function ras_calc_ndets(no, nelec, fock, ras1_min, ras3_max)
    x = ActiveSpaceSolvers.RASCI.make_ras_x(no, nelec, fock, ras1_min, ras3_max)
    dim_x = findmax(x)[1]
    #dim_x = no 
    return dim_x, x
end

function calc_ndets(no,nelec)
    if no > 20
        x = factorial(big(no))
        y = factorial(nelec)
        z = factorial(big(no-nelec))
        return Int64(x÷(y*z))
    end

    return factorial(no)÷(factorial(nelec)*factorial(no-nelec))
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
    return ActiveSpacesolvers.RASCI.compute_operator_a_a(bra::Solution{RASCIAnsatz},                                                       ket::Solution{RASCIAnsatz})
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
    return ActiveSpaceSolvers.RASCI.compute_operator_a_b(bra::Solution{RASCIAnsatz}, 
                                                 ket::Solution{RASCIAnsatz})
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
    return ActiveSpaceSolvers.RASCI.compute_operator_ca_aa(bra::Solution{RASCIAnsatz}, 
                                                 ket::Solution{RASCIAnsatz})
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
    return ActiveSpaceSolvers.RASCI.compute_operator_ca_bb(bra::Solution{RASCIAnsatz}, 
                                                 ket::Solution{RASCIAnsatz})
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
    return ActiveSpaceSolvers.RASCI.compute_operator_ca_ab(bra::Solution{RASCIAnsatz}, 
                                                           ket::Solution{RASCIAnsatz})
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
    return ActiveSpaceSolvers.RASCI.compute_operator_cc_bb(bra::Solution{RASCIAnsatz}, 
                                                 ket::Solution{RASCIAnsatz})
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
    return ActiveSpaceSolvers.RASCI.compute_operator_cc_aa(bra::Solution{RASCIAnsatz}, 
                                                 ket::Solution{RASCIAnsatz})
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
    return ActiveSpaceSolvers.RASCI.compute_operator_cc_ab(bra::Solution{RASCIAnsatz}, 
                                                 ket::Solution{RASCIAnsatz})
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
    return ActiveSpaceSolvers.RASCI.compute_operator_cca_aaa(bra::Solution{RASCIAnsatz}, 
                                                 ket::Solution{RASCIAnsatz})
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
    return ActiveSpaceSolvers.RASCI.compute_operator_cca_bbb(bra::Solution{RASCIAnsatz}, 
                                                 ket::Solution{RASCIAnsatz})
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
    return ActiveSpaceSolvers.RASCI.compute_operator_cca_aba(bra::Solution{RASCIAnsatz}, 
                                                             ket::Solution{RASCIAnsatz})
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
    return ActiveSpaceSolvers.RASCI.compute_operator_cca_abb(bra::Solution{RASCIAnsatz}, 
                                                 ket::Solution{RASCIAnsatz}) 
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
