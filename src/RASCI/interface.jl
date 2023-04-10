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
    ras_spaces::SVector{3, Int}   #fock sector orbitals (RAS1, RAS2, RAS3)
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
    ras_spaces::SVector{3, Int}   # Number of orbitals in each ras space (RAS1, RAS2, RAS3)
    ras1_min::Int       # Minimum number of electrons in RAS1 (SPIN SPECIFIC)
    ras3_max::Int       # Max number of electrons in RAS3 (SPIN SPECIFIC)
    max_h::Int  #max number of holes in ras1 (GLOBAL, Slater Det)
    max_p::Int #max number of particles in ras3 (GLOBAL, Slater Det)
    xalpha::Array{Int}
    xbeta::Array{Int}
    converged::Bool
    restarted::Bool
    iteration::Int
    algorithm::String   #  options: direct/davidson
    n_roots::Int
end

"""
    RASCIAnsatz(no, na, nb, ras_spaces::Any, ras1_min=1, ras3_max=2)
Constructor
# Arguments
- `no`: Number of spatial orbitals
- `na`: Number of α electrons
- `nb`: Number of β electrons
- `ras_spaces`: Number of orbitals in each (RAS1, RAS2, RAS3)
- `ras1_min`: Minimum number of electrons in RAS1
- `ras3_max`: Max number of electrons in RAS3
"""
function RASCIAnsatz(no::Int, na, nb, ras_spaces::Any; ras1_min=0, ras3_max=ras_spaces[3], max_h=0, max_p=ras_spaces[3])
    na <= no || throw(DimensionMismatch)
    nb <= no || throw(DimensionMismatch)
    #ras1_min <= ras_spaces[1] || throw(DimensionMismatch)
    #ras3_max <= ras_spaces[3] || throw(DimensionMismatch)
    sum(ras_spaces) == no || throw(DimensionMismatch)
    ras_spaces = convert(SVector{3,Int},collect(ras_spaces))
    na = convert(Int, na)
    nb = convert(Int, nb)
    #if ras_spaces[1] == 0 && ras_spaces[3] == 0
        #FCI problem
        #dima = calc_ndets(no, na)
        #dimb = calc_ndets(no, nb)
    #    dima, xalpha = ras_calc_ndets(no, na, ras_spaces, ras1_min, ras3_max)
    #    dimb, xbeta = ras_calc_ndets(no, nb, ras_spaces, ras1_min, ras3_max)
    #else
    #    dima, xalpha = ras_calc_ndets(no, na, ras_spaces, ras1_min, ras3_max)
    #    dimb, xbeta = ras_calc_ndets(no, nb, ras_spaces, ras1_min, ras3_max)
    #end
    dima, xalpha = ras_calc_ndets(no, na, ras_spaces, ras1_min, ras3_max)
    dimb, xbeta = ras_calc_ndets(no, nb, ras_spaces, ras1_min, ras3_max)
    return RASCIAnsatz(no, na, nb, dima, dimb, dima*dimb, ras_spaces, ras1_min, ras3_max, max_h, max_p, xalpha, xbeta, false, false, 1, "direct", 1)
end

function Base.display(p::RASCIAnsatz)
    @printf(" RASCIAnsatz:: #Orbs = %-3i #α = %-2i #β = %-2i Fock Spaces: (%i, %i, %i) Dimension: %-3i RAS1 min: %i RAS3 max: %i MAX Holes: %i MAX Particles: %i\n",p.no,p.na,p.nb,p.ras_spaces[1], p.ras_spaces[2], p.ras_spaces[3], p.dim,  p.ras1_min, p.ras3_max, p.max_h, p.max_p)
end

function Base.print(p::RASCIAnsatz)
    @printf(" RASCIAnsatz:: #Orbs = %-3i #α = %-2i #β = %-2i Fock Spaces: (%i, %i, %i) Dimension: %-3i RAS1 min: %i RAS3 max: %i\n",p.no,p.na,p.nb,p.ras_spaces[1], p.ras_spaces[2], p.ras_spaces[3], p.dim,  p.ras1_min, p.ras3_max)
end

"""
    LinearMap(ints, prb::RASCIAnsatz)

Get LinearMap with takes a vector and returns action of H on that vector

# Arguments
- ints: `InCoreInts` object
- prb:  `RASCIAnsatz` object
"""
function LinearMaps.LinearMap(ints::InCoreInts, prb::RASCIAnsatz)
    a_categories = ActiveSpaceSolvers.RASCI.make_categories(prb, spin="alpha")
    b_categories = ActiveSpaceSolvers.RASCI.make_categories(prb, spin="beta")

    iters = 0
    function mymatvec(v)
        iters += 1
        @printf(" Iter: %4i\n", iters)
        #print("Iter: ", iters, " ")
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
        
        sigma1 = ActiveSpaceSolvers.RASCI.sigma_one(prb, a_categories, b_categories, ints, v)
        sigma2 = ActiveSpaceSolvers.RASCI.sigma_two(prb, a_categories, b_categories, ints, v)
        #sigma2 = permutedims(sigma1, [2,1,3])
        #sigma1 = zeros(prb.dima, prb.dimb, size(v,3))
        #sigma2 = zeros(prb.dima, prb.dimb, size(v,3))
        #sigma3 = zeros(prb.dima, prb.dimb, size(v,3))
        sigma3 = ActiveSpaceSolvers.RASCI.slow_sigma_three(prb, a_categories, b_categories, ints, v)
        #sigma3 = ActiveSpaceSolvers.RASCI.sigma_three(prb, categories, ints, v)
        
        sig = sigma1 + sigma2 + sigma3
        
        v = reshape(v,prb.dim, nr)
        sig = reshape(sig, prb.dim, nr)
        sig .+= ints.h0*v
        return sig
    end
    return LinearMap(mymatvec, prb.dim, prb.dim, issymmetric=true, ismutating=false, ishermitian=true)
end

"""
    LinearMap(ints, prb::RASCIAnsatz)

Get LinearMap with takes a vector and returns action of H on that vector

# Arguments
- ints: `InCoreInts` object
- prb:  `RASCIAnsatz` object
"""
#function LinearMaps.LinearMap(ints::InCoreInts, prb::RASCIAnsatz)
#    a_configs = ActiveSpaceSolvers.RASCI.compute_configs(prb)[1]
#    b_configs = ActiveSpaceSolvers.RASCI.compute_configs(prb)[2]
#    
#    #fill single excitation lookup tables
#    a_lookup = ActiveSpaceSolvers.RASCI.fill_lookup(prb, a_configs, prb.dima)
#    b_lookup = ActiveSpaceSolvers.RASCI.fill_lookup(prb, b_configs, prb.dimb)
#    #iters = 0
#    function mymatvec(v)
#        #iters += 1
#        #@printf(" Iter: %4i\n", iters)
#        #@printf(" %-50s", "Compute sigma 1: ")
#        #flush(stdout)
#        
#        nr = 0
#        if length(size(v)) == 1
#            nr = 1
#            v = reshape(v,prb.dim, nr)
#        else 
#            nr = size(v)[2]
#        end
#        v = reshape(v, prb.dima, prb.dimb, nr)
#        
#        sigma1 = ActiveSpaceSolvers.RASCI.compute_sigma_one(b_configs, b_lookup, v, ints, prb)
#        sigma2 = ActiveSpaceSolvers.RASCI.compute_sigma_two(a_configs, a_lookup, v, ints, prb)
#        #sigma3 = ActiveSpaceSolvers.RASCI.slow_compute_sigma_three(a_configs, b_configs, a_lookup, b_lookup, v, ints, prb)
#        sigma3 = ActiveSpaceSolvers.RASCI.compute_sigma_three(a_configs, b_configs, a_lookup, b_lookup, v, ints, prb)
#        
#        sig = sigma1 + sigma2 + sigma3
#        
#        v = reshape(v,prb.dim, nr)
#        sig = reshape(sig, prb.dim, nr)
#        sig .+= ints.h0*v
#        return sig
#    end
#    return LinearMap(mymatvec, prb.dim, prb.dim, issymmetric=true, ismutating=false, ishermitian=true)
#end

function ras_calc_ndets(no, nelec, ras_spaces, ras1_min, ras3_max)
    x = ActiveSpaceSolvers.RASCI.make_ras_x(no, nelec, ras_spaces, ras1_min, ras3_max)
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
    ActiveSpaceSolvers.compute_s2(sol::Solution)

Compute the <S^2> expectation values for each state in `sol`
"""
function ActiveSpaceSolvers.compute_s2(sol::Solution{RASCIAnsatz,T}) where {T}
    return compute_S2_expval(sol.vectors, sol.ansatz)
end

"""
    build_S2_matrix(P::RASCIAnsatz)

Build the S2 matrix in the Slater Determinant Basis  specified by `P`
"""
function ActiveSpaceSolvers.apply_S2_matrix(P::RASCIAnsatz, v::AbstractArray{T}) where T
    return apply_S2_matrix(P,v)
end

"""
"""
function ActiveSpaceSolvers.apply_sminus(v::Matrix, ansatz::RASCIAnsatz)
    if ansatz.nb + 1 > ansatz.no#={{{=#
        error(" Can't decrease Ms further")
    end
    # Sm = b'a
    # = c(IJ,s) <IJ|b'a|KL> c(KL,t)
    # = c(IJ,s)c(KL,t) <J|<I|b'a|K>|L>
    # = c(IJ,s)c(KL,t) <J|<I|ab'|K>|L> (-1)
    # = c(IJ,s)c(KL,t) <J|<I|a|K>b'|L> (-1) (-1)^ket_a.ne
    # = c(IJ,s)c(KL,t) <I|a|K><J|b'|L> (-1) (-1)^ket_a.ne
    
    nroots = size(v,2)
    bra_ansatz = RASCIAnsatz(ansatz.no, ansatz.na-1, ansatz.nb+1, ansatz.ras_spaces, ansatz.ras1_min, ansatz.ras3_max)
    
    tbla, tbla_sign = generate_single_index_lookup(bra_ansatz, ansatz, "alpha")
    tblb, tblb_sign = generate_single_index_lookup(bra_ansatz, ansatz, "beta")

    sgnK = -1
    if ansatz.na % 2 != 0 
        sgnK = -sgnK
    end
    
    #w = zeros(bra_ansatz.dima * bra_ansatz.dimb, size(v,2))
    w = zeros(bra_ansatz.dima, bra_ansatz.dimb, nroots)
    v = reshape(v, ansatz.dima, ansatz.dimb, nroots)

    for Kb in 1:size(tblb, 1)
        for Ka in 1:size(tbla, 1)
            #K = Ka + (Kb-1)*ansatz.dima
            for ai in 1:ansatz.no
                La = tbla[Ka, ai]
                La != 0 || continue
                La_sign = tbla_sign[Ka, ai]
                Lb = tblb[Kb, ai]
                Lb != 0 || continue
                Lb_sign = tblb_sign[Kb, ai]
                #L = La + (Lb-1)*bra_ansatz.dima
                #w[L,:] .+= sgnK*La_sign*Lb_sign*v[K,:]
                w[La, Lb, :] .+= sgnK*La_sign*Lb_sign*v[Ka, Kb, :]
            end
        end
    end
    w = reshape(w, bra_ansatz.dima * bra_ansatz.dimb, nroots)

    #only keep the states that aren't zero (that weren't killed by S-)
    wout = zeros(size(w,1),0)
    for i in 1:size(w,2)
        ni = norm(w[:,i])
        if isapprox(ni, 0, atol=1e-4) == false
            wout = hcat(wout, w[:,i]./ni)
        end
    end

    return wout, bra_ansatz#=}}}=#
end

"""
"""
function ActiveSpaceSolvers.apply_splus(v::Matrix, ansatz::RASCIAnsatz)

    # Sp = a'b{{{
    # = c(IJ,s) <IJ|a'b|KL> c(KL,t)
    # = c(IJ,s)c(KL,t) <J|<I|a'b|K>|L>
    # = c(IJ,s)c(KL,t) <J|<I|a'|K>b|L> (-1)^ket_a.ne
    # = c(IJ,s)c(KL,t) <I|a'|K><J|b|L> (-1)^ket_a.ne

    nroots = size(v,2)
    
    if ansatz.na + 1 > ansatz.no
        error(" Can't increase Ms further")
    end

    bra_ansatz = RASCIAnsatz(ansatz.no, ansatz.na+1, ansatz.nb-1, ansatz.ras_spaces, ansatz.ras1_min, ansatz.ras3_max)
    
    tbla, tbla_sign = generate_single_index_lookup(bra_ansatz, ansatz, "alpha")
    tblb, tblb_sign = generate_single_index_lookup(bra_ansatz, ansatz, "beta")

    sgnK = 1
    if ansatz.na % 2 != 0 
        sgnK = -sgnK
    end
    
    #w = zeros(bra_ansatz.dima * bra_ansatz.dimb, size(v,2))
    w = zeros(bra_ansatz.dima, bra_ansatz.dimb, nroots)
    v = reshape(v, ansatz.dima, ansatz.dimb, nroots)
    for Kb in 1:size(tblb, 1)
        for Ka in 1:size(tbla, 1)
            #K = Ka + (Kb-1)*ansatz.dima
            for ai in 1:ansatz.no
                La = tbla[Ka, ai]
                La != 0 || continue
                La_sign = tbla_sign[Ka, ai]
                Lb = tblb[Kb, ai]
                Lb != 0 || continue
                Lb_sign = tblb_sign[Kb, ai]
                #L = La + (Lb-1)*ansatz.dima
                #println("L ", L)
                #println("K ", K)
                w[La, Lb, :] .+= sgnK*La_sign*Lb_sign*v[Ka, Kb, :]
                #w[L,:] .+= sgnK*La_sign*Lb_sign*v[K,:]
            end
        end
    end

    w = reshape(w, bra_ansatz.dima * bra_ansatz.dimb, nroots)
    #only keep the states that aren't zero (that weren't killed by S-)
    wout = zeros(size(w,1),0)
    for i in 1:size(w,2)
        ni = norm(w[:,i])
        if isapprox(ni, 0, atol=1e-4) == false
            wout = hcat(wout, w[:,i]./ni)
        end
    end

    return wout, bra_ansatz#=}}}=#
end

"""
    build_H_matrix(ints, P::RASCIAnsatz)

Build the Hamiltonian defined by `ints` in the Slater Determinant Basis  specified by `P`
"""
function ActiveSpaceSolvers.build_H_matrix(ints::InCoreInts, p::RASCIAnsatz)
    return ActiveSpaceSolvers.RASCI.build_H_matrix(ints::InCoreInts, p::RASCIAnsatz)
end

"""
    compute_operator_c_a(bra::Solution{RASCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a operator between states `bra_v` and `ket_v` for alpha
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_c_a(bra::Solution{RASCIAnsatz,T}, 
                                                 ket::Solution{RASCIAnsatz,T}) where {T}
    return ActiveSpaceSolvers.RASCI.compute_operator_c_a(bra::Solution{RASCIAnsatz}, ket::Solution{RASCIAnsatz})
end

"""
    compute_operator_a_b(bra::Solution{RASCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a operator between states `bra_v` and `ket_v` for beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_c_b(bra::Solution{RASCIAnsatz,T}, 
                                                 ket::Solution{RASCIAnsatz,T}) where {T}
    return ActiveSpaceSolvers.RASCI.compute_operator_c_b(bra::Solution{RASCIAnsatz}, 
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
