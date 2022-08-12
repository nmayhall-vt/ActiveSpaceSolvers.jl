using ActiveSpaceSolvers
import LinearMaps


"""
Type containing all the metadata needed to define a FCI problem 

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
"""
struct FCIAnsatz <: Ansatz 
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
end

"""
    FCIAnsatz(no, na, nb)

Constructor

# Arguments
- `no`: Number of spatial orbitals
- `na`: Number of α electrons
- `nb`: Number of β electrons
"""
function FCIAnsatz(no, na, nb)
    na <= no || throw(DimensionMismatch)
    nb <= no || throw(DimensionMismatch)
    dima = calc_nchk(no,na)
    dimb = calc_nchk(no,nb)
    return FCIAnsatz(no, na, nb, dima, dimb, dima*dimb, false, false, 1, "direct", 1)
end

function display(p::FCIAnsatz)
    @printf(" FCIAnsatz:: #Orbs = %-3i #α = %-2i #β = %-2i Dimension: %-9i\n",p.no,p.na,p.nb,p.dim)
end


"""
    LinearMap(ints, prb::FCIAnsatz)

Get LinearMap with takes a vector and returns action of H on that vector

# Arguments
- ints: `InCoreInts` object
- prb:  `FCIAnsatz` object
"""
function LinearMaps.LinearMap(ints::InCoreInts, prb::FCIAnsatz)
    #={{{=#
    ket_a = DeterminantString(prb.no, prb.na)
    ket_b = DeterminantString(prb.no, prb.nb)

    #@btime lookup_a = $fill_ca_lookup2($ket_a)
    lookup_a = fill_ca_lookup2(ket_a)
    lookup_b = fill_ca_lookup2(ket_b)
    iters = 0
    function mymatvec(v)
        iters += 1
        #@printf(" Iter: %4i\n", iters)
        nr = 0
        if length(size(v)) == 1
            nr = 1
            v = reshape(v,ket_a.max*ket_b.max, nr)
        else 
            nr = size(v)[2]
        end
        v = reshape(v, ket_a.max, ket_b.max, nr)
        sig = compute_ab_terms2(v, ints, prb, lookup_a, lookup_b)
        sig += compute_ss_terms2(v, ints, prb, lookup_a, lookup_b)

        v = reshape(v, ket_a.max*ket_b.max, nr)
        sig = reshape(sig, ket_a.max*ket_b.max, nr)
        return sig 
    end
    return LinearMap(mymatvec, prb.dim, prb.dim; issymmetric=true, ismutating=false, ishermitian=true)
end
#=}}}=#


"""
    build_H_matrix(ints, P::FCIAnsatz)

Build the Hamiltonian defined by `ints` in the Slater Determinant Basis  specified by `P`
"""
function ActiveSpaceSolvers.build_H_matrix(ints::InCoreInts{T}, P::FCIAnsatz) where T
#={{{=#
    Hmat = zeros(T, P.dim, P.dim)

    Hdiag_a = precompute_spin_diag_terms(ints,P,P.na)
    Hdiag_b = precompute_spin_diag_terms(ints,P,P.nb)
    # 
    #   Create ci_strings
    ket_a = DeterminantString(P.no, P.na)
    ket_b = DeterminantString(P.no, P.nb)
    bra_a = DeterminantString(P.no, P.na)
    bra_b = DeterminantString(P.no, P.nb)
    #   
    #   Add spin diagonal components
    Hmat += kron(Matrix(1.0I, P.dimb, P.dimb), Hdiag_a)
    Hmat += kron(Hdiag_b, Matrix(1.0I, P.dima, P.dima))
    #
    #   Add opposite spin term (todo: make this reasonably efficient)
    Hmat += compute_ab_terms_full(ints, P, T=T)
    
    Hmat = .5*(Hmat+Hmat')

    return Hmat
end
#=}}}=#


"""
    build_S2_matrix(P::FCIAnsatz)

Build the S2 matrix in the Slater Determinant Basis  specified by `P`
"""
function ActiveSpaceSolvers.build_S2_matrix(P::FCIAnsatz) where T
#={{{=#
    return build_S2_matrix(P)
end
#=}}}=#


"""
    compute_operator_a_a(bra::Solution{FCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a operator between states `bra_v` and `ket_v` for alpha
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_a_a(bra::Solution{FCIAnsatz,T}, 
                                                 ket::Solution{FCIAnsatz,T}) where {T}
    #={{{=#
    n_orbs(bra) == n_orbs(ket) || throw(DimensionMismatch) 
    return compute_annihilation(n_orbs(bra), 
                                n_elec_a(bra), n_elec_b(bra),
                                n_elec_a(ket), n_elec_b(ket),
                                bra.vectors, ket.vectors,
                                "alpha")

    
#=}}}=#
end



"""
    compute_operator_a_b(bra::Solution{FCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a operator between states `bra_v` and `ket_v` for beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_a_b(bra::Solution{FCIAnsatz,T}, 
                                                 ket::Solution{FCIAnsatz,T}) where {T}
    #={{{=#
    n_orbs(bra) == n_orbs(ket) || throw(DimensionMismatch) 
    return compute_annihilation(n_orbs(bra), 
                                n_elec_a(bra), n_elec_b(bra),
                                n_elec_a(ket), n_elec_b(ket),
                                bra.vectors, ket.vectors,
                                "beta")

    
#=}}}=#
end



"""
    compute_operator_ca_aa(bra::Solution{FCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a'a operators between states `bra_v` and `ket_v` for alpha-alpha
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_ca_aa(bra::Solution{FCIAnsatz,T}, 
                                                   ket::Solution{FCIAnsatz,T}) where {T}
    #={{{=#
    n_orbs(bra) == n_orbs(ket) || throw(DimensionMismatch) 
    return compute_Aa(n_orbs(bra), 
                      n_elec_a(bra), n_elec_b(bra),
                      n_elec_a(ket), n_elec_b(ket),
                      bra.vectors, ket.vectors,
                      "alpha")

    
#=}}}=#
end

"""
    compute_operator_ca_bb(bra::Solution{FCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a'a operators between states `bra_v` and `ket_v` for beta-beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_ca_bb(bra::Solution{FCIAnsatz,T}, 
                                                   ket::Solution{FCIAnsatz,T}) where {T}
    #={{{=#
    n_orbs(bra) == n_orbs(ket) || throw(DimensionMismatch) 
    return compute_Aa(n_orbs(bra), 
                      n_elec_a(bra), n_elec_b(bra),
                      n_elec_a(ket), n_elec_b(ket),
                      bra.vectors, ket.vectors,
                      "beta")

    
#=}}}=#
end


"""
    compute_operator_ca_ab(bra::Solution{FCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a'a operators between states `bra_v` and `ket_v` for alpha-beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_ca_ab(bra::Solution{FCIAnsatz,T}, 
                                                   ket::Solution{FCIAnsatz,T}) where {T}
    #={{{=#
    n_orbs(bra) == n_orbs(ket) || throw(DimensionMismatch) 
    return compute_Ab(n_orbs(bra), 
                      n_elec_a(bra), n_elec_b(bra),
                      n_elec_a(ket), n_elec_b(ket),
                      bra.vectors, ket.vectors)

    
#=}}}=#
end


"""
    compute_operator_cc_aa(bra::Solution{FCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a'a' operators between states `bra_v` and `ket_v` for beta-beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_cc_bb(bra::Solution{FCIAnsatz,T}, 
                                                   ket::Solution{FCIAnsatz,T}) where {T}
    #={{{=#
    n_orbs(bra) == n_orbs(ket) || throw(DimensionMismatch) 
    return compute_AA(n_orbs(bra), 
                      n_elec_a(bra), n_elec_b(bra),
                      n_elec_a(ket), n_elec_b(ket),
                      bra.vectors, ket.vectors,
                      "beta")

    
#=}}}=#
end


"""
    compute_operator_cc_aa(bra::Solution{FCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a'a' operators between states `bra_v` and `ket_v` for alpha-alpha 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_cc_aa(bra::Solution{FCIAnsatz,T}, 
                                                   ket::Solution{FCIAnsatz,T}) where {T}
    #={{{=#
    n_orbs(bra) == n_orbs(ket) || throw(DimensionMismatch) 
    return compute_AA(n_orbs(bra), 
                      n_elec_a(bra), n_elec_b(bra),
                      n_elec_a(ket), n_elec_b(ket),
                      bra.vectors, ket.vectors,
                      "alpha")

    
#=}}}=#
end


"""
    compute_operator_cc_ab(bra::Solution{FCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a'a' operators between states `bra_v` and `ket_v` for alpha-beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_cc_ab(bra::Solution{FCIAnsatz,T}, 
                                                   ket::Solution{FCIAnsatz,T}) where {T}
    #={{{=#
    n_orbs(bra) == n_orbs(ket) || throw(DimensionMismatch) 
    return compute_AB(n_orbs(bra), 
                      n_elec_a(bra), n_elec_b(bra),
                      n_elec_a(ket), n_elec_b(ket),
                      bra.vectors, ket.vectors)

    
#=}}}=#
end


"""
    compute_operator_cca_aaa(bra::Solution{FCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a'a'a operators between states `bra_v` and `ket_v` for alpha-alpha-alpha 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_cca_aaa(bra::Solution{FCIAnsatz,T}, 
                                                   ket::Solution{FCIAnsatz,T}) where {T}
    #={{{=#
    n_orbs(bra) == n_orbs(ket) || throw(DimensionMismatch) 
    return compute_AAa(n_orbs(bra), 
                      n_elec_a(bra), n_elec_b(bra),
                      n_elec_a(ket), n_elec_b(ket),
                      bra.vectors, ket.vectors, 
                      "alpha")

    
#=}}}=#
end


"""
    compute_operator_cca_bbb(bra::Solution{FCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a'a'a operators between states `bra_v` and `ket_v` for beta-beta-beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_cca_bbb(bra::Solution{FCIAnsatz,T}, 
                                                   ket::Solution{FCIAnsatz,T}) where {T}
    #={{{=#
    n_orbs(bra) == n_orbs(ket) || throw(DimensionMismatch) 
    return compute_AAa(n_orbs(bra), 
                      n_elec_a(bra), n_elec_b(bra),
                      n_elec_a(ket), n_elec_b(ket),
                      bra.vectors, ket.vectors, 
                      "beta")

    
#=}}}=#
end


"""
    compute_operator_cca_aba(bra::Solution{FCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a'a'a operators between states `bra_v` and `ket_v` for alpha-beta-alpha 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_cca_aba(bra::Solution{FCIAnsatz,T}, 
                                                     ket::Solution{FCIAnsatz,T}) where {T}
    #={{{=#
    n_orbs(bra) == n_orbs(ket) || throw(DimensionMismatch) 
    return compute_ABa(n_orbs(bra), 
                      n_elec_a(bra), n_elec_b(bra),
                      n_elec_a(ket), n_elec_b(ket),
                      bra.vectors, ket.vectors)

    
#=}}}=#
end


"""
    compute_operator_cca_abb(bra::Solution{FCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a'a'a operators between states `bra_v` and `ket_v` for alpha-beta-beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_cca_abb(bra::Solution{FCIAnsatz,T}, 
                                                     ket::Solution{FCIAnsatz,T}) where {T}
    #={{{=#
    n_orbs(bra) == n_orbs(ket) || throw(DimensionMismatch) 
    return compute_ABb(n_orbs(bra), 
                      n_elec_a(bra), n_elec_b(bra),
                      n_elec_a(ket), n_elec_b(ket),
                      bra.vectors, ket.vectors)

    
#=}}}=#
end


"""
    compute_1rdm(sol::Solution{FCIAnsatz,T}; root=1) where {T}

"""
function ActiveSpaceSolvers.compute_1rdm(sol::Solution{FCIAnsatz,T}; root=1) where {T}
    #={{{=#

    rdma = compute_Aa(n_orbs(sol),
                      n_elec_a(sol), n_elec_b(sol),                     
                      n_elec_a(sol), n_elec_b(sol),                     
                      reshape(sol.vectors[:,root], dim(sol), 1), 
                      reshape(sol.vectors[:,root], dim(sol), 1), 
                      "alpha") 

    rdmb = compute_Aa(n_orbs(sol),
                      n_elec_a(sol), n_elec_b(sol),                     
                      n_elec_a(sol), n_elec_b(sol),                     
                      reshape(sol.vectors[:,root], dim(sol), 1), 
                      reshape(sol.vectors[:,root], dim(sol), 1), 
                      "beta") 


    rdma = reshape(rdma, n_orbs(sol), n_orbs(sol))
    rdmb = reshape(rdmb, n_orbs(sol), n_orbs(sol))
    return rdma, rdmb
end
#=}}}=#


"""
    compute_2rdm(sol::Solution{A,T}; root=1) where {A,T}

"""
function ActiveSpaceSolvers.compute_1rdm_2rdm(sol::Solution{FCIAnsatz,T}; root=1) where {T}
    #={{{=#

    return compute_rdm1_rdm2(sol.ansatz, sol.vectors[:,root], sol.vectors[:,root])
end
#=}}}=#






