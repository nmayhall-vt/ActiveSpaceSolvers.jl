#using KrylovKit
using LinearAlgebra
using Printf
using NPZ
using StaticArrays
using JLD2
using BenchmarkTools
#using InteractiveUtils
using LinearMaps
using TensorOperations

#using FermiCG
using QCBase
using InCoreIntegrals 
using BlockDavidson

"""
    compute_operator_a_a(bra::Solution{RASCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a operator between states `bra_v` and `ket_v` for alpha
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function compute_operator_a_a(bra::Solution{RASCIAnsatz,T}, 
                              ket::Solution{RASCIAnsatz,T}) where {T}
    n_orb(bra) == n_orb(ket) || throw(DimensionMismatch) 
end

"""
    compute_operator_a_b(bra::Solution{RASCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a operator between states `bra_v` and `ket_v` for beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function compute_operator_a_b(bra::Solution{RASCIAnsatz,T}, 
                              ket::Solution{RASCIAnsatz,T}) where {T}
end

"""
    compute_operator_ca_aa(bra::Solution{RASCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a'a operators between states `bra_v` and `ket_v` for alpha-alpha
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function compute_operator_ca_aa(bra::Solution{RASCIAnsatz,T}, 
                                ket::Solution{RASCIAnsatz,T}) where {T}
end

"""
    compute_operator_ca_bb(bra::Solution{RASCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a'a operators between states `bra_v` and `ket_v` for beta-beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function compute_operator_ca_bb(bra::Solution{RASCIAnsatz,T}, 
                                                   ket::Solution{RASCIAnsatz,T}) where {T}
end


"""
    compute_operator_ca_ab(bra::Solution{RASCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a'a operators between states `bra_v` and `ket_v` for alpha-beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function compute_operator_ca_ab(bra::Solution{RASCIAnsatz,T}, 
                                                   ket::Solution{RASCIAnsatz,T}) where {T}
end


"""
    compute_operator_cc_aa(bra::Solution{RASCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a'a' operators between states `bra_v` and `ket_v` for beta-beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function compute_operator_cc_bb(bra::Solution{RASCIAnsatz,T}, 
                                                   ket::Solution{RASCIAnsatz,T}) where {T}
end


"""
    compute_operator_cc_aa(bra::Solution{RASCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a'a' operators between states `bra_v` and `ket_v` for alpha-alpha 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function compute_operator_cc_aa(bra::Solution{RASCIAnsatz,T}, 
                                                   ket::Solution{RASCIAnsatz,T}) where {T}
end


"""
    compute_operator_cc_ab(bra::Solution{RASCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a'a' operators between states `bra_v` and `ket_v` for alpha-beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function compute_operator_cc_ab(bra::Solution{RASCIAnsatz,T}, 
                                                   ket::Solution{RASCIAnsatz,T}) where {T}
end


"""
    compute_operator_cca_aaa(bra::Solution{RASCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a'a'a operators between states `bra_v` and `ket_v` for alpha-alpha-alpha 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function compute_operator_cca_aaa(bra::Solution{RASCIAnsatz,T}, 
                                                   ket::Solution{RASCIAnsatz,T}) where {T}
end


"""
    compute_operator_cca_bbb(bra::Solution{RASCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a'a'a operators between states `bra_v` and `ket_v` for beta-beta-beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function compute_operator_cca_bbb(bra::Solution{RASCIAnsatz,T}, 
                                                   ket::Solution{RASCIAnsatz,T}) where {T}
end


"""
    compute_operator_cca_aba(bra::Solution{RASCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a'a'a operators between states `bra_v` and `ket_v` for alpha-beta-alpha 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function compute_operator_cca_aba(bra::Solution{RASCIAnsatz,T}, 
                                                     ket::Solution{RASCIAnsatz,T}) where {T}
end


"""
    compute_operator_cca_abb(bra::Solution{RASCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a'a'a operators between states `bra_v` and `ket_v` for alpha-beta-beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""

function compute_operator_cca_abb(bra::Solution{RASCIAnsatz}, 
                                                     ket::Solution{RASCIAnsatz}) 
    bra.ansatz.na - 1 == ket.ansatz.na || throw(DimensionMismatch) 
    bra.ansatz.nb == ket.ansatz.nb     || throw(DimensionMismatch) 
    
    # <s|p'q'r|t>
    # I and K are α strings
    # J and L are β strings
    # c(IJ,s) <IJ|a'b'b|KL> c(KL,t) = 
    # c(IJ,s) c(KL,t) <J|<I|a'b'b|K>|L> 
    # c(IJ,s) c(KL,t) <I|a'|K><J|b'b|L> 
    # c(IJ,s) c(KL,t) \sum_m <I|a'|K><J|b'|m><m|b|L> 
    
    ansatz_m1 = RASCIAnsatz(ket.ansatz.no, ket.ansatz.na, ket.ansatz.nb-1, ket.ansatz.fock)
    #ansatz_m2 = RASCIAnsatz(ket.ansatz.no, ket.ansatz.na, ket.ansatz.nb, ket.ansatz.fock)
    
    tbl_a, tbl_a_sign = generate_single_index_lookup(bra.ansatz, ket.ansatz, "alpha")
    #println("α dim ", ket.ansatz.dima, " --> α dim ", bra.ansatz.dima)

    tbl1b, tbl1b_sign = generate_single_index_lookup(ansatz_m1, ket.ansatz, "beta")
    #println("β dim ", ket.ansatz.dimb, " --> β dim ", ansatz_m1.dimb)
    
    tbl2b, tbl2b_sign = generate_single_index_lookup(bra.ansatz, ansatz_m1, "beta")
    #println("β dim ", ansatz_m1.dimb, " --> β dim ", bra.ansatz.dimb)
    
    bra_M = size(bra,2)
    ket_M = size(ket,2)

    v1 = reshape(bra.vectors, bra.ansatz.dima, bra.ansatz.dimb, bra_M)
    v2 = reshape(ket.vectors, ket.ansatz.dima, ket.ansatz.dimb, ket_M)
    v1 = permutedims(v1, [3,1,2])
    v2 = permutedims(v2, [3,1,2])
    
    #
    #   TDM[pqr,s,t] = 
    tdm = zeros(Float64, bra_M, ket_M,  bra.ansatz.no, bra.ansatz.no, bra.ansatz.no)
    
    for K in 1:size(tbl_a,1)
        for L in 1:size(tbl1b,1)
            for p in 1:bra.ansatz.no
                I = tbl_a[K,p]
                I != 0 || continue
                Ksign = tbl_a_sign[K,p]
                
                for q in 1:bra.ansatz.no
                    for r in 1:bra.ansatz.no
                        rL = tbl1b[L,r]
                        rL != 0 || continue
                        Lsign = tbl1b_sign[L,r]
                        J = tbl2b[rL,q]
                        J != 0 || continue
                        Lsign = Lsign*tbl2b_sign[rL,q]

                        @views tdm_pqr = tdm[:,:,p,q,r] 
                        @views v1_IJ = v1[:,I,J]
                        @views v2_KL = v2[:,K,L]
                        sgn = Ksign*Lsign

                        if sgn == 1
                            @tensor begin 
                                tdm_pqr[s,t] += v1_IJ[s] * v2_KL[t]
                            end
                        else
                            @tensor begin 
                                tdm_pqr[s,t] -= v1_IJ[s] * v2_KL[t]
                            end
                        end
                    end
                end
            end
        end
    end
    #                      [p,q,r,s,t]
    tdm = permutedims(tdm, [3,4,5,1,2])
    return tdm
end
