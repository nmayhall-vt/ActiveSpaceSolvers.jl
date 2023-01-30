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

function compute_operator_cca_abb(bra::Solution{RASCIAnsatz,T}, 
                                                     ket::Solution{RASCIAnsatz,T}) where {T}
    bra.ansatz.na == ket.ansatz.na - 1 || throw(DimensionMismatch) 
    bra.ansatz.nb == ket.ansatz.nb     || throw(DimensionMismatch) 
    
    # <s|p'q'r|t>
    #
    # c(IJ,s) <IJ|a'b'b|KL> c(KL,t) = 
    # c(IJ,s) c(KL,t) <J|<I|a'b'b|K>|L> 
    # c(IJ,s) c(KL,t) <I|a'|K><J|b'b|L> 
    # c(IJ,s) c(KL,t) \sum_m <I|a'|K><J|b'|m><m|b|L> 
    ansatz_m1 = FCIAnsatz(ket.ansatz.no, ket.ansatz.na, ket.ansatz.nb-1)
    ansatz_m2 = FCIAnsatz(ket.ansatz.no, ket.ansatz.na, ket.ansatz.nb)
    
    tbl1a, tbl1a_sign, tbl1b, tbl1b_sign = generate_single_index_lookup(ansatz_m1, ket)
    tbl2a, tbl2a_sign, tbl2b, tbl2b_sign = generate_single_index_lookup(ansatz_m2, ansatz_m1)
    tbl3a, tbl3a_sign, tbl3b, tbl3b_sign = generate_single_index_lookup(bra, ansatz_m2)
    
    #
    #   TDM[pqr,s,t] = 
    tdm = zeros(Float64, size(bra)[2], size(ket)[2], no, no, no)
    
    Ksign = 1
    Lsign = 1
    for K in 1:length(tbl1a)
        for L in 1:length(tbl1b)
            for r in 1:no
                rK = tbl1a[K,r]
                rL = tbl1b[L,r]
                Ksign *= tbl1a_sign[K,r]
                Lsign *= tbl1b_sign[L,r]
                
                for q in 1:no
                    qrK = tbl2a[rK,q]
                    qrL = tbl2b[rL,q]
                    Ksign *= tbl2a_sign[rK,q]
                    Lsign *= tbl2b_sign[rL,q]
                    
                    for p in 1:no
                        pqrK = tbl3a[qrK,p]
                        pqrL = tbl3b[qrL,p]
                        Ksign *= tbl3a_sign[qrK,p]
                        Lsign *= tbl3b_sign[qrL,p]
                        
                        @views tdm_pqr = tdm[:,:,p,q,r] 
                        @views v1_IJ = v1[:,I,J]
                        @views v2_KL = v2[:,K,L]
                        
                        @tensor begin 
                            tdm[s,t] += v1_IJ[s] * v2_KL[t]
                        end
                    end
                end
            end
        end
    end
end
