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
function compute_operator_c_a(bra::Solution{RASCIAnsatz,T}, 
                              ket::Solution{RASCIAnsatz,T}) where {T}

    bra.ansatz.na-1 == ket.ansatz.na     || throw(DimensionMismatch) #={{{=#
    bra.ansatz.nb == ket.ansatz.nb     || throw(DimensionMismatch) 
    
    tbl1a, tbl1a_sign = generate_single_index_lookup(bra.ansatz, ket.ansatz, "alpha")
    
    bra_M = size(bra,2)
    ket_M = size(ket,2)
    
    v1 = reshape(bra.vectors, bra.ansatz.dima, bra.ansatz.dimb, bra_M)
    v2 = reshape(ket.vectors, ket.ansatz.dima, ket.ansatz.dimb, ket_M)
    
    v1 = permutedims(v1, [2,3,1])
    v2 = permutedims(v2, [2,3,1])
    
    #   TDM[p,s,t] = 
    tdm = zeros(Float64, bra_M, ket_M,  bra.ansatz.no)

    for K in 1:size(tbl1a, 1)
        for p in 1:bra.ansatz.no
            I = tbl1a[K,p]
            I != 0 || continue
            Ksign = tbl1a_sign[K, p]
            @views tdm_pqr = tdm[:,:,p] 
            @views v1_IJ = v1[:,:,I]
            @views v2_KL = v2[:,:,K]

            if Ksign == 1
                @tensor begin 
                    tdm_pqr[s,t] += v1_IJ[I,s] * v2_KL[I,t]
                end
            else
                @tensor begin 
                    tdm_pqr[s,t] -= v1_IJ[I,s] * v2_KL[I,t]
                end
            end
        end
    end
    #                      [p,s,t]
    tdm = permutedims(tdm, [3,1,2])
    return tdm#=}}}=#
end

"""
    compute_operator_a_b(bra::Solution{RASCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a operator between states `bra_v` and `ket_v` for beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function compute_operator_c_b(bra::Solution{RASCIAnsatz,T}, 
                              ket::Solution{RASCIAnsatz,T}) where {T}
    bra.ansatz.na == ket.ansatz.na     || throw(DimensionMismatch) #={{{=#
    bra.ansatz.nb-1 == ket.ansatz.nb     || throw(DimensionMismatch) 
    
    tbl1b, tbl1b_sign = generate_single_index_lookup(bra.ansatz, ket.ansatz, "beta")
    
    bra_M = size(bra,2)
    ket_M = size(ket,2)
    
    v1 = reshape(bra.vectors, bra.ansatz.dima, bra.ansatz.dimb, bra_M)
    v2 = reshape(ket.vectors, ket.ansatz.dima, ket.ansatz.dimb, ket_M)
    
    v1 = permutedims(v1, [1,3,2])
    v2 = permutedims(v2, [1,3,2])
    
    sgnK = 1 
    if (ket.ansatz.na) % 2 != 0 
        sgnK = -sgnK
    end
    
    #   TDM[p,s,t] = 
    tdm = zeros(Float64, bra_M, ket_M,  bra.ansatz.no)
    
    for L in 1:size(tbl1b, 1)
        for p in 1:bra.ansatz.no
            J = tbl1b[L,p]
            J != 0 || continue
            Lsign = tbl1b_sign[L, p]
            @views tdm_pqr = tdm[:,:,p] 
            @views v1_IJ = v1[:,:,J]
            @views v2_KL = v2[:,:,L]
            Lsign = Lsign*sgnK

            if Lsign == 1
                @tensor begin 
                    tdm_pqr[s,t] += v1_IJ[J,s] * v2_KL[J,t]
                end
            else
                @tensor begin 
                    tdm_pqr[s,t] -= v1_IJ[J,s] * v2_KL[J,t]
                end
            end
        end
    end
    #                      [p,s,t]
    tdm = permutedims(tdm, [3,1,2])
    return tdm#=}}}=#
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
    bra.ansatz.na == ket.ansatz.na     || throw(DimensionMismatch) #={{{=#
    bra.ansatz.nb == ket.ansatz.nb     || throw(DimensionMismatch) 
    
    # <s|p'q'|t>
    # I and K are α strings
    # J and L are β strings
    # c(IJ,s) <IJ|a'a|KL> c(KL,t) =
    # c(IJ,s) c(KL,t) <J|<I|a'a|K>|L>
    # c(IJ,s) c(KL,t) <J|L><I|a'a|K>     
    # c(IJ,s) c(KL,t) sum_m<J|L><I|a'|m><m|a|K>     
    
    ansatz_m1 = RASCIAnsatz(ket.ansatz.no, ket.ansatz.na-1, ket.ansatz.nb, ket.ansatz.fock)
    
    tbl1a, tbl1a_sign = generate_single_index_lookup(ansatz_m1, ket.ansatz, "alpha")
    tbl2a, tbl2a_sign = generate_single_index_lookup(bra.ansatz, ansatz_m1, "alpha")

    bra_M = size(bra,2)
    ket_M = size(ket,2)

    v1 = reshape(bra.vectors, bra.ansatz.dima, bra.ansatz.dimb, bra_M)
    v2 = reshape(ket.vectors, ket.ansatz.dima, ket.ansatz.dimb, ket_M)
    
    v1 = permutedims(v1, [2,3,1])
    v2 = permutedims(v2, [2,3,1])
    
    #   TDM[pq,s,t] = 
    tdm = zeros(Float64, bra_M, ket_M,  bra.ansatz.no, bra.ansatz.no)
    
    for K in 1:size(tbl1a, 1)
        for q in 1:bra.ansatz.no
            for p in 1:bra.ansatz.no
                Kq = tbl1a[K,q]
                Kq != 0 || continue
                Ksign = tbl1a_sign[K, q]
                I = tbl2a[Kq, p]
                I != 0 || continue
                Ksign = Ksign*tbl2a_sign[Kq, p]
                @views tdm_pqr = tdm[:,:,p,q] 
                @views v1_IJ = v1[:,:,I]
                @views v2_KL = v2[:,:,K]

                if Ksign == 1
                    @tensor begin 
                        tdm_pqr[s,t] += v1_IJ[I,s] * v2_KL[I,t]
                    end
                else
                    @tensor begin 
                        tdm_pqr[s,t] -= v1_IJ[I,s] * v2_KL[I,t]
                    end
                end
            end
        end
    end
    #                      [p,q,s,t]
    tdm = permutedims(tdm, [3,4,1,2])
    return tdm#=}}}=#
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
    bra.ansatz.na == ket.ansatz.na     || throw(DimensionMismatch) #={{{=#
    bra.ansatz.nb == ket.ansatz.nb     || throw(DimensionMismatch) 
    
    # <s|p'q'|t>
    # I and K are α strings
    # J and L are β strings
    # c(IJ,s) <IJ|b'b|KL> c(KL,t) =
    # c(IJ,s) c(KL,t) <J|<I|b'b|K>|L>
    # c(IJ,s) c(KL,t) <J|<I|b'|K>b|L> (-1)^ket.ansatz.na
    # c(IJ,s) c(KL,t) <I|K><J|b'b|L>     
    # c(IJ,s) c(KL,t) sum_m<I|K><J|b|m><m|'b|L>     
    
    ansatz_m1 = RASCIAnsatz(ket.ansatz.no, ket.ansatz.na, ket.ansatz.nb-1, ket.ansatz.fock)
    
    tbl1b, tbl1b_sign = generate_single_index_lookup(ansatz_m1, ket.ansatz, "beta")
    tbl2b, tbl2b_sign = generate_single_index_lookup(bra.ansatz, ansatz_m1, "beta")
    
    bra_M = size(bra,2)
    ket_M = size(ket,2)

    v1 = reshape(bra.vectors, bra.ansatz.dima, bra.ansatz.dimb, bra_M)
    v2 = reshape(ket.vectors, ket.ansatz.dima, ket.ansatz.dimb, ket_M)
    
    v1 = permutedims(v1, [1,3,2])
    v2 = permutedims(v2, [1,3,2])
    
    #   TDM[pq,s,t] = 
    tdm = zeros(Float64, bra_M, ket_M,  bra.ansatz.no, bra.ansatz.no)
    
    for L in 1:size(tbl1b, 1)
        for q in 1:bra.ansatz.no
            for p in 1:bra.ansatz.no
                Lq = tbl1b[L,q]
                Lq != 0 || continue
                Lsign = tbl1b_sign[L, q]
                J = tbl2b[Lq, p]
                J != 0 || continue
                Lsign = Lsign*tbl2b_sign[Lq, p]
                @views tdm_pqr = tdm[:,:,p,q] 
                @views v1_IJ = v1[:,:,J]
                @views v2_KL = v2[:,:,L]

                if Lsign == 1
                    @tensor begin 
                        tdm_pqr[s,t] += v1_IJ[J,s] * v2_KL[J,t]
                    end
                else
                    @tensor begin 
                        tdm_pqr[s,t] -= v1_IJ[J,s] * v2_KL[J,t]
                    end
                end
            end
        end
    end
    #                      [p,q,s,t]
    tdm = permutedims(tdm, [3,4,1,2])
    return tdm#=}}}=#
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
    bra.ansatz.na-1 == ket.ansatz.na     || throw(DimensionMismatch) #={{{=#
    bra.ansatz.nb+1 == ket.ansatz.nb     || throw(DimensionMismatch) 
    
    # <s|p'q'|t>
    # I and K are α strings
    # J and L are β strings
    # c(IJ,s) <IJ|a'b|KL> c(KL,t) =
    # c(IJ,s) c(KL,t) <J|<I|a'b|K>|L>
    # c(IJ,s) c(KL,t) <J|<I|a'|K>b|L> (-1)^ket.ansatz.na
    # c(IJ,s) c(KL,t) <I|a'|K><J|b|L> (-1)^ket.ansatz.na
    
    tbl1a, tbl1a_sign = generate_single_index_lookup(bra.ansatz, ket.ansatz, "alpha")
    tbl1b, tbl1b_sign = generate_single_index_lookup(bra.ansatz, ket.ansatz, "beta")
    
    bra_M = size(bra,2)
    ket_M = size(ket,2)

    v1 = reshape(bra.vectors, bra.ansatz.dima, bra.ansatz.dimb, bra_M)
    v2 = reshape(ket.vectors, ket.ansatz.dima, ket.ansatz.dimb, ket_M)
    
    v1 = permutedims(v1, [3,1,2])
    v2 = permutedims(v2, [3,1,2])
    
    #   TDM[pq,s,t] = 
    tdm = zeros(Float64, bra_M, ket_M,  bra.ansatz.no, bra.ansatz.no)
    
    sgnK = 1 
    if (ket.ansatz.na) % 2 != 0 
        sgnK = -sgnK
    end

    for K in 1:size(tbl1a, 1)
        for p in 1:bra.ansatz.no
            I = tbl1a[K,p]
            I != 0 || continue
            Ksign = tbl1a_sign[K, p]
            for L in 1:size(tbl1b, 1)
                for q in 1:bra.ansatz.no
                    J = tbl1b[L, q]
                    J != 0 || continue
                    Lsign = tbl1b_sign[L, q]
                    @views tdm_pqr = tdm[:,:,p,q] 
                    @views v1_IJ = v1[:,:,J]
                    @views v2_KL = v2[:,:,L]

                    sgn = Ksign*Lsign*sgnK

                    if sgn == 1
                        @tensor begin 
                            tdm_pqr[s,t] += v1_IJ[J,s] * v2_KL[J,t]
                        end
                    else
                        @tensor begin 
                            tdm_pqr[s,t] -= v1_IJ[J,s] * v2_KL[J,t]
                        end
                    end
                end
            end
        end

    end
    #                      [p,q,s,t]
    tdm = permutedims(tdm, [3,4,1,2])
    return tdm#=}}}=#
end


"""
    compute_operator_cc_aa(bra::solution{rasciansatz,t}, ket::solution{fciansatz,t}) where {t}

Compute representation of a'a' operators between states `bra_v` and `ket_v` for beta-beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function compute_operator_cc_bb(bra::Solution{RASCIAnsatz,T}, 
                                                   ket::Solution{RASCIAnsatz,T}) where {T}
    bra.ansatz.na == ket.ansatz.na     || throw(DimensionMismatch) #={{{=#
    bra.ansatz.nb-2 == ket.ansatz.nb     || throw(DimensionMismatch) 
    
    # <s|p'q'|t>
    # I and K are α strings
    # J and L are β strings
    # c(IJ,s) <IJ|b'b'|KL> c(KL,t) =
    # c(IJ,s) c(KL,t) <J|<I|b'b'|K>|L>
    # c(IJ,s) c(KL,t) <I|K><J|b'b'|L>
    # c(IJ,s) c(KL,t) sum_m <I|K><J|b'|m><m|b'|L>
    
    ansatz_m1 = RASCIAnsatz(ket.ansatz.no, ket.ansatz.na, ket.ansatz.nb+1, ket.ansatz.fock)
    
    tbl1b, tbl1b_sign = generate_single_index_lookup(ansatz_m1, ket.ansatz, "beta")
    tbl2b, tbl2b_sign = generate_single_index_lookup(bra.ansatz, ansatz_m1, "beta")
    
    v1 = reshape(bra.vectors, bra.ansatz.dima, bra.ansatz.dimb, bra_M)
    v2 = reshape(ket.vectors, ket.ansatz.dima, ket.ansatz.dimb, ket_M)
    
    v1 = permutedims(v1, [1,3,2])
    v2 = permutedims(v2, [1,3,2])
    
    #   TDM[pq,s,t] = 
    tdm = zeros(Float64, bra_M, ket_M,  bra.ansatz.no, bra.ansatz.no)
    
    for L in 1:size(tbl1b,1)
        for p in 1:bra.ansatz.no
            for q in 1:bra.ansatz.no
                rL = tbl1b[L,q]
                rL != 0 || continue
                Lsign = tbl1b_sign[L,q]
                J = tbl2b[rL,p]
                J != 0 || continue
                Lsign = Lsign*tbl2b_sign[rL,p]

                @views tdm_pqr = tdm[:,:,p,q] 
                @views v1_IJ = v1[:,:,J]
                @views v2_KL = v2[:,:,L]

                if Ksign == 1
                    @tensor begin 
                        tdm_pqr[s,t] += v1_IJ[J,s] * v2_KL[J,t]
                    end
                else
                    @tensor begin 
                        tdm_pqr[s,t] -= v1_IJ[J,s] * v2_KL[J,t]
                    end
                end
            end
        end
    end
    #                      [p,q,s,t]
    tdm = permutedims(tdm, [3,4,1,2])
    return tdm#=}}}=#
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
    bra.ansatz.na-2 == ket.ansatz.na     || throw(DimensionMismatch) #={{{=#
    bra.ansatz.nb == ket.ansatz.nb     || throw(DimensionMismatch) 
    
    # <s|p'q'|t>
    # I and K are α strings
    # J and L are β strings
    # c(IJ,s) <IJ|a'a'|KL> c(KL,t) =
    # c(IJ,s) c(KL,t) <J|<I|a'a'|K>|L>
    # c(IJ,s) c(KL,t) <J|L><I|a'a'|K>
    # c(IJ,s) c(KL,t) sum_m <J|L><I|a'|m><m|a'|K>
    
    ansatz_m1 = RASCIAnsatz(ket.ansatz.no, ket.ansatz.na+1, ket.ansatz.nb, ket.ansatz.fock)
    
    tbl1a, tbl1a_sign = generate_single_index_lookup(ansatz_m1, ket.ansatz, "alpha")
    tbl2a, tbl2a_sign = generate_single_index_lookup(bra.ansatz, ansatz_m1, "alpha")
    
    v1 = reshape(bra.vectors, bra.ansatz.dima, bra.ansatz.dimb, bra_M)
    v2 = reshape(ket.vectors, ket.ansatz.dima, ket.ansatz.dimb, ket_M)
    
    v1 = permutedims(v1, [2,3,1])
    v2 = permutedims(v2, [2,3,1])
    
    #   TDM[pq,s,t] = 
    tdm = zeros(Float64, bra_M, ket_M,  bra.ansatz.no, bra.ansatz.no)
    
    for K in 1:size(tbl1a,1)
        for p in 1:bra.ansatz.no
            for q in 1:bra.ansatz.no
                rK = tbl1a[K,q]
                rK != 0 || continue
                Ksign = tbl1a_sign[K,q]
                I = tbl2a[rK,p]
                I != 0 || continue
                Ksign = Ksign*tbl2a_sign[rK,p]

                @views tdm_pqr = tdm[:,:,p,q] 
                @views v1_IJ = v1[:,:,I]
                @views v2_KL = v2[:,:,K]

                if Ksign == 1
                    @tensor begin 
                        tdm_pqr[s,t] += v1_IJ[I,s] * v2_KL[I,t]
                    end
                else
                    @tensor begin 
                        tdm_pqr[s,t] -= v1_IJ[I,s] * v2_KL[I,t]
                    end
                end
            end
        end
    end
    #                      [p,q,s,t]
    tdm = permutedims(tdm, [3,4,1,2])
    return tdm#=}}}=#
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
    bra.ansatz.na-1 == ket.ansatz.na     || throw(DimensionMismatch) #={{{=#
    bra.ansatz.nb-1 == ket.ansatz.nb     || throw(DimensionMismatch) 
    
    # <s|p'q'|t>
    # I and K are α strings
    # J and L are β strings
    # c(IJ,s) <IJ|a'b'|KL> c(KL,t) = 
    # c(IJ,s) c(KL,t) <J|<I|a'b'|K>|L>
    # c(IJ,s) c(KL,t) <J|<I|a'|K>b'|L> (-1)^ket.ansatz.na
    # c(IJ,s) c(KL,t) <I|a'|K><J|b'|L> (-1)^ket.ansatz.na
    
    tbl1a, tbl1a_sign = generate_single_index_lookup(bra.ansatz, ket.ansatz, "alpha")
    tbl1b, tbl1b_sign = generate_single_index_lookup(bra.ansatz, ket.ansatz, "beta")
    
    bra_M = size(bra,2)
    ket_M = size(ket,2)

    v1 = reshape(bra.vectors, bra.ansatz.dima, bra.ansatz.dimb, bra_M)
    v2 = reshape(ket.vectors, ket.ansatz.dima, ket.ansatz.dimb, ket_M)
    
    v1 = permutedims(v1, [3,1,2])
    v2 = permutedims(v2, [3,1,2])
    
    #   TDM[pq,s,t] = 
    tdm = zeros(Float64, bra_M, ket_M,  bra.ansatz.no, bra.ansatz.no)
    
    sgnK = 1 
    if (ket.ansatz.na) % 2 != 0 
        sgnK = -sgnK
    end

    for K in 1:size(tbl1a,1)
        for p in 1:bra.ansatz.no
            I = tbl1a[K,p]
            I != 0 || continue
            Ksign = tbl1a_sign[K,p]
            for L in 1:size(tbl1b, 1)
                for q in 1:bra.ansatz.no
                    J = tbl1b[L,q]
                    J != 0 || continue
                    Lsign = tbl1b_sign[L,q]
                    
                    @views tdm_pqr = tdm[:,:,p,q] 
                    @views v1_IJ = v1[:,I,J]
                    @views v2_KL = v2[:,K,L]
                    sgn = Ksign*Lsign*sgnK
                    
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
    tdm = permutedims(tdm, [3,4,1,2])
    return tdm#=}}}=#
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
    bra.ansatz.na - 1 == ket.ansatz.na || throw(DimensionMismatch) #={{{=#
    bra.ansatz.nb == ket.ansatz.nb     || throw(DimensionMismatch) 
    
    # <s|p'q'r|t>
    # I and K are α strings
    # J and L are β strings
    # c(IJ,s) <IJ|a'a'a|KL> c(KL,t) = 
    # c(IJ,s) c(KL,t) <J|<I|a'a'a|K>|L>
    # c(IJ,s) c(KL,t) <J|L><I|a'a'a|K>   
    # c(IJ,s) c(KL,t) <J|L><I|a'|m><m|a'|m><m|a|K>
    
    ansatz_m1 = RASCIAnsatz(ket.ansatz.no, ket.ansatz.na-1, ket.ansatz.nb, ket.ansatz.fock)
    ansatz_m2 = RASCIAnsatz(ket.ansatz.no, ket.ansatz.na, ket.ansatz.nb, ket.ansatz.fock)
    
    tbl1a, tbl1a_sign = generate_single_index_lookup(ansatz_m1, ket.ansatz, "alpha")
    #println("α dim ", ket.ansatz.dima, " --> α dim ", bra.ansatz.dima)
    
    tbl2a, tbl2a_sign = generate_single_index_lookup(ansatz_m2, ansatz_m1, "alpha")
    #println("α dim ", ansatz_m1.dima, " --> α dim ", ansatz_m2.dima)
    
    tbl3a, tbl3a_sign = generate_single_index_lookup(bra.ansatz, ansatz_m2, "alpha")
    #println("α dim ", ansatz_m2.dima, " --> α dim ", bra.ansatz.dima)
    
    bra_M = size(bra,2)
    ket_M = size(ket,2)

    v1 = reshape(bra.vectors, bra.ansatz.dima, bra.ansatz.dimb, bra_M)
    v2 = reshape(ket.vectors, ket.ansatz.dima, ket.ansatz.dimb, ket_M)
    v1 = permutedims(v1, [2,3,1])
    v2 = permutedims(v2, [2,3,1])
    
    #   TDM[pqr,s,t] = 
    tdm = zeros(Float64, bra_M, ket_M,  bra.ansatz.no, bra.ansatz.no, bra.ansatz.no)
    
    for K in 1:size(tbl1a,1)
        for p in 1:bra.ansatz.no
            for q in 1:bra.ansatz.no
                for r in 1:bra.ansatz.no
                    rK = tbl1a[K,r]
                    rK != 0 || continue
                    Ksign = tbl1a_sign[K,r]
                    rqK = tbl2a[rK,q]
                    rqK != 0 || continue
                    Ksign = Ksign*tbl2a_sign[rK,q]

                    I = tbl3a[rqK,p]
                    I != 0 || continue
                    Ksign = Ksign*tbl3a_sign[rqK,p]

                    @views tdm_pqr = tdm[:,:,p,q,r] 
                    @views v1_IJ = v1[:,:,I]
                    @views v2_KL = v2[:,:,K]

                    if Ksign == 1
                        @tensor begin 
                            tdm_pqr[s,t] += v1_IJ[I,s] * v2_KL[I,t]
                        end
                    else
                        @tensor begin 
                            tdm_pqr[s,t] -= v1_IJ[I,s] * v2_KL[I,t]
                        end
                    end
                end
            end
        end
    end
    #                      [p,q,r,s,t]
    tdm = permutedims(tdm, [3,4,5,1,2])
    return tdm#=}}}=#
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
    bra.ansatz.na  == ket.ansatz.na || throw(DimensionMismatch) #={{{=#
    bra.ansatz.nb - 1 == ket.ansatz.nb  || throw(DimensionMismatch) 

    # <s|p'q'r|t>
    # I and K are α strings
    # J and L are β strings
    # c(IJ,s) <IJ|b'b'b|KL> c(KL,t) = 
    # c(IJ,s) c(KL,t) <J|<I|b'b'b|K>|L>
    # c(IJ,s) c(KL,t) <I|K><J|b'b'b|L> (-1)^ket.ansatz.na  
    # c(IJ,s) c(KL,t) <I|K><J|b'|m><m|b'|m><m|b|L>(-1)^ket.ansatz.na
    
    sgnK = 1 
    if (ket.ansatz.na) % 2 != 0 
        sgnK = -sgnK
    end
    
    ansatz_m1 = RASCIAnsatz(ket.ansatz.no, ket.ansatz.na, ket.ansatz.nb-1, ket.ansatz.fock)
    ansatz_m2 = RASCIAnsatz(ket.ansatz.no, ket.ansatz.na, ket.ansatz.nb, ket.ansatz.fock)
    
    tbl1b, tbl1b_sign = generate_single_index_lookup(ansatz_m1, ket.ansatz, "beta")
    #println("β dim ", ket.ansatz.dimb, " --> β dim ", ansatz_m1.dimb)
    
    tbl2b, tbl2b_sign = generate_single_index_lookup(ansatz_m2, ansatz_m1, "beta")
    #println("β dim ", ansatz_m1.dimb, " --> β dim ", ansatz_m2.dimb)
    
    tbl3b, tbl3b_sign = generate_single_index_lookup(bra.ansatz, ansatz_m2, "beta")
    #println("β dim ", ansatz_m2.dimb, " --> β dim ", bra.ansatz.dimb)
    
    bra_M = size(bra,2)
    ket_M = size(ket,2)

    v1 = reshape(bra.vectors, bra.ansatz.dima, bra.ansatz.dimb, bra_M)
    v2 = reshape(ket.vectors, ket.ansatz.dima, ket.ansatz.dimb, ket_M)
    #permuted like this since "beta" spin
    v1 = permutedims(v1, [1,3,2])
    v2 = permutedims(v2, [1,3,2])
    
    #
    #   TDM[pqr,s,t] = 
    tdm = zeros(Float64, bra_M, ket_M,  bra.ansatz.no, bra.ansatz.no, bra.ansatz.no)
    
    for L in 1:size(tbl1b,1)
        for p in 1:bra.ansatz.no
            for q in 1:bra.ansatz.no
                for r in 1:bra.ansatz.no
                    rL = tbl1b[L,r]
                    rL != 0 || continue
                    Lsign = tbl1b_sign[L,r]
                    rqL = tbl2b[rL,q]
                    rqL != 0 || continue
                    Lsign = Lsign*tbl2b_sign[rL,q]

                    I = tbl3b[rqL,p]
                    I != 0 || continue
                    Lsign = Lsign*tbl3b_sign[rqL,p]

                    @views tdm_pqr = tdm[:,:,p,q,r] 
                    @views v1_IJ = v1[:,:,I]
                    @views v2_KL = v2[:,:,L]
                    sgn = Lsign*sgnK

                    if sgn == 1
                        @tensor begin 
                            tdm_pqr[s,t] += v1_IJ[I,s] * v2_KL[I,t]
                        end
                    else
                        @tensor begin 
                            tdm_pqr[s,t] -= v1_IJ[I,s] * v2_KL[I,t]
                        end
                    end
                end
            end
        end
    end
    #                      [p,q,r,s,t]
    tdm = permutedims(tdm, [3,4,5,1,2])
    return tdm#=}}}=#
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
    bra.ansatz.na  == ket.ansatz.na || throw(DimensionMismatch) #={{{=#
    bra.ansatz.nb - 1 == ket.ansatz.nb  || throw(DimensionMismatch) 
    
    # <s|p'q'r|t>
    # I and K are α strings
    # J and L are β strings
    # c(IJ,s) <IJ|a'b'a|KL> c(KL,t) = 
    # c(IJ,s) c(KL,t) <J|<I|a'ab'|K>|L> (-1) 
    # c(IJ,s) c(KL,t) <I|a'a|K><J|b'|L> (-1) (-1)^ket.ansatz.na  
    # c(IJ,s) c(KL,t) \sum_m <I|a'|m><m|a|K><J|b'|L> 
    
    ansatz_m1 = RASCIAnsatz(ket.ansatz.no, ket.ansatz.na-1, ket.ansatz.nb, ket.ansatz.fock)
    
    tbl1a, tbl1a_sign = generate_single_index_lookup(ansatz_m1, ket.ansatz, "alpha")
    #println("α dim ", ket.ansatz.dima, " --> α dim ", bra.ansatz.dima)
    
    tbl2a, tbl2a_sign = generate_single_index_lookup(bra.ansatz, ansatz_m1, "alpha")
    #println("α dim ", ansatz_m1.dima, " --> α dim ", bra.ansatz.dima)
    
    tbl1b, tbl1b_sign = generate_single_index_lookup(bra.ansatz, ket.ansatz, "beta")
    #println("β dim ", ket.ansatz.dimb, " --> β dim ", bra.ansatz.dimb)
    
    bra_M = size(bra,2)
    ket_M = size(ket,2)

    v1 = reshape(bra.vectors, bra.ansatz.dima, bra.ansatz.dimb, bra_M)
    v2 = reshape(ket.vectors, ket.ansatz.dima, ket.ansatz.dimb, ket_M)
    v1 = permutedims(v1, [3,1,2])
    v2 = permutedims(v2, [3,1,2])
    
    sgnK = -1 # start from -1 to account for a'b'a -> a'ab'
    if (ket.ansatz.na) % 2 != 0 
        sgnK = -sgnK
    end
    
    #
    #   TDM[pqr,s,t] = 
    tdm = zeros(Float64, bra_M, ket_M,  bra.ansatz.no, bra.ansatz.no, bra.ansatz.no)
    
    for K in 1:size(tbl1a,1)
        for L in 1:size(tbl1b,1)
            for q in 1:bra.ansatz.no
                J = tbl1b[L,q]
                J != 0 || continue
                Lsign = tbl1b_sign[L,q]
                
                for p in 1:bra.ansatz.no
                    for r in 1:bra.ansatz.no
                        rK = tbl1a[K,r]
                        rK != 0 || continue
                        Ksign = tbl1a_sign[K,r]
                        I = tbl2a[rK,p]
                        I != 0 || continue
                        Ksign = Ksign*tbl2a_sign[rK,p]

                        @views tdm_pqr = tdm[:,:,p,q,r] 
                        @views v1_IJ = v1[:,I,J]
                        @views v2_KL = v2[:,K,L]
                        sgn = Ksign*Lsign*sgnK

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
    return tdm#=}}}=#
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
    bra.ansatz.na - 1 == ket.ansatz.na || throw(DimensionMismatch) #={{{=#
    bra.ansatz.nb == ket.ansatz.nb     || throw(DimensionMismatch) 
    
    # <s|p'q'r|t>
    # I and K are α strings
    # J and L are β strings
    # c(IJ,s) <IJ|a'b'b|KL> c(KL,t) = 
    # c(IJ,s) c(KL,t) <J|<I|a'b'b|K>|L> 
    # c(IJ,s) c(KL,t) <I|a'|K><J|b'b|L> 
    # c(IJ,s) c(KL,t) \sum_m <I|a'|K><J|b'|m><m|b|L> 
    
    ansatz_m1 = RASCIAnsatz(ket.ansatz.no, ket.ansatz.na, ket.ansatz.nb-1, ket.ansatz.fock)
    
    tbl1a, tbl1a_sign = generate_single_index_lookup(bra.ansatz, ket.ansatz, "alpha")
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
    
    for K in 1:size(tbl1a,1)
        for L in 1:size(tbl1b,1)
            for p in 1:bra.ansatz.no
                I = tbl1a[K,p]
                I != 0 || continue
                Ksign = tbl1a_sign[K,p]
                
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
    return tdm#=}}}=#
end
