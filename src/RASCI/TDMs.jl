using LinearAlgebra
using Printf
using NPZ
using StaticArrays
using JLD2
using BenchmarkTools
using LinearMaps
using TensorOperations

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

    bra.ansatz.na-1 == ket.ansatz.na     || throw(DimensionMismatch)#={{{=#
    bra.ansatz.nb == ket.ansatz.nb     || throw(DimensionMismatch) 
    
    bra_M = size(bra,2)
    ket_M = size(ket,2)

    cats_a_bra, cats_a = ActiveSpaceSolvers.RASCI.fill_lu_HP(bra.ansatz, ket.ansatz, spin="alpha", type="c")
    cats_b_bra, cats_b = ActiveSpaceSolvers.RASCI.fill_lu_HP(bra.ansatz, ket.ansatz, spin="beta", type="no_op")
    spin_pairs = ActiveSpaceSolvers.RASCI.make_spin_pairs(ket.ansatz, cats_a, cats_b)
    spin_pairs_bra = ActiveSpaceSolvers.RASCI.make_spin_pairs(bra.ansatz, cats_a_bra, cats_b_bra)
    
    v1 = Dict{Int, Array{Float64, 3}}()
    start = 1
    for m in 1:length(spin_pairs_bra)
        tmp = bra.vectors[start:start+spin_pairs_bra[m].dim-1, :]
        v1[m] = reshape(tmp, (length(cats_a_bra[spin_pairs_bra[m].pair[1]].idxs), length(cats_b_bra[spin_pairs_bra[m].pair[2]].idxs), bra_M))
        start += spin_pairs_bra[m].dim
    end
    
    v2 = Dict{Int, Array{Float64, 3}}()
    start = 1
    for m in 1:length(spin_pairs)
        tmp = ket.vectors[start:start+spin_pairs[m].dim-1, :]
        v2[m] = reshape(tmp, (length(cats_a[spin_pairs[m].pair[1]].idxs), length(cats_b[spin_pairs[m].pair[2]].idxs), ket_M))
        start += spin_pairs[m].dim
    end
    
    #   TDM[p,s,t] = 
    tdm = zeros(Float64, bra_M, ket_M,  bra.ansatz.no)

    for m in 1:length(spin_pairs)
        cat_Ia = cats_a[spin_pairs[m].pair[1]]
        for Ia in cats_a[spin_pairs[m].pair[1]].idxs
            Ia_local = Ia-spin_pairs[m].ashift
            for p in 1:ket.ansatz.no
                Ja = cat_Ia.lookup[p,Ia_local]
                Ja != 0 || continue
                sign_p = sign(Ja)
                Ja = abs(Ja)
                cata_kl = find_cat(Ja, cats_a_bra)
                n = find_spin_pair(spin_pairs_bra, (cata_kl.idx, spin_pairs[m].pair[2]))
                n != 0 || continue
                Ja_local = Ja-spin_pairs_bra[n].ashift
                @views tdm_pqr = tdm[:,:,p]
                @views v1_IJ = v1[n][Ja_local, :, :]
                @views v2_KL = v2[m][Ia_local, :, :]
                if sign_p == 1
                    @tensor begin
                        tdm_pqr[s,t] += v1_IJ[K,s] * v2_KL[K,t]
                    end
                else
                    @tensor begin 
                        tdm_pqr[s,t] -= v1_IJ[K,s] * v2_KL[K,t]
                    end
                end
            end
        end
    end
    tdm = permutedims(tdm, [3,1,2])
    return tdm
end#=}}}=#

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
    
    bra_M = size(bra,2)
    ket_M = size(ket,2)

    cats_a_bra, cats_a = ActiveSpaceSolvers.RASCI.fill_lu_HP(bra.ansatz, ket.ansatz, spin="alpha", type="c")
    cats_b_bra, cats_b = ActiveSpaceSolvers.RASCI.fill_lu_HP(bra.ansatz, ket.ansatz, spin="beta", type="a")
    spin_pairs = ActiveSpaceSolvers.RASCI.make_spin_pairs(ket.ansatz, cats_a, cats_b)
    spin_pairs_bra = ActiveSpaceSolvers.RASCI.make_spin_pairs(bra.ansatz, cats_a_bra, cats_b_bra)
    
    v1 = Dict{Int, Array{Float64, 3}}()
    start = 1
    for m in 1:length(spin_pairs_bra)
        tmp = bra.vectors[start:start+spin_pairs_bra[m].dim-1, :]
        v1[m] = reshape(tmp, (length(cats_a_bra[spin_pairs_bra[m].pair[1]].idxs), length(cats_b_bra[spin_pairs_bra[m].pair[2]].idxs), bra_M))
        start += spin_pairs_bra[m].dim
    end
    
    v2 = Dict{Int, Array{Float64, 3}}()
    start = 1
    for m in 1:length(spin_pairs)
        tmp = ket.vectors[start:start+spin_pairs[m].dim-1, :]
        v2[m] = reshape(tmp, (length(cats_a[spin_pairs[m].pair[1]].idxs), length(cats_b[spin_pairs[m].pair[2]].idxs), ket_M))
        start += spin_pairs[m].dim
    end
    
    
    #   TDM[pq,s,t] = 
    tdm = zeros(Float64, bra_M, ket_M,  bra.ansatz.no, bra.ansatz.no)
    
    sgnK = 1 
    if (ket.ansatz.na) % 2 != 0 
        sgnK = -sgnK
    end
    
    for m in 1:length(spin_pairs)
        cat_Ia = cats_a[spin_pairs[m].pair[1]]
        cat_Ib = cats_b[spin_pairs[m].pair[2]]
        for Ia in cats_a_ca[spin_pairs[m].pair[1]].idxs
            Ia_local = Ia-spin_pairs[m].ashift
            for p in 1:prob.no
                Ja = cat_Ia.lookup[p,Ia_local]
                Ja != 0 || continue
                sign_p = sign(Ja)
                Ja = abs(Ja)
                cata_p = find_cat(Ja, cats_a_bra)
                for Ib in cats_b_ca[spin_pairs[m].pair[2]].idxs
                    Ib_local = Ib-spin_pairs[m].bshift
                    for q in 1:prob.no
                        Jb = cat_Ib.lookup[q,Ib_local]
                        Jb != 0 || continue
                        sign_q = sign(Jb)
                        Jb = abs(Jb)
                        catb_q = find_cat(Jb, cats_b_bra)
                        n = find_spin_pair(spin_pairs_bra, (cata_p.idx, catb_q.idx))
                        n != 0 || continue
                        Ja_local = Ja-spin_pairs_bra[n].ashift
                        Jb_local = Jb-spin_pairs_bra[n].bshift
                        @views tdm_pqr = tdm[:,:,p,q] 
                        @views v1_IJ = v1[n][Ja_local, Jb_local, :]
                        @views v2_KL = v2[m][Ia_local, Ib_local, :]
                        sgn = sign_p*sign_q*sgnK
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
    #                      [p,q,s,t]
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
    
    cats_a_bra, cats_a = ActiveSpaceSolvers.RASCI.fill_lu_HP(bra.ansatz, ket.ansatz, spin="alpha", type="cca")
    cats_b_bra, cats_b = ActiveSpaceSolvers.RASCI.fill_lu_HP(bra.ansatz, ket.ansatz, spin="beta", type="no_op")
    spin_pairs = ActiveSpaceSolvers.RASCI.make_spin_pairs(ket.ansatz, cats_a, cats_b)
    spin_pairs_bra = ActiveSpaceSolvers.RASCI.make_spin_pairs(bra.ansatz, cats_a_bra, cats_b_bra)
    
    v1 = Dict{Int, Array{Float64, 3}}()
    start = 1
    for m in 1:length(spin_pairs_bra)
        tmp = bra.vectors[start:start+spin_pairs_bra[m].dim-1, :]
        v1[m] = reshape(tmp, (length(cats_a_bra[spin_pairs_bra[m].pair[1]].idxs), length(cats_b_bra[spin_pairs_bra[m].pair[2]].idxs), bra_M))
        start += spin_pairs_bra[m].dim
    end
    
    v2 = Dict{Int, Array{Float64, 3}}()
    start = 1
    for m in 1:length(spin_pairs)
        tmp = ket.vectors[start:start+spin_pairs[m].dim-1, :]
        v2[m] = reshape(tmp, (length(cats_a[spin_pairs[m].pair[1]].idxs), length(cats_b[spin_pairs[m].pair[2]].idxs), ket_M))
        start += spin_pairs[m].dim
    end
    
    
    bra_M = size(bra,2)
    ket_M = size(ket,2)

    
    #   TDM[pqr,s,t] = 
    tdm = zeros(Float64, bra_M, ket_M,  bra.ansatz.no, bra.ansatz.no, bra.ansatz.no)
    
    for m in 1:length(spin_pairs)
        cat_Ia = cats_a[spin_pairs[m].pair[1]]
        for Ia in cats_a[spin_pairs[m].pair[1]].idxs
            Ia_local = Ia-spin_pairs[m].ashift
            for p in 1:ket.ansatz.no, q in 1:ket.ansatz.no, r in 1:ket.ansatz.no
                Ja = cat_Ia.lookup[r,q,p,Ia_local]
                Ja != 0 || continue
                sign_pqr = sign(Ja)
                Ja = abs(Ja)
                cata_kl = find_cat(Ja, cats_a_bra)
                n = find_spin_pair(spin_pairs_bra, (cata_kl.idx, spin_pairs[m].pair[2]))
                n != 0 || continue
                Ja_local = Ja-spin_pairs_bra[n].ashift
                @views tdm_pqr = tdm[:,:,p,q,r] 
                @views v1_IJ = v1[Ia_local,:,:]
                @views v2_KL = v2[Ja_local,:,:]

                if sign_pqr == 1
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
    #                      [p,q,r,s,t]
    tdm = permutedims(tdm, [3,4,5,1,2])
    return tdm#=}}}=#
end
    




