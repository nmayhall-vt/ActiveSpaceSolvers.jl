using JLD2
using InCoreIntegrals

function sigma_one(prob::RASCIAnsatz, spin_pairs::Vector{Spin_Pair}, cats_a::Vector{HP_Category}, cats_b::Vector{HP_Category}, ints::InCoreInts, C)
    sigma_one = Dict{Int, Array{Float64,3}}()
    v = Dict{Int, Array{Float64, 3}}()
    n_spin_pairs = length(spin_pairs)
    nroots = size(C,2)
    F = zeros(Float64, prob.dimb)
    start = 1
    
    for m in 1:n_spin_pairs
        sigma_one[m] = zeros(length(cats_a[spin_pairs[m].pair[1]].idxs), length(cats_b[spin_pairs[m].pair[2]].idxs), nroots)
        tmp = C[start:start+spin_pairs[m].dim-1, :]
        v[m] = reshape(tmp, (length(cats_a[spin_pairs[m].pair[1]].idxs), length(cats_b[spin_pairs[m].pair[2]].idxs), nroots))
        start += spin_pairs[m].dim
    end
    
    gkl = get_gkl(ints, prob) 
    for Ib in 1:prob.dimb
        fill!(F,0.0)
        comb_kl = 0
        comb_ij = 0
        cat_Ib = find_cat(Ib, cats_b)
        pair_Ib = find_spin_pair(spin_pairs, cat_Ib.idx, "beta")
        Ib_local = Ib-spin_pairs[pair_Ib].bshift
        for k in 1:prob.no, l in 1:prob.no
            Kb = cat_Ib.lookup[l,k,Ib_local]
            Kb != 0 || continue
            sign_kl = sign(Kb)
            Kb = abs(Kb)
            @inbounds F[Kb] += sign_kl*gkl[k,l]
            catb_kl = find_cat(Kb, cats_b)
            comb_kl = (k-1)*prob.no + l
            pair_Kb = find_spin_pair(spin_pairs, catb_kl.idx, "beta") 
            Kb_local = Kb-spin_pairs[pair_Kb].bshift

            for i in 1:prob.no, j in 1:prob.no
                comb_ij = (i-1)*prob.no + j
                if comb_ij < comb_kl
                    continue
                end
                Jb = catb_kl.lookup[j,i,Kb_local]
                Jb != 0 || continue
                sign_ij = sign(Jb)
                Jb = abs(Jb)
                if comb_kl == comb_ij
                    delta = 1
                else
                    delta = 0
                end
                if sign_kl == sign_ij
                    F[Jb] += (ints.h2[i,j,k,l]*1/(1+delta))
                else
                    F[Jb] -= (ints.h2[i,j,k,l]*1/(1+delta))
                end
            end
        end
        _sum_spin_pairs!(sigma_one, v, F, Ib, cats_a, cats_b, spin_pairs)
    end
    
    start = 1
    sig = zeros(Float64, prob.ras_dim, nroots)
    for m in 1:n_spin_pairs
        tmp = reshape(sigma_one[m], (size(sigma_one[m],1)*size(sigma_one[m],2), nroots))
        sig[start:start+spin_pairs[m].dim-1, :] .= tmp
        start += spin_pairs[m].dim
    end
    return sig
end

function sigma_two(prob::RASCIAnsatz, spin_pairs::Vector{Spin_Pair}, cats_a::Vector{HP_Category}, cats_b::Vector{HP_Category}, ints::InCoreInts, C)
    sigma_two = Dict{Int, Array{Float64,3}}()
    v = Dict{Int, Array{Float64, 3}}()
    n_spin_pairs = length(spin_pairs)
    nroots = size(C,2)
    F = zeros(Float64, prob.dima)
    start = 1

    for m in 1:n_spin_pairs
        sigma_two[m] = zeros(length(cats_a[spin_pairs[m].pair[1]].idxs), length(cats_b[spin_pairs[m].pair[2]].idxs), nroots)
        tmp = C[start:start+spin_pairs[m].dim-1, :]
        v[m] = reshape(tmp, (length(cats_a[spin_pairs[m].pair[1]].idxs), length(cats_b[spin_pairs[m].pair[2]].idxs), nroots))
        start += spin_pairs[m].dim
    end

    gkl = get_gkl(ints, prob) 
    
    for Ia in 1:prob.dima
        fill!(F,0.0)
        comb_kl = 0
        comb_ij = 0
        cat_Ia = find_cat(Ia, cats_a)
        pair_Ia = find_spin_pair(spin_pairs, cat_Ia.idx, "alpha")
        Ia_local = Ia-spin_pairs[pair_Ia].ashift
        for k in 1:prob.no, l in 1:prob.no
            Ka = cat_Ia.lookup[l,k,Ia_local]
            Ka != 0 || continue
            sign_kl = sign(Ka)
            Ka = abs(Ka)
            @inbounds F[Ka] += sign_kl*gkl[k,l]
            cata_kl = find_cat(Ka, cats_a)
            comb_kl = (k-1)*prob.no + l
            pair_Ka = find_spin_pair(spin_pairs, cata_kl.idx, "alpha") 
            Ka_local = Ka-spin_pairs[pair_Ka].ashift

            for i in 1:prob.no, j in 1:prob.no
                comb_ij = (i-1)*prob.no + j
                if comb_ij < comb_kl
                    continue
                end
                Ja = cata_kl.lookup[j,i,Ka_local]
                Ja != 0 || continue
                sign_ij = sign(Ja)
                Ja = abs(Ja)
                if comb_kl == comb_ij
                    delta = 1
                else
                    delta = 0
                end
                if sign_kl == sign_ij
                    F[Ja] += (ints.h2[i,j,k,l]*1/(1+delta))
                else
                    F[Ja] -= (ints.h2[i,j,k,l]*1/(1+delta))
                end
            end
        end
        _sum_spin_pairs!(sigma_two, v, F, Ia, cats_a, cats_b, spin_pairs, sigma="two")
    end

    starti = 1
    sig = zeros(Float64, prob.ras_dim, nroots)
    for m in 1:n_spin_pairs
        tmp = reshape(sigma_two[m], (size(sigma_two[m],1)*size(sigma_two[m],2), nroots))
        sig[starti:starti+spin_pairs[m].dim-1, :] .= tmp
        starti += spin_pairs[m].dim
    end
    return sig
end

function sigma_three(prob::RASCIAnsatz, spin_pairs::Vector{Spin_Pair}, cats_a::Vector{HP_Category}, cats_b::Vector{HP_Category}, ints::InCoreInts, C)
    sigma_three = Dict{Int, Array{Float64,3}}()
    v = Dict{Int, Array{Float64, 3}}()
    n_spin_pairs = length(spin_pairs)
    nroots = size(C,2)
    
    hkl = zeros(Float64, prob.no, prob.no)
    
    start = 1

    for m in 1:n_spin_pairs
        sigma_three[m] = zeros(length(cats_a[spin_pairs[m].pair[1]].idxs), length(cats_b[spin_pairs[m].pair[2]].idxs), nroots)
        tmp = C[start:start+spin_pairs[m].dim-1, :]
        v[m] = reshape(tmp, (length(cats_a[spin_pairs[m].pair[1]].idxs), length(cats_b[spin_pairs[m].pair[2]].idxs), nroots))
        start += spin_pairs[m].dim
    end
    
    for Ia in 1:prob.dima
        cat_Ia = find_cat(Ia, cats_a)
        pair_Ia = find_spin_pair(spin_pairs, cat_Ia.idx, "alpha")
        Ia_local = Ia-spin_pairs[pair_Ia].ashift
        for k in 1:prob.no, l in 1:prob.no
            Ja = cat_Ia.lookup[l,k,Ia_local]
            Ja != 0 || continue
            sign_kl = sign(Ja)
            Ja = abs(Ja)
            hkl .= ints.h2[:,:,k,l]
            cata_kl = find_cat(Ja, cats_a)
            pair_Ja = find_spin_pair(spin_pairs, cata_kl.idx, "alpha") 
            Ja_local = Ja-spin_pairs[pair_Ja].ashift
            for Ib in 1:prob.dimb
                cat_Ib = find_cat(Ib, cats_b)
                pair_Ib = find_spin_pair(spin_pairs, cat_Ib.idx, "beta")
                Ib_local = Ib-spin_pairs[pair_Ib].bshift
                for i in 1:prob.no, j in 1:prob.no
                    Jb = cat_Ib.lookup[j,i,Ib_local]
                    Jb != 0 || continue
                    sign_ij = sign(Jb)
                    Jb = abs(Jb)
                    catb_ij = find_cat(Jb, cats_b)
                    pair_Jb = find_spin_pair(spin_pairs, catb_ij.idx, "beta") 
                    Jb_local = Jb-spin_pairs[pair_Jb].bshift
                    if cat_Ib.idx in cat_Ia.connected
                        if catb_ij.idx in cata_kl.connected
                            m = find_spin_pair(spin_pairs, (cat_Ia.idx, cat_Ib.idx))
                            n = find_spin_pair(spin_pairs, (cata_kl.idx, catb_ij.idx))
                            for si in 1:nroots
                                sigma_three[m][Ia_local, Ib_local, si] += hkl[i,j]*v[n][Ja_local, Jb_local, si]*sign_ij*sign_kl
                            end
                        end
                    end
                end
            end
        end
    end
    
    start = 1
    sig = zeros(Float64, prob.ras_dim, nroots)
    for m in 1:n_spin_pairs
        tmp = reshape(sigma_three[m], (size(sigma_three[m],1)*size(sigma_three[m],2), nroots))
        sig[start:start+spin_pairs[m].dim-1, :] .= tmp
        start += spin_pairs[m].dim
    end
    return sig
end


function _sum_spin_pairs!(sig::Dict{Int, Array{T, 3}}, v::Dict{Int,Array{T,3}}, F::Vector{T}, I::Int, cats_a::Vector{HP_Category}, cats_b::Vector{HP_Category}, spin_pairs::Vector{Spin_Pair}; sigma="one") where {T}
    n_roots = size(v[1],3)

    if sigma == "one"
        current_cat = find_cat(I, cats_b)
        for catb in cats_b    
            for cats in catb.connected
                for Ia in cats_a[cats].idxs
                    n = find_spin_pair(spin_pairs, (cats_a[cats].idx, catb.idx))
                    Ia_v_local = Ia-spin_pairs[n].ashift
                    for Jb in catb.idxs
                        Jb_local = Jb-spin_pairs[n].bshift
                        if cats_a[cats].idx in current_cat.connected
                            m = find_spin_pair(spin_pairs, (cats_a[cats].idx, current_cat.idx))
                            Ia_local = Ia-spin_pairs[m].ashift
                            Ib_local = I-spin_pairs[m].bshift
                            @inbounds @simd for si in 1:n_roots
                                sig[m][Ia_local,Ib_local,si] += F[Jb]*v[n][Ia_v_local,Jb_local, si]
                            end
                        end
                    end
                end
            end
        end
    else
        current_cat = find_cat(I, cats_a)
        for cata in cats_a
            for cats in cata.connected
                for Ib in cats_b[cats].idxs
                    n = find_spin_pair(spin_pairs, (cata.idx, cats_b[cats].idx))
                    Ib_v_local = Ib-spin_pairs[n].bshift
                    for Ja in cata.idxs
                        Ja_local = Ja-spin_pairs[n].ashift
                        if cats_b[cats].idx in current_cat.connected
                            m = find_spin_pair(spin_pairs, (current_cat.idx, cats_b[cats].idx))
                            Ia_local = I-spin_pairs[m].ashift
                            Ib_local = Ib-spin_pairs[m].bshift
                            @inbounds @simd for si in 1:n_roots
                                sig[m][Ia_local,Ib_local,si] += F[Ja]*v[n][Ja_local, Ib_v_local,si]
                            end
                        end
                    end
                end
            end
        end
    end
end










    

