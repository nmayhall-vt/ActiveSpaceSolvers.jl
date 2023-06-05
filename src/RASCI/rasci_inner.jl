using JLD2
using InCoreIntegrals

function sigma_one(prob::RASCIAnsatz, spin_pairs::Vector{Spin_Pair}, cats_a::Vector{HP_Category_CA}, cats_b::Vector{HP_Category_CA}, ints::InCoreInts, C)
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

function sigma_two(prob::RASCIAnsatz, spin_pairs::Vector{Spin_Pair}, cats_a::Vector{HP_Category_CA}, cats_b::Vector{HP_Category_CA}, ints::InCoreInts, C)
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

function sigma_three(prob::RASCIAnsatz, spin_pairs::Vector{Spin_Pair}, cats_a::Vector{HP_Category_CA}, cats_b::Vector{HP_Category_CA}, ints::InCoreInts, C)
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


function _sum_spin_pairs!(sig::Dict{Int, Array{T, 3}}, v::Dict{Int,Array{T,3}}, F::Vector{T}, I::Int, cats_a::Vector{HP_Category_CA}, cats_b::Vector{HP_Category_CA}, spin_pairs::Vector{Spin_Pair}; sigma="one") where {T}
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

function ras_compute_1rdm(prob::RASCIAnsatz, C::Vector)
    spin_pairs, cats_a, cats_b = ActiveSpaceSolvers.RASCI.make_spin_pairs(prob)
    rdm1a = zeros(prob.no, prob.no)
    rdm1b = zeros(prob.no, prob.no)
    v = Dict{Int, Array{Float64, 2}}()
    
    start = 1
    for m in 1:length(spin_pairs)
        tmp = C[start:start+spin_pairs[m].dim-1]
        v[m] = reshape(tmp, (length(cats_a[spin_pairs[m].pair[1]].idxs), length(cats_b[spin_pairs[m].pair[2]].idxs)))
        start += spin_pairs[m].dim
    end

    ##alpha l'k
    for m in 1:length(spin_pairs)
        cat_Ia = cats_a[spin_pairs[m].pair[1]]
        for Ia in cats_a[spin_pairs[m].pair[1]].idxs
            Ia_local = Ia-spin_pairs[m].ashift
            for k in 1:prob.no, l in 1:prob.no
                Ja = cat_Ia.lookup[l,k,Ia_local]
                Ja != 0 || continue
                sign_kl = sign(Ja)
                Ja = abs(Ja)
                cata_kl = find_cat(Ja, cats_a)
                n = find_spin_pair(spin_pairs, (cata_kl.idx, spin_pairs[m].pair[2]))
                n != 0 || continue
                Ja_local = Ja-spin_pairs[n].ashift
                rdm1a[l,k] += sign_kl*dot(v[n][Ja_local,:], v[m][Ia_local,:])
            end
        end
    end
    
    ##beta l'k
    for m in 1:length(spin_pairs)
        cat_Ib = cats_b[spin_pairs[m].pair[2]]
        for Ib in cats_b[spin_pairs[m].pair[2]].idxs
            Ib_local = Ib-spin_pairs[m].bshift
            for k in 1:prob.no, l in 1:prob.no
                Jb = cat_Ib.lookup[l,k,Ib_local]
                Jb != 0 || continue
                sign_kl = sign(Jb)
                Jb = abs(Jb)
                catb_kl = find_cat(Jb, cats_b)
                n = find_spin_pair(spin_pairs, (spin_pairs[m].pair[1], catb_kl.idx))
                n != 0 || continue
                Jb_local = Jb-spin_pairs[n].bshift
                rdm1b[l,k] += sign_kl*dot(v[n][:, Jb_local], v[m][:, Ib_local])
            end
        end
    end

    return rdm1a, rdm1b
end

function ras_compute_1rdm_2rdm(prob::RASCIAnsatz, C::Vector)
    cats_a_bra, cats_a = ActiveSpaceSolvers.RASCI.fill_lu_HP(prob, prob, spin="alpha", type="ccaa")
    cats_b_bra, cats_b = ActiveSpaceSolvers.RASCI.fill_lu_HP(prob, prob, spin="beta", type="ccaa")
    #need to make spin_pairs for ccaa!!!!

    spin_pairs_ca, cats_a_ca, cats_b_ca = ActiveSpaceSolvers.RASCI.make_spin_pairs(prob)

    rdm1a, rdm1b = ras_compute_1rdm(prob, C)
    rdm2aa = zeros(prob.no, prob.no, prob.no, prob.no)
    rdm2bb = zeros(prob.no, prob.no, prob.no, prob.no)
    rdm2ab = zeros(prob.no, prob.no, prob.no, prob.no)
    
    v = Dict{Int, Array{Float64, 2}}()
    
    start = 1
    for m in 1:length(spin_pairs)
        tmp = C[start:start+spin_pairs[m].dim-1]
        v[m] = reshape(tmp, (length(cats_a[spin_pairs[m].pair[1]].idxs), length(cats_b[spin_pairs[m].pair[2]].idxs)))
        start += spin_pairs[m].dim
    end

    ## alpha alpha p'r'sq
    for m in 1:length(spin_pairs)
        cat_Ia = cats_a[spin_pairs[m].pair[1]]
        for Ia in cats_a[spin_pairs[m].pair[1]].idxs
            Ia_local = Ia-spin_pairs[m].ashift
            for p in 1:prob.no, q in 1:prob.no, r in 1:prob.no, s in 1:prob.no
                Ja = cat_Ia.lookup[p,q,r,s,Ia_local]
                Ja != 0 || continue
                sign_kl = sign(Ja)
                Ja = abs(Ja)
                cata_kl = find_cat(Ja, cats_a)
                n = find_spin_pair(spin_pairs, (cata_kl.idx, spin_pairs[m].pair[2]))
                n != 0 || continue
                Ja_local = Ja-spin_pairs[n].ashift
                rdm2aa[p,q,r,s] += sign_kl*dot(v[n][Ja_local,:], v[m][Ia_local,:])
            end
        end
    end
end

function fill_lu_HP(bra::RASCIAnsatz, ket::RASCIAnsatz; spin="alpha", type="c")
    categories = ActiveSpaceSolvers.RASCI.generate_spin_categories(ket)#={{{=#
    all_cats = Vector{HP_Category}()
    all_cats_bra = Vector{ActiveSpaceSolvers.RASCI.HP_Category_Bra}()
    cats_a = deepcopy(categories)
    cats_b = deepcopy(categories)
    fock_list_a, del_at_a = make_fock_from_categories(categories, ket, "alpha")
    deleteat!(cats_a, del_at_a)
    len_cat_a = length(cats_a)
        
    fock_list_b, del_at_b = make_fock_from_categories(categories, ket, "beta")
    deleteat!(cats_b, del_at_b)
    len_cat_b = length(cats_b)

    if spin == "alpha"
        connected = make_spincategory_connections(cats_a, cats_b, ket)
        #compute configs
        as = compute_config_dict(fock_list_a, ket, "alpha")
        #this reverses the config dictionary to get the index as the key 
        rev_as = Dict(value => key for (key, value) in as)
        max_a = length(as)

        #calculate bra config dictonary
        categories_bra = ActiveSpaceSolvers.RASCI.generate_spin_categories(bra)
        cats_a_bra = deepcopy(categories_bra)
        fock_list_a_bra, del_at_a_bra = make_fock_from_categories(categories_bra, bra, "alpha")
        deleteat!(cats_a_bra, del_at_a_bra)
        
        cats_b_bra = deepcopy(categories_bra)
        fock_list_b_bra, del_at_b_bra = make_fock_from_categories(categories_bra, bra, "beta")
        deleteat!(cats_b_bra, del_at_b_bra)
        
        connected_bra = make_spincategory_connections(cats_a_bra, cats_b_bra, bra)
        as_bra = compute_config_dict(fock_list_a_bra, bra, "alpha")
        #this reverses the config dictionary to get the index as the key 
        rev_as_bra = Dict(value => key for (key, value) in as_bra)
        max_a_bra = length(as_bra)

        if type == "c"
            #HP_Category_C
            for j in 1:len_cat_a
                idxas = Vector{Int}()
                graph_a = make_cat_graphs(fock_list_a[j], ket)
                idxas = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_a, 1, graph_a.max, idxas, rev_as) 
                sort!(idxas)
                lu = zeros(Int, graph_a.no, length(idxas))
                push!(all_cats, HP_Category_C(j, cats_a[j], connected[j], idxas, lu))
            end
            
            for m in 1:length(cats_a_bra)
                idxas_bra = Vector{Int}()
                graph_a_bra = make_cat_graphs(fock_list_a_bra[m], bra)
                idxas_bra = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_a_bra, 1, graph_a_bra.max, idxas_bra, rev_as_bra) 
                sort!(idxas_bra)
                push!(all_cats_bra, HP_Category_Bra(m, connected_bra[m], idxas_bra))
            end
            
            for k in 1:len_cat_a
                graph_a = make_cat_graphs(fock_list_a[k], ket)
                lu = ActiveSpaceSolvers.RASCI.dfs_c(graph_a, 1, graph_a.max, all_cats[k].lookup, all_cats, all_cats_bra, rev_as, rev_as_bra)
                #all_cats[k].lookup .= lu
            end

            return all_cats_bra, all_cats        
        end

        if type == "a"
            #HP_Category_A
            for j in 1:len_cat_a
                idxas = Vector{Int}()
                graph_a = make_cat_graphs(fock_list_a[j], ket)
                idxas = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_a, 1, graph_a.max, idxas, rev_as) 
                sort!(idxas)
                lu = zeros(Int, graph_a.no,length(idxas))
                push!(all_cats, HP_Category_A(j, cats_a[j], connected[j], idxas, lu))
            end

            
            for m in 1:length(cats_a_bra)
                idxas_bra = Vector{Int}()
                graph_a_bra = make_cat_graphs(fock_list_a_bra[m], bra)
                idxas_bra = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_a_bra, 1, graph_a_bra.max, idxas_bra, rev_as_bra) 
                sort!(idxas_bra)
                push!(all_cats_bra, HP_Category_Bra(m, connected_bra[m], idxas_bra))
            end
            
            for k in 1:len_cat_a
                graph_a = make_cat_graphs(fock_list_a[k], ket)
                lu = ActiveSpaceSolvers.RASCI.dfs_a(graph_a, 1, graph_a.max, all_cats[k].lookup, all_cats, all_cats_bra, rev_as, rev_as_bra)
                #all_cats[k].lookup .= lu
            end
            
            return all_cats_bra, all_cats        
        end

        if type == "ca"
            #HP_Category_CA
            for j in 1:len_cat_a
                idxas = Vector{Int}()
                graph_a = make_cat_graphs(fock_list_a[j], ket)
                idxas = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_a, 1, graph_a.max, idxas, rev_as) 
                sort!(idxas)
                lu = zeros(Int, graph_a.no, graph_a.no, length(idxas))
                push!(all_cats, HP_Category_CA(j, cats_a[j], connected[j], idxas, lu))
            end

            #for k in 1:len_cat_a
            #    graph_a = make_cat_graphs(fock_list_a[k], ket)
            #    lu = ActiveSpaceSolvers.RASCI.dfs_single_excitation(graph_a, 1, graph_a.max, all_cats[k].lookup, all_cats, rev_as)
            #end
            
            for m in 1:length(cats_a_bra)
                idxas_bra = Vector{Int}()
                graph_a_bra = make_cat_graphs(fock_list_a_bra[m], bra)
                idxas_bra = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_a_bra, 1, graph_a_bra.max, idxas_bra, rev_as_bra) 
                sort!(idxas_bra)
                push!(all_cats_bra, HP_Category_Bra(m, connected_bra[m], idxas_bra))
            end
            return all_cats_bra, all_cats        
        end

        if type == "cc"
            #HP_Category_CC
            for j in 1:len_cat_a
                idxas = Vector{Int}()
                graph_a = make_cat_graphs(fock_list_a[j], ket)
                idxas = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_a, 1, graph_a.max, idxas, rev_as) 
                sort!(idxas)
                lu = zeros(Int, graph_a.no, graph_a.no, length(idxas))
                push!(all_cats, HP_Category_CC(j, cats_a[j], connected[j], idxas, lu))
            end
            
            for m in 1:length(cats_a_bra)
                idxas_bra = Vector{Int}()
                graph_a_bra = make_cat_graphs(fock_list_a_bra[m], bra)
                idxas_bra = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_a_bra, 1, graph_a_bra.max, idxas_bra, rev_as_bra) 
                sort!(idxas_bra)
                push!(all_cats_bra, HP_Category_Bra(m, connected_bra[m], idxas_bra))
            end
            for k in 1:len_cat_a
                graph_a = make_cat_graphs(fock_list_a[k], ket)
                lu = ActiveSpaceSolvers.RASCI.dfs_cc(graph_a, 1, graph_a.max, all_cats[k].lookup, all_cats, all_cats_bra, rev_as, rev_as_bra)
                #all_cats[k].lookup .= lu
            end
        
            return all_cats_bra, all_cats        
        end

        if type == "cca"
            #HP_Category_CCA
            for j in 1:len_cat_a
                idxas = Vector{Int}()
                graph_a = make_cat_graphs(fock_list_a[j], ket)
                idxas = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_a, 1, graph_a.max, idxas, rev_as) 
                sort!(idxas)
                lu = zeros(Int, graph_a.no, graph_a.no, graph_a.no, length(idxas))
                push!(all_cats, HP_Category_CCA(j, cats_a[j], connected[j], idxas, lu))
            end

            for m in 1:length(cats_a_bra)
                idxas_bra = Vector{Int}()
                graph_a_bra = make_cat_graphs(fock_list_a_bra[m], bra)
                idxas_bra = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_a_bra, 1, graph_a_bra.max, idxas_bra, rev_as_bra) 
                sort!(idxas_bra)
                push!(all_cats_bra, HP_Category_Bra(m, connected_bra[m], idxas_bra))
            end
            
            for k in 1:len_cat_a
                graph_a = make_cat_graphs(fock_list_a[k], ket)
                lu = ActiveSpaceSolvers.RASCI.dfs_cca(graph_a, 1, graph_a.max, all_cats[k].lookup, all_cats, all_cats_bra, rev_as, rev_as_bra)
                #all_cats[k].lookup .= lu
            end

            return all_cats_bra, all_cats        
        end

        if type == "ccaa"
            #HP_Category_CCAA
            for j in 1:len_cat_a
                idxas = Vector{Int}()
                graph_a = make_cat_graphs(fock_list_a[j], ket)
                idxas = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_a, 1, graph_a.max, idxas, rev_as) 
                sort!(idxas)
                lu = zeros(Int, graph_a.no, graph_a.no, graph_a.no, graph_a.no, length(idxas))
                push!(all_cats, HP_Category_CCAA(j, cats_a[j], connected[j], idxas, lu))
            end

            for m in 1:length(cats_a_bra)
                idxas_bra = Vector{Int}()
                graph_a_bra = make_cat_graphs(fock_list_a_bra[m], bra)
                idxas_bra = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_a_bra, 1, graph_a_bra.max, idxas_bra, rev_as_bra) 
                sort!(idxas_bra)
                push!(all_cats_bra, HP_Category_Bra(m, connected_bra[m], idxas_bra))
            end
            
            for k in 1:len_cat_a
                graph_a = make_cat_graphs(fock_list_a[k], ket)
                lu = ActiveSpaceSolvers.RASCI.dfs_ccaa(graph_a, 1, graph_a.max, all_cats[k].lookup, all_cats, all_cats_bra, rev_as, rev_as_bra)
            end
            return all_cats_bra, all_cats        
        end
    else
        connected = make_spincategory_connections(cats_b, cats_a, ket)
        #compute configs
        bs = compute_config_dict(fock_list_b, ket, "beta")
        rev_bs = Dict(value => key for (key, value) in bs)
        max_b = length(bs)
        
        #calculate bra config dictonary
        categories_bra = ActiveSpaceSolvers.RASCI.generate_spin_categories(bra)
        cats_a_bra = deepcopy(categories_bra)
        fock_list_a_bra, del_at_a_bra = make_fock_from_categories(categories_bra, bra, "alpha")
        deleteat!(cats_a_bra, del_at_a_bra)
        
        cats_b_bra = deepcopy(categories_bra)
        fock_list_b_bra, del_at_b_bra = make_fock_from_categories(categories_bra, bra, "beta")
        deleteat!(cats_b_bra, del_at_b_bra)
        
        connected_bra = make_spincategory_connections(cats_b_bra, cats_a_bra, bra)
        bs_bra = compute_config_dict(fock_list_b_bra, bra, "beta")
        #this reverses the config dictionary to get the index as the key 
        rev_bs_bra = Dict(value => key for (key, value) in bs_bra)
        max_b_bra = length(bs_bra)
        
        if type == "c"
            #HP_Category_C
            for j in 1:len_cat_b
                idxbs = Vector{Int}()
                graph_b = make_cat_graphs(fock_list_b[j], ket)
                idxbs = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_b, 1, graph_b.max, idxbs, rev_bs) 
                sort!(idxbs)
                lu = zeros(Int, graph_b.no, length(idxbs))
                push!(all_cats, HP_Category_C(j, cats_b[j], connected[j], idxbs, lu))
            end
            
            for m in 1:length(cats_b_bra)
                idxbs_bra = Vector{Int}()
                graph_b_bra = make_cat_graphs(fock_list_b_bra[m], bra)
                idxbs_bra = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_b_bra, 1, graph_b_bra.max, idxbs_bra, rev_bs_bra) 
                sort!(idxbs_bra)
                push!(all_cats_bra, HP_Category_Bra(m, connected_bra[m], idxbs_bra))
            end
            
            for k in 1:len_cat_b
                graph_b = make_cat_graphs(fock_list_b[k], ket)
                lu = ActiveSpaceSolvers.RASCI.dfs_c(graph_b, 1, graph_b.max, all_cats[k].lookup, all_cats, all_cats_bra, rev_as, rev_as_bra)
            end

            return all_cats_bra, all_cats        
        end

        if type == "a"
            #HP_Category_A
            for j in 1:len_cat_b
                idxbs = Vector{Int}()
                graph_b = make_cat_graphs(fock_list_b[j], ket)
                idxbs = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_b, 1, graph_b.max, idxbs, rev_bs) 
                sort!(idxbs)
                lu = zeros(Int, graph_b.no,length(idxbs))
                push!(all_cats, HP_Category_A(j, cats_b[j], connected[j], idxbs, lu))
            end

            #for k in 1:len_cat_b
            #    graph_b = make_cat_graphs(fock_list_b[k], ket)
            #    lu = ActiveSpaceSolvers.RASCI.dfs_a(graph_b, 1, graph_b.max, all_cats[k].lookup, all_cats, rev_as)
            #end
            for m in 1:length(cats_b_bra)
                idxbs_bra = Vector{Int}()
                graph_b_bra = make_cat_graphs(fock_list_b_bra[m], bra)
                idxbs_bra = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_b_bra, 1, graph_b_bra.max, idxbs_bra, rev_bs_bra) 
                sort!(idxbs_bra)
                push!(all_cats_bra, HP_Category_Bra(m, connected_bra[m], idxbs_bra))
            end
            return all_cats_bra, all_cats        
        end

        if type == "ca"
            #HP_Category_CA
            for j in 1:len_cat_b
                idxbs = Vector{Int}()
                graph_b = make_cat_graphs(fock_list_b[j], ket)
                idxbs = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_b, 1, graph_b.max, idxbs, rev_bs) 
                sort!(idxbs)
                lu = zeros(Int, graph_b.no, graph_b.no, length(idxbs))
                push!(all_cats, HP_Category_CA(j, cats_b[j], connected[j], idxbs, lu))
            end

            #for k in 1:len_cat_b
            #    graph_b = make_cat_graphs(fock_list_b[k], ket)
            #    lu = ActiveSpaceSolvers.RASCI.dfs_single_excitation(graph_b, 1, graph_b.max, all_cats[k].lookup, all_cats, rev_as)
            #end
            #
            for m in 1:length(cats_b_bra)
                idxbs_bra = Vector{Int}()
                graph_b_bra = make_cat_graphs(fock_list_b_bra[m], bra)
                idxbs_bra = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_b_bra, 1, graph_b_bra.max, idxbs_bra, rev_bs_bra) 
                sort!(idxbs_bra)
                push!(all_cats_bra, HP_Category_Bra(m, connected_bra[m], idxbs_bra))
            end
            return all_cats_bra, all_cats        
        end

        if type == "cc"
            #HP_Category_CC
            for j in 1:len_cat_b
                idxbs = Vector{Int}()
                graph_b = make_cat_graphs(fock_list_b[j], ket)
                idxbs = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_b, 1, graph_b.max, idxbs, rev_bs) 
                sort!(idxbs)
                lu = zeros(Int, graph_b.no, graph_b.no, length(idxbs))
                push!(all_cats, HP_Category_CC(j, cats_b[j], connected[j], idxbs, lu))
            end

            #for k in 1:len_cat_b
            #    graph_b = make_cat_graphs(fock_list_b[k], ket)
            #    lu = ActiveSpaceSolvers.RASCI.dfs_cc(graph_b, 1, graph_b.max, all_cats[k].lookup, all_cats, rev_as)
            #end
            for m in 1:length(cats_b_bra)
                idxbs_bra = Vector{Int}()
                graph_b_bra = make_cat_graphs(fock_list_b_bra[m], bra)
                idxbs_bra = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_b_bra, 1, graph_b_bra.max, idxbs_bra, rev_bs_bra) 
                sort!(idxbs_bra)
                push!(all_cats_bra, HP_Category_Bra(m, connected_bra[m], idxbs_bra))
            end
            return all_cats_bra, all_cats        
        end

        if type == "cca"
            #HP_Category_CCA
            for j in 1:len_cat_b
                idxbs = Vector{Int}()
                graph_b = make_cat_graphs(fock_list_b[j], ket)
                idxbs = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_b, 1, graph_b.max, idxbs, rev_bs) 
                sort!(idxbs)
                lu = zeros(Int, graph_b.no, graph_b.no, graph_b.no, length(idxbs))
                push!(all_cats, HP_Category_CCA(j, cats_b[j], connected[j], idxbs, lu))
            end

            #for k in 1:len_cat_b
            #    graph_b = make_cat_graphs(fock_list_b[k], ket)
            #    lu = ActiveSpaceSolvers.RASCI.dfs_cca(graph_b, 1, graph_b.max, all_cats[k].lookup, all_cats, all_cats_bra, rev_bs, rev_bs_bra)
            #end
            for m in 1:length(cats_b_bra)
                idxbs_bra = Vector{Int}()
                graph_b_bra = make_cat_graphs(fock_list_b_bra[m], bra)
                idxbs_bra = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_b_bra, 1, graph_b_bra.max, idxbs_bra, rev_bs_bra) 
                sort!(idxbs_bra)
                push!(all_cats_bra, HP_Category_Bra(m, connected_bra[m], idxbs_bra))
            end
            return all_cats_bra, all_cats        
        end

       if type == "ccaa"
           #HP_Category_CCAA
           for j in 1:len_cat_b
               idxbs = Vector{Int}()
               graph_b = make_cat_graphs(fock_list_b[j], ket)
               idxbs = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_b, 1, graph_b.max, idxbs, rev_bs) 
               sort!(idxbs)
               lu = zeros(Int, graph_b.no, graph_b.no, graph_b.no, graph_b.no, length(idxbs))
               push!(all_cats, HP_Category_CCAA(j, cats_b[j], connected[j], idxbs, lu))
           end

           for m in 1:length(cats_b_bra)
                idxbs_bra = Vector{Int}()
                graph_b_bra = make_cat_graphs(fock_list_b_bra[m], bra)
                idxbs_bra = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_b_bra, 1, graph_b_bra.max, idxbs_bra, rev_bs_bra) 
                sort!(idxbs_bra)
                push!(all_cats_bra, HP_Category_Bra(m, connected_bra[m], idxbs_bra))
            end
           
            for k in 1:len_cat_b
                graph_b = make_cat_graphs(fock_list_b[k], ket)
                lu = ActiveSpaceSolvers.RASCI.dfs_ccaa(graph_b, 1, graph_b.max, all_cats[k].lookup, all_cats, all_cats_bra, rev_bs, rev_bs_bra)
            end
            return all_cats_bra, all_cats        
       end
    end
end
    




    

    










    

