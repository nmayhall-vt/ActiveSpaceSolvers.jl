using JLD2
using InCoreIntegrals
using StaticArrays

abstract type HP_Category end

#the hp probably isnt need i dont use it for anything 

struct HP_Category_CA <: HP_Category
    idx::Int #its index
    hp::Tuple{Int, Int} #(holes, particles)
    connected::Vector{Int} #list of allowed pairings of other spin categories
    idxs::Vector{Int}
    lookup::Array{Int, 3} #single spin lookup table for single excitations
end

struct HP_Category_C <: HP_Category
    idx::Int #its index
    hp::Tuple{Int, Int} #(holes, particles)
    connected::Vector{Int} #list of allowed pairings of other spin categories
    idxs::Vector{Int}
    lookup::Array{Int, 2} #single spin lookup table for single excitations
end

struct HP_Category_A <: HP_Category
    idx::Int #its index
    hp::Tuple{Int, Int} #(holes, particles)
    connected::Vector{Int} #list of allowed pairings of other spin categories
    idxs::Vector{Int}
    lookup::Array{Int, 2} #single spin lookup table for single excitations
end

struct HP_Category_CC <: HP_Category
    idx::Int #its index
    hp::Tuple{Int, Int} #(holes, particles)
    connected::Vector{Int} #list of allowed pairings of other spin categories
    idxs::Vector{Int}
    lookup::Array{Int, 3} #single spin lookup table for single excitations
end

struct HP_Category_CCA <: HP_Category
    idx::Int #its index
    hp::Tuple{Int, Int} #(holes, particles)
    connected::Vector{Int} #list of allowed pairings of other spin categories
    idxs::Vector{Int}
    lookup::Array{Int, 4} #single spin lookup table for single excitations
end

struct HP_Category_CCAA <: HP_Category
    idx::Int #its index
    hp::Tuple{Int, Int} #(holes, particles)
    connected::Vector{Int} #list of allowed pairings of other spin categories
    idxs::Vector{Int}
    lookup::Array{Int, 5} #single spin lookup table for single excitations
end

struct HP_Category_Bra <: HP_Category
    idx::Int
    connected::Vector{Int}
    idxs::Vector{Int}
end


function HP_Category_CA()
    return HP_Category_CA(1,(0,0),Vector{Int}(), Vector{Int}(), Vector{Int}(), Array{Int, 3}())
    #return HP_Category_CA(1,(1,1),Vector{Int}(), Vector{Int}(), Vector{Int}(), Vector{Array{Int, 3}}())
end

function HP_Category_CA(idx::Int, hp::Tuple{Int,Int}, connected::Vector{Int})
    return HP_Category_CA(idx, hp, Vector{Int}(), Vector{Int}(), connected, Array{Int, 3}())
end


function make_categories(prob::RASCIAnsatz; spin="alpha")
    categories = ActiveSpaceSolvers.RASCI.generate_spin_categories(prob)#={{{=#
    all_cats = Vector{HP_Category_CA}()
    
    cats_a = deepcopy(categories)
    cats_b = deepcopy(categories)
    fock_list_a, del_at_a = make_fock_from_categories(categories, prob, "alpha")
    deleteat!(cats_a, del_at_a)
    len_cat_a = length(cats_a)
        
    fock_list_b, del_at_b = make_fock_from_categories(categories, prob, "beta")
    deleteat!(cats_b, del_at_b)
    len_cat_b = length(cats_b)

    if spin == "alpha"
        connected = make_spincategory_connections(cats_a, cats_b, prob)

        #compute configs
        as = compute_config_dict(fock_list_a, prob, "alpha")
        #this reverses the config dictionary to get the index as the key 
        rev_as = Dict(value => key for (key, value) in as)
        #as_old =  ActiveSpaceSolvers.RASCI.compute_configs(prob)[1]
        #for i in keys(rev_as)
        #    idx = as_old[i]
        #    rev_as[i] = idx
        #end
        max_a = length(as)

        for j in 1:len_cat_a
            idxas = Vector{Int}()
            graph_a = make_cat_graphs(fock_list_a[j], prob)
            idxas = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_a, 1, graph_a.max, idxas, rev_as) 
            sort!(idxas)
            lu = zeros(Int, graph_a.no, graph_a.no, length(idxas))
            #lu = zeros(Int, graph_a.no, graph_a.no, max_a)
            push!(all_cats, HP_Category_CA(j, cats_a[j], connected[j], idxas, lu))
        end
        
        #have to do same loop as before bec all categories need initalized for the dfs search for lookup tables
        for k in 1:len_cat_a
            graph_a = make_cat_graphs(fock_list_a[k], prob)
            lu = ActiveSpaceSolvers.RASCI.dfs_single_excitation(graph_a, 1, graph_a.max, all_cats[k].lookup, all_cats, rev_as)
            all_cats[k].lookup .= lu
        end
        return all_cats

    else
        connected = make_spincategory_connections(cats_b, cats_a, prob)
        #compute configs
        bs = compute_config_dict(fock_list_b, prob, "beta")
        rev_bs = Dict(value => key for (key, value) in bs)
        #bs_old = ActiveSpaceSolvers.RASCI.compute_configs(prob)[2]
        #for i in keys(rev_bs)
        #    idx = bs_old[i]
        #    rev_bs[i] = idx
        #end
        max_b = length(bs)
        
        for j in 1:len_cat_b
            idxbs = Vector{Int}()
            graph_b = make_cat_graphs(fock_list_b[j], prob)
            idxbs = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_b, 1, graph_b.max, idxbs, rev_bs) 
            sort!(idxbs)
            lu = zeros(Int, graph_b.no, graph_b.no, length(idxbs))
            #lu = zeros(Int, graph_b.no, graph_b.no, max_b)
            push!(all_cats, HP_Category_CA(j, cats_b[j], connected[j], idxbs, lu))
        end

        for k in 1:len_cat_b
            graph_b = make_cat_graphs(fock_list_b[k], prob)
            lu = ActiveSpaceSolvers.RASCI.dfs_single_excitation(graph_b, 1, graph_b.max, all_cats[k].lookup, all_cats, rev_bs)
            all_cats[k].lookup .= lu
        end
        return all_cats
    end#=}}}=#
end

function test_lus(categories, old_configs, prob)
    a_lookup = ActiveSpaceSolvers.RASCI.fill_lookup(prob, old_configs, prob.dima)#={{{=#

    for Ia in 1:prob.dima
        cat = find_cat(Ia, categories)
        
        for k in 1:prob.no, l in 1:prob.no
            K_idx = cat.lookup[k,l,Ia]
            K_old = a_lookup[k,l,Ia]
            
            if K_idx != K_old
                println("nope")
            end

        end
    end
end#=}}}=#

function compute_config_dict(fock_list, prob::RASCIAnsatz, spin="alpha")
    configs_dict = Dict{Int, Vector{Int32}}()#={{{=#
    configs = []
    for j in 1:length(fock_list)
        graph_a = ActiveSpaceSolvers.RASCI.make_cat_graphs(fock_list[j], prob)
        if spin == "alpha"
            config_dict = ActiveSpaceSolvers.RASCI.old_dfs(prob.na, graph_a.connect, graph_a.weights, 1, graph_a.max)
        else
            config_dict = ActiveSpaceSolvers.RASCI.old_dfs(prob.nb, graph_a.connect, graph_a.weights, 1, graph_a.max)
        end
        for x in keys(config_dict)
            push!(configs, x)
        end
        
    end
    
    for i in 1:length(configs)
        configs_dict[i] = configs[i]
    end
    return configs_dict#=}}}=#
end


function generate_spin_categories(prob::RASCIAnsatz)
    categories = []#={{{=#

    for h in 1:prob.max_h+1
        holes = h-1
        for p in 1:prob.max_p+1
            particles = p-1
            cat = (holes, particles)
            push!(categories, cat)
        end
    end
    return categories#=}}}=#
end

function make_spincategory_connections(cats1, cats2, prob::RASCIAnsatz)
    connected = Vector{Vector{Int}}()#={{{=#
    for i in 1:length(cats1)
        tmp = Vector{Int}()
        for j in 1:length(cats2)
            if cats1[i][1]+cats2[j][1] <= prob.max_h
                if cats1[i][2]+cats2[j][2] <= prob.max_p
                    append!(tmp, j)
                end
            end
        end
        append!(connected, [tmp])
    end
    return connected#=}}}=#
end

function make_category_connections(categories, prob::RASCIAnsatz)
    #={{{=#
    connected = Vector{Vector{Int}}()
    for i in 1:length(categories)
        tmp = Vector{Int}()
        for j in 1:length(categories)
            if categories[i][1]+categories[j][1] <= prob.max_h
                if categories[i][2]+categories[j][2] <= prob.max_p
                    append!(tmp, j)
                end
            end
        end
        append!(connected, [tmp])
    end
    return connected#=}}}=#
end


function make_fock_from_categories(categories, prob::RASCIAnsatz, spin="alpha")
    fock_list = []#={{{=#
    cat_delete = []
    if spin == "alpha"
        if prob.na < prob.ras_spaces[1]
            start = (prob.na, 0, 0)
        elseif prob.na > prob.ras_spaces[1]+prob.ras_spaces[2]
            start = (prob.ras_spaces[1], prob.ras_spaces[2], prob.na-(prob.ras_spaces[1]+prob.ras_spaces[2]))
        else
            start = (prob.ras_spaces[1], prob.na-prob.ras_spaces[1], 0)
        end

        for i in 1:length(categories)
            fock = (start[1]-categories[i][1],prob.na-((start[3]+categories[i][2])+(start[1]-categories[i][1])) ,start[3]+categories[i][2])
            push!(fock_list, fock)

            if any(fock.<0)
                push!(cat_delete, i)
                continue
            end

            if fock[1]>prob.ras_spaces[1] || fock[2]>prob.ras_spaces[2] || fock[3]>prob.ras_spaces[3]
                push!(cat_delete, i)
            end
        end
    
    else

        if prob.nb < prob.ras_spaces[1]
            start = (prob.nb, 0, 0)

        elseif prob.nb > prob.ras_spaces[1]+prob.ras_spaces[2]
            start = (prob.ras_spaces[1], prob.ras_spaces[2], prob.nb-(prob.ras_spaces[1]+prob.ras_spaces[2]))
        else
            start = (prob.ras_spaces[1], prob.nb-prob.ras_spaces[1], 0)
        end

        for i in 1:length(categories)
            fock = (start[1]-categories[i][1],prob.nb-((start[3]+categories[i][2])+(start[1]-categories[i][1])) ,start[3]+categories[i][2])
            push!(fock_list, fock)
            if any(fock.<0)
                push!(cat_delete, i)
                continue
            end
            
            if fock[1]>prob.ras_spaces[1] || fock[2]>prob.ras_spaces[2] || fock[3]>prob.ras_spaces[3]
                push!(cat_delete, i)
            end
        end
    end
    deleteat!(fock_list, cat_delete)
    return fock_list, cat_delete#=}}}=#
end

function make_cat_graphs(fock_list, prob::RASCIAnsatz)
    #this function will make RASCI Olsen graphs from given fock sector lists{{{
    ras1 = ActiveSpaceSolvers.RASCI.make_ras_x(prob.ras_spaces[1], fock_list[1], SVector(prob.ras_spaces[1], 0, 0), 0, 0)
    ras2 = ActiveSpaceSolvers.RASCI.make_ras_x(prob.ras_spaces[2], fock_list[2], SVector(prob.ras_spaces[2], 0, 0), 0, 0)
    ras3 = ActiveSpaceSolvers.RASCI.make_ras_x(prob.ras_spaces[3], fock_list[3], SVector(prob.ras_spaces[3], 0, 0), 0, 0)
    
    n_unocc_ras2 = (prob.ras_spaces[2]-fock_list[2])+1
    n_unocc_ras3 = (prob.ras_spaces[3]-fock_list[3])+1
    
    update_x_subgraphs!(ras2, n_unocc_ras2, fock_list[2], maximum(ras1))
    update_x_subgraphs!(ras3, n_unocc_ras3, fock_list[3], maximum(ras2))
    
    rows = size(ras1,1)+size(ras2,1)+ size(ras3, 1)-2
    columns = size(ras1, 2) + size(ras2, 2) + size(ras3,2)-2
    full = zeros(Int, rows, columns)
    loc = [size(ras1,1),size(ras1,2)]
    full[1:size(ras1,1), 1:size(ras1,2)] .= ras1
    loc2 = [size(ras2,1)+loc[1]-1, loc[2]+size(ras2,2)-1]
    full[loc[1]:loc2[1], loc[2]:loc2[2]] .= ras2
    loc3 = [size(ras3,1)+loc2[1]-1, size(ras3,2)+loc2[2]-1]
    full[loc2[1]:loc3[1], loc2[2]:loc3[2]] .= ras3
    y = ActiveSpaceSolvers.RASCI.make_ras_y(full)
    vert, max_val = ActiveSpaceSolvers.RASCI.make_vert_graph_ras(full)
    connect, weights = ActiveSpaceSolvers.RASCI.make_graph_dict(y, vert)
    graph = ActiveSpaceSolvers.RASCI.RASCI_OlsenGraph(prob.no, sum(fock_list), prob.ras_spaces, max_val, vert, connect, weights)
    #graph = ActiveSpaceSolvers.RASCI.RASCI_OlsenGraph(prob.no, sum(fock_list), prob.ras_spaces, prob.ras1_min, prob.ras3_max, max_val, vert, connect, weights)
    return graph#=}}}=#
end

function update_x_subgraphs!(x, n_unocc, nelec, shift)
    if size(x,2) != 0#={{{=#
        x[:,1] .= shift
    end

    if size(x,1) != 0
        x[1,:] .= shift
    end
    for i in 2:nelec+1
        for j in 2:n_unocc
            x[j, i] = x[j-1, i] + x[j, i-1]
        end
    end#=}}}=#
end


function find_cat(idx::Int, categories::Vector{<:HP_Category})
    #this function will find the category that idx belongs to{{{
    for cat in categories
        if idx in cat.idxs
            return cat
        else
            continue
        end
    end
    return 0#=}}}=#
end


function sigma_one(prob::RASCIAnsatz, cats_a::Vector{HP_Category_CA}, cats_b::Vector{HP_Category_CA}, ints::InCoreInts, v)
    T = eltype(v[1])#={{{=#
    n_roots::Int = size(v,3)
    sigma_one = zeros(prob.dima, prob.dimb, n_roots)
    
    F = zeros(T, prob.dimb)
    gkl = get_gkl(ints, prob) 
    
    sigma_one = permutedims(sigma_one,[1,3,2])
    v = permutedims(v,[1,3,2])

    for Ib in 1:prob.dimb
        cat = find_cat(Ib, cats_b)
        fill!(F,0.0)
        comb_kl = 0
        comb_ij = 0
        for k in 1:prob.no, l in 1:prob.no
            K_idx = cat.lookup[l,k,Ib]
            K_idx != 0 || continue
            sign_kl = sign(K_idx)
            K = abs(K_idx)
            @inbounds F[K] += sign_kl*gkl[k,l]
            comb_kl = (k-1)*prob.no + l
            next_cat = find_cat(K, cats_b)

            for i in 1:prob.no, j in 1:prob.no
                comb_ij = (i-1)*prob.no + j
                if comb_ij < comb_kl
                    continue
                end
                J_idx = next_cat.lookup[j,i,K]
                J_idx != 0 || continue
                sign_ij = sign(J_idx)
                J = abs(J_idx)
                final_cat = find_cat(J, cats_b)
                if comb_kl == comb_ij
                    delta = 1
                else
                    delta = 0
                end

                if sign_kl == sign_ij
                    F[J] += (ints.h2[i,j,k,l]*1/(1+delta))
                else
                    F[J] -= (ints.h2[i,j,k,l]*1/(1+delta))
                end
            end
        end

        _ras_ss_sum!(sigma_one, v, F, Ib, cats_a, cats_b, sigma="one")
    end
    sigma_one = permutedims(sigma_one,[1,3,2])
    v = permutedims(v,[1,3,2])
    return sigma_one#=}}}=#
end

function sigma_two(prob::RASCIAnsatz, cats_a::Vector{HP_Category_CA}, cats_b::Vector{HP_Category_CA}, ints::InCoreInts, v)
    T = eltype(v[1])#={{{=#
    n_roots::Int = size(v,3)
    sigma_two = zeros(prob.dima, prob.dimb, n_roots)
    
    F = zeros(T, prob.dima)
    gkl = get_gkl(ints, prob) 
    
    sigma_two = permutedims(sigma_two,[2,3,1])
    v = permutedims(v,[2,3,1])
    
    for Ib in 1:prob.dima
        cat = find_cat(Ib, cats_a)
        fill!(F,0.0)
        comb_kl = 0
        comb_ij = 0
        for k in 1:prob.no, l in 1:prob.no
            K_idx = cat.lookup[l,k,Ib]
            K_idx != 0 || continue
            sign_kl = sign(K_idx)
            K = abs(K_idx)
            @inbounds F[K] += sign_kl*gkl[k,l]
            comb_kl = (k-1)*prob.no + l
            next_cat = find_cat(K, cats_a)

            for i in 1:prob.no, j in 1:prob.no
                comb_ij = (i-1)*prob.no + j
                if comb_ij < comb_kl
                    continue
                end
                J_idx = next_cat.lookup[j,i,K]
                J_idx != 0 || continue
                sign_ij = sign(J_idx)
                J = abs(J_idx)
                final_cat = find_cat(J, cats_a)
                if comb_kl == comb_ij
                    delta = 1
                else
                    delta = 0
                end

                if sign_kl == sign_ij
                    F[J] += ((ints.h2[i,j,k,l])*1/(1+delta))
                else
                    F[J] -= ((ints.h2[i,j,k,l])*1/(1+delta))
                end
            end
        end

        _ras_ss_sum!(sigma_two, v, F, Ib, cats_a, cats_b, sigma="two")
    end
    
    sigma_two = permutedims(sigma_two,[3,1,2])
    v = permutedims(v,[3,1,2])
    return sigma_two#=}}}=#
end

function slow_sigma_three(prob::RASCIAnsatz, cats_a::Vector{HP_Category_CA}, cats_b::Vector{HP_Category_CA}, ints::InCoreInts, v)
    #={{{=#
    T = eltype(v[1])
    n_roots::Int = size(v,3)
    sigma_three = zeros(Float64, prob.dima, prob.dimb,n_roots)
    
    hkl = zeros(Float64, prob.no, prob.no)

    for Ia in 1:prob.dima
        cat_Ia = find_cat(Ia, cats_a)
        fill!(hkl, T(0.0))
        for k in 1:prob.no, l in 1:prob.no
            Ja = cat_Ia.lookup[l,k,Ia]
            Ja != 0 || continue
            sign_kl = sign(Ja)
            Ja = abs(Ja)
            hkl .= ints.h2[:,:,k,l]
            cat_Ja = find_cat(Ja, cats_a)
            for Ib in 1:prob.dimb
                cat_Ib = find_cat(Ib, cats_b)
                for i in 1:prob.no, j in 1:prob.no
                    Jb = cat_Ib.lookup[j,i,Ib]
                    Jb != 0 || continue
                    sign_ij = sign(Jb)
                    Jb = abs(Jb)
                    cat_Jb = find_cat(Jb, cats_b)
                    if cat_Ib.idx in cat_Ia.connected
                        if cat_Jb.idx in cat_Ja.connected
                            for si in 1:n_roots
                                sigma_three[Ia, Ib, si] += hkl[i,j]*v[Ja, Jb, si]*sign_ij*sign_kl
                            end
                        end
                    end
                end
            end
        end
    end#=}}}=#
    return sigma_three
end


function sigma_three(prob::RASCIAnsatz, cats_a::Vector{HP_Category_CA}, cats_b::Vector{HP_Category_CA}, ints::InCoreInts, v)
    T = eltype(v[1])#={{{=#
    n_roots::Int = size(v,3)
    sigma_three = zeros(T, prob.dima, prob.dimb,n_roots)
    
    hkl = zeros(T, prob.no, prob.no)
    FJb = zeros(T, prob.dimb)
    
    sigma_three = permutedims(sigma_three,[1,3,2])
    v = permutedims(v,[1,3,2])

    for k in 1:prob.no, l in 1:prob.no
        L = Vector{Int}()
        R = Vector{Int}()
        sign_I = Vector{Int8}()
        #loop over all alpha configs
        for I in 1:prob.dima
            cat_I = find_cat(I, cats_a)
            Iidx = cat_I.lookup[l,k,I]
            if Iidx != 0
                push!(R,I)
                push!(L,abs(Iidx))
                push!(sign_I, sign(Iidx))
            end
        end
        
        Ckl = zeros(T, length(L), n_roots, prob.dimb)
        
        #Gather
        _gather!(Ckl, v, R, L, sign_I)
        VI = zeros(T, length(L), n_roots)
        
        hkl .= ints.h2[:,:,k,l]
        for Ib in 1:prob.dimb
            cat_Ib = find_cat(Ib, cats_b)
            @inbounds fill!(FJb, T(0.0))
            @inbounds fill!(VI, T(0.0))
            for i in 1:prob.no, j in 1:prob.no
                Jb = cat_Ib.lookup[j,i,Ib]
                Jb != 0 || continue
                sign_ij = sign(Jb)
                Jb = abs(Jb)
                @inbounds FJb[Jb] += sign_ij*hkl[i,j]
            end
            
            _ras_ss_sum_sig3!(VI, Ckl, FJb, L, cats_a, cats_b)
            #Scatter
            _scatter!(sigma_three, VI, Ib, R, cats_a, cats_b)
        end
    end
    sigma_three = permutedims(sigma_three,[1,3,2])
    return sigma_three#=}}}=#
end

function _ras_ss_sum_sig3!(VI::Array{T,2}, Ckl::Array{T,3}, F::Vector{T}, L::Vector{Int}, cats_a::Vector{HP_Category_CA}, cats_b::Vector{HP_Category_CA}) where {T}
    nIa = size(L)[1]#={{{=#
    n_roots = size(VI)[2]
    
    for catb in cats_b    
        for Jb in catb.idxs
            for Ia in 1:nIa
                cat_Ia = find_cat(L[Ia], cats_a)
                if cat_Ia.idx in catb.connected
                    if abs(F[Jb]) > 1e-14 
                        @inbounds @simd for si in 1:n_roots
                            VI[Ia,si] += F[Jb]*Ckl[Ia,si,Jb]
                        end
                    end
                end
            end
        end
    end
end#=}}}=#

function _ras_ss_sum!(sig::Array{T,3}, v::Array{T,3}, F::Vector{T},Ib::Int, cats_a::Vector{HP_Category_CA}, cats_b::Vector{HP_Category_CA}; sigma="one") where {T}
    n_roots = size(v)[2]#={{{=#
    count = 0
    
    nIa     = size(v)[1]
    n_roots = size(v)[2]
    nJb     = size(v)[3]
    if sigma == "one"
        current_cat = find_cat(Ib, cats_b)
        for catb in cats_b    
            for cats in catb.connected
                for Ia in cats_a[cats].idxs
                    for Jb in catb.idxs
                        if abs(F[Jb]) > 1e-14 
                            if cats_a[cats].idx in current_cat.connected
                                @inbounds @simd for si in 1:n_roots
                                    sig[Ia,si,Ib] += F[Jb]*v[Ia,si,Jb]
                                end
                            end
                        end
                    end
                end
            end
        end
    else
        current_cat = find_cat(Ib, cats_a)
        for cata in cats_a    
            for cats in cata.connected
                for Ia in cats_b[cats].idxs
                    for Jb in cata.idxs
                        if abs(F[Jb]) > 1e-14 
                            if cats_b[cats].idx in current_cat.connected
                                @inbounds @simd for si in 1:n_roots
                                    sig[Ia,si,Ib] += F[Jb]*v[Ia,si,Jb]
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end#=}}}=#

function _gather!(Ckl::Array{T,3}, v, R::Vector{Int}, L::Vector{Int}, sign_I::Vector{Int8}) where {T}
    nI = length(L)#={{{=#
    n_roots = size(v)[2]
    ket_max = size(v)[3]
    
    @inbounds @simd for si in 1:n_roots
        for Jb in 1:ket_max
            for Li in 1:nI
                Ckl[Li,si, Jb] = v[L[Li], si, Jb]*sign_I[Li]
            end
        end
    end
end#=}}}=#

function _scatter!(sig::Array{T, 3}, VI::Array{T, 2}, Ib::Int, R::Vector{Int}, cats_a::Vector{HP_Category_CA}, cats_b::Vector{HP_Category_CA}) where {T}
    n_roots = size(sig)[2]#={{{=#

    curr_cat = find_cat(Ib, cats_b)

    for si in 1:n_roots
        for I in 1:length(R)
            cat_I = find_cat(R[I], cats_a)
            if cat_I.idx in curr_cat.connected
                sig[R[I], si, Ib] += VI[I, si]
            end
        end
    end
end#=}}}=#

function S2_helper(P::RASCIAnsatz)
    categories = ActiveSpaceSolvers.RASCI.generate_spin_categories(P)#={{{=#
    all_cats_a = Vector{HP_Category_CA}()
    all_cats_b = Vector{HP_Category_CA}()
    
    cats_a = deepcopy(categories)
    cats_b = deepcopy(categories)
    fock_list_a, del_at_a = make_fock_from_categories(categories, P, "alpha")
    deleteat!(cats_a, del_at_a)
    len_cat_a = length(cats_a)
        
    fock_list_b, del_at_b = make_fock_from_categories(categories, P, "beta")
    deleteat!(cats_b, del_at_b)
    len_cat_b = length(cats_b)

    #alpha
    connected_a = make_spincategory_connections(cats_a, cats_b, P)

    #compute configs
    as = compute_config_dict(fock_list_a, P, "alpha")
    as_old =  ActiveSpaceSolvers.RASCI.compute_configs(P)[1]
    rev_as = Dict(value => key for (key, value) in as)
    #this reverses the config dictionary to get the index as the key 
    #for i in keys(rev_as)
    #    idx = as_old[i]
    #    rev_as[i] = idx
    #end
    max_a = length(as)

    for j in 1:len_cat_a
        idxas = Vector{Int}()
        graph_a = make_cat_graphs(fock_list_a[j], P)
        idxas = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_a, 1, graph_a.max, idxas, rev_as) 
        lu = zeros(Int, graph_a.no, graph_a.no, max_a)
        push!(all_cats_a, HP_Category_CA(j, cats_a[j], connected_a[j], idxas, lu))
    end

    #have to do same loop as before bec all categories need initalized for the dfs search for lookup tables
    for k in 1:len_cat_a
        graph_a = make_cat_graphs(fock_list_a[k], P)
        ActiveSpaceSolvers.RASCI.dfs_single_excitation!(graph_a, 1, graph_a.max, all_cats_a[k].lookup, all_cats_a, rev_as)
    end

    #beta
    connected_b = make_spincategory_connections(cats_b, cats_a, P)
    #compute configs
    bs = compute_config_dict(fock_list_b, P, "beta")
    bs_old = ActiveSpaceSolvers.RASCI.compute_configs(P)[2]
    rev_bs = Dict(value => key for (key, value) in bs)
    #for i in keys(rev_bs)
    #    idx = bs_old[i]
    #    rev_bs[i] = idx
    #end
    max_b = length(bs)

    for j in 1:len_cat_b
        idxbs = Vector{Int}()
        graph_b = make_cat_graphs(fock_list_b[j], P)
        idxbs = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_b, 1, graph_b.max, idxbs, rev_bs) 
        lu = zeros(Int, graph_b.no, graph_b.no, max_b)
        push!(all_cats_b, HP_Category_CA(j, cats_b[j], connected_b[j], idxbs, lu))
    end

    for k in 1:len_cat_b
        graph_b = make_cat_graphs(fock_list_b[k], P)
        ActiveSpaceSolvers.RASCI.dfs_single_excitation!(graph_b, 1, graph_b.max, all_cats_b[k].lookup, all_cats_b, rev_bs)
    end
#=}}}=#
    return as, bs, rev_as, rev_bs, all_cats_a, all_cats_b
end

"""
    compute_S2_expval(prb::RASCIAnsatz)
- `prb`: RASCIAnsatz just defines the current CI ansatz (i.e., ras_spaces sector)
"""
function compute_S2_expval(v::Matrix, P::RASCIAnsatz)
    ###{{{
    #S2 = (S+S- + S-S+)1/2 + Sz.Sz
    #   = 1/2 sum_ij(ai'bi bj'ai + bj'aj ai'bi) + Sz.Sz
    #   do swaps and you can end up adding the two together to get rid
    #   of the 1/2 factor so 
    #   = (-1) sum_ij(ai'aj|alpha>bj'bi|beta> + Sz.Sz
    ###
    
    as, bs, rev_as, rev_bs, all_cats_a, all_cats_b = S2_helper(P)
    
    
    nr = size(v,2)
    s2 = zeros(nr)
    
    for Ib in 1:P.dimb
        config_b = bs[Ib]
        cat_Ib = find_cat(Ib, all_cats_b)
        for Ia in 1:P.dima
            config_a = as[Ia]
            cat_Ia = find_cat(Ia, all_cats_a)
            #if cat_Ia.idx in cat_Ib.connected

                K = Ia + (Ib-1)*P.dima  
                
                #Sz.Sz (α) 
                count_a = (P.na-1)*P.na
                for i in 1:count_a
                    for r in 1:nr
                        s2[r] += 0.25*v[K,r]*v[K,r]
                    end
                end
                #for ai in config_a
                #    for aj in config_a
                #        if ai!= aj
                #            for r in 1:nr
                #                s2[r] += 0.25 * v[K,r]*v[K,r]
                #            end
                #        end
                #    end
                #end
                #
                #Sz.Sz (β)
                count_b = (P.nb-1)*P.nb
                for j in 1:count_b
                    for r in 1:nr
                        s2[r] += 0.25*v[K,r]*v[K,r]
                    end
                end
                #for bi in config_b
                #    for bj in config_b
                #        if bi != bj
                #            for r in 1:nr
                #                s2[r] += 0.25 * v[K,r]*v[K,r]
                #            end
                #        end
                #    end
                #end

                ###Sz.Sz (α,β)
                for ai in config_a
                    for bj in config_b
                        if ai != bj
                            for r in 1:nr
                                s2[r] -= .5 * v[K,r]*v[K,r] 
                            end
                        end
                    end
                end

                ##Sp.Sm + Sm.Sp Diagonal Part
                for ai in config_a
                    if ai in config_b
                    else
                        for r in 1:nr
                            s2[r] += .75 * v[K,r]*v[K,r] 
                        end
                    end
                end

                for bi in config_b
                    if bi in config_a
                    else
                        for r in 1:nr
                            s2[r] += .75 * v[K,r]*v[K,r] 
                        end
                    end
                end
                
                #(Sp.Sm + Sm.Sp)1/2 Off Diagonal Part
                for ai in config_a
                    for bj in config_b
                        if ai ∉ config_b
                            if bj ∉ config_a
                                #Sp.Sm + Sm.Sp
                                La = cat_Ia.lookup[ai,bj,Ia]
                                if La != 0
                                    sign_a = sign(La)
                                    La = abs(La)
                                    Lb = cat_Ib.lookup[bj,ai,Ib]
                                    if Lb != 0
                                        sign_b = sign(Lb)
                                        Lb = abs(Lb)
                                        L = La + (Lb-1)*P.dima
                                        for r in 1:nr
                                            s2[r] -= sign_a*sign_b*v[K,r]*v[L,r]
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            #end
        end
    end
    return s2#=}}}=#
end

"""
    apply_S2_matrix(P::RASCIAnsatz, v::AbstractArray{T}) where {T}
- `P`: RASCIAnsatz just defines the current CI ansatz (i.e., ras_spaces sector)
"""
function apply_S2_matrix(P::RASCIAnsatz, v::AbstractArray{T}) where T
    ###{{{
    #S2 = (S+S- + S-S+)1/2 + Sz.Sz
    #   = 1/2 sum_ij(ai'bi bj'ai + bj'aj ai'bi) + Sz.Sz
    #   do swaps and you can end up adding the two together to get rid
    #   of the 1/2 factor so 
    #   = (-1) sum_ij(ai'aj|alpha>bj'bi|beta> + Sz.Sz
    ###
    
    as, bs, rev_as, rev_bs, all_cats_a, all_cats_b = S2_helper(P)
    
    P.dim == size(v,1) || throw(DimensionMismatch)
    S2v = zeros(size(v)...)
    
    for Ib in 1:P.dimb
        config_b = bs[Ib]
        cat_Ib = find_cat(Ib, all_cats_b)
        for Ia in 1:P.dima
            config_a = as[Ia]
            cat_Ia = find_cat(Ia, all_cats_a)
            #if cat_Ia.idx in cat_Ib.connected

                K = Ia + (Ib-1)*P.dima  
                
                #Sz.Sz (α) 
                count_a = (P.na-1)*P.na
                for i in 1:count_a
                    S2v[K,:] .+= 0.25.*v[K,:]
                end
                
                #Sz.Sz (β)
                count_b = (P.nb-1)*P.nb
                for j in 1:count_b
                    S2v[K,:] .+= 0.25.*v[K,:]
                end
                
                #Sz.Sz (α,β)
                for ai in config_a
                    for bj in config_b
                        if ai != bj
                            S2v[K,:] .-= 0.5.*v[K,:]
                        end
                    end
                end

                ##Sp.Sm + Sm.Sp Diagonal Part
                for ai in config_a
                    if ai in config_b
                    else
                        S2v[K,:] .+= 0.75.*v[K,:]
                    end
                end

                for bi in config_b
                    if bi in config_a
                    else
                        S2v[K,:] .+= 0.75.*v[K,:]
                    end
                end
                
                #(Sp.Sm + Sm.Sp)1/2 Off Diagonal Part
                for ai in config_a
                    for bj in config_b
                        if ai ∉ config_b
                            if bj ∉ config_a
                                #Sp.Sm + Sm.Sp
                                La = cat_Ia.lookup[ai,bj,Ia]
                                if La != 0
                                    sign_a = sign(La)
                                    La = abs(La)
                                    Lb = cat_Ib.lookup[bj,ai,Ib]
                                    if Lb != 0
                                        sign_b = sign(Lb)
                                        Lb = abs(Lb)
                                        L = La + (Lb-1)*P.dima
                                        S2v[K,:] .-= sign_a*sign_b*v[L,:]
                                    end
                                end
                            end
                        end
                    end
                end
            #end
        end
    end
    return S2v#=}}}=#
end


function generate_fock_sectors(prob::RASCIAnsatz)#={{{=#
    alpha_sectors = []
    beta_sectors = []

    for i in 1:prob.ras3_max+1
        ras3 = i-1
        ras3 <= prob.ras_spaces[3] || continue
        for j in 1:prob.ras_spaces[1]+1
            ras1 = j-1
            ras1 <= prob.ras_spaces[1] || continue
            ras2 = prob.na-ras1-ras3
            ras2 <= prob.ras_spaces[2] && ras2 >=0 || continue
            push!(alpha_sectors, [ras1, ras2, ras3])
        end
    end
    if prob.na == prob.nb
        beta_sectors = alpha_sectors
        return alpha_sectors, beta_sectors
    else
        for i in 1:prob.ras3_max+1
            ras3 = i-1
            ras3 <= prob.ras_spaces[3] || continue
            for j in 1:prob.ras_spaces[1]+1
                ras1 = j-1 
                ras1 <= prob.ras_spaces[1] || continue
                ras2 = prob.nb-ras1-ras3
                ras2 <= prob.ras_spaces[2] && ras2 >=0 || continue
                push!(beta_sectors, [ras1, ras2, ras3])
            end
        end
    end

    return alpha_sectors, beta_sectors
end

function find_paths_indexes(categories::Vector{HP_Category}, graph::ActiveSpaceSolvers.RASCI.RASCI_OlsenGraph, max_dim::Int, config_dict::Dict{Vector{Int32}, Int}, lookup::Array{Int, 3})#={{{=#
    #this function will do a depth first search (single excitation dfs), find binomial index, and fill lookup tables
    #using binomial indexing from nicks FCI code
    #lu = zeros(Int, graph.no, graph.no, max_dim)
    lu  = ActiveSpaceSolvers.RASCI.dfs_single_excitation!(graph, 1, graph.max, lookup, categories, config_dict)
    return lu
end#=}}}=#


function compute_sector_coupling(a, b, prob::RASCIAnsatz; spin="alpha")
    pairs = Vector{Vector{Int}}()
    if spin == "alpha"
        for i in 1:length(a)
            tmp = Vector{Int}()
            for j in 1:length(b)
                if a[i][3]+b[j][3]<=prob.ras3_max && a[i][1]+b[j][1]>=prob.ras1_min
                    append!(tmp, j)
                    #push!(tmp, (i, j))
                end
            end
            append!(pairs, [tmp])
        end

    else
        for i in 1:length(b)
            tmp = Vector{Int}()
            for j in 1:length(a)
                if a[i][3]+b[j][3]<=prob.ras3_max && a[i][1]+b[j][1]>=prob.ras1_min
                    append!(tmp, j)
                    #push!(tmp, (i, j))
                end
            end
            append!(pairs, [tmp])
        end
    end
    return pairs
end#=}}}=#

    









