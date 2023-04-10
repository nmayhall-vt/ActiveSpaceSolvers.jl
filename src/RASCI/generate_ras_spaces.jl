using JLD2
using InCoreIntegrals
using StaticArrays

mutable struct Spin_Categories
    idx::Int #its index
    hp::Tuple{Int, Int} #(holes, particles)
    connected::Vector{Int} #list of allowed pairings of other spin categories
    idxs::Vector{Int}
    lookup::Array{Int, 3} #single spin lookup table for single excitations
    #list_idxs::Tuple{Vector{Int}, Vector{Int}}  #([alpha config idxs], [beta config idxs]) knows what configurations are in this spin category
    #lookup::Vector{Array{Int, 3}} #both alpha and beta lookup tables for single excitations
end

function Spin_Categories()
    return Spin_Categories(1,(0,0),Vector{Int}(), Vector{Int}(), Vector{Int}(), Array{Int, 3}())
    #return Spin_Categories(1,(1,1),Vector{Int}(), Vector{Int}(), Vector{Int}(), Vector{Array{Int, 3}}())
end

function Spin_Categories(idx::Int, hp::Tuple{Int,Int}, connected::Vector{Int})
    return Spin_Categories(idx, hp, Vector{Int}(), Vector{Int}(), connected, Array{Int, 3}())
end


function make_categories(prob::RASCIAnsatz; spin="alpha")
    categories = ActiveSpaceSolvers.RASCI.generate_spin_categories(prob)#={{{=#
    all_cats = Vector{Spin_Categories}()
    
    cats_a = deepcopy(categories)
    cats_b = deepcopy(categories)
    fock_list_a, del_at_a = make_fock_from_categories(categories, prob, "alpha")
    #println(fock_list_a)
    #println(cats_a)
    deleteat!(cats_a, del_at_a)
    len_cat_a = length(cats_a)
        
    fock_list_b, del_at_b = make_fock_from_categories(categories, prob, "beta")
    deleteat!(cats_b, del_at_b)
    len_cat_b = length(cats_b)

    if spin == "alpha"
        connected = make_spincategory_connections(cats_a, cats_b, prob)

        #compute configs
        as = compute_config_dict(fock_list_a, prob, "alpha")
        as_old =  ActiveSpaceSolvers.RASCI.compute_configs(prob)[1]
        rev_as = Dict(value => key for (key, value) in as)
        #this reverses the config dictionary to get the index as the key 
        for i in keys(rev_as)
            idx = as_old[i]
            rev_as[i] = idx
        end
        max_a = length(as)

        for j in 1:len_cat_a
            idxas = Vector{Int}()
            graph_a = make_cat_graphs(fock_list_a[j], prob)
            idxas = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_a, 1, graph_a.max, idxas, rev_as) 
            lu = zeros(Int, graph_a.no, graph_a.no, max_a)
            push!(all_cats, Spin_Categories(j, cats_a[j], connected[j], idxas, lu))
        end
        
        #have to do same loop as before bec all categories need initalized for the dfs search for lookup tables
        for k in 1:len_cat_a
            graph_a = make_cat_graphs(fock_list_a[k], prob)
            ActiveSpaceSolvers.RASCI.dfs_single_excitation!(graph_a, 1, graph_a.max, all_cats[k].lookup, all_cats, rev_as)
        end
        return all_cats

    else
        connected = make_spincategory_connections(cats_b, cats_a, prob)
        #compute configs
        bs = compute_config_dict(fock_list_b, prob, "beta")
        bs_old = ActiveSpaceSolvers.RASCI.compute_configs(prob)[2]
        rev_bs = Dict(value => key for (key, value) in bs)
        for i in keys(rev_bs)
            idx = bs_old[i]
            rev_bs[i] = idx
        end
        max_b = length(bs)
        
        for j in 1:len_cat_b
            idxbs = Vector{Int}()
            graph_b = make_cat_graphs(fock_list_b[j], prob)
            idxbs = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_b, 1, graph_b.max, idxbs, rev_bs) 
            lu = zeros(Int, graph_b.no, graph_b.no, max_b)
            push!(all_cats, Spin_Categories(j, cats_b[j], connected[j], idxbs, lu))
        end

        for k in 1:len_cat_b
            graph_b = make_cat_graphs(fock_list_b[k], prob)
            ActiveSpaceSolvers.RASCI.dfs_single_excitation!(graph_b, 1, graph_b.max, all_cats[k].lookup, all_cats, rev_bs)
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
            
            #println("K-idx: ", K_idx, " K_old: ", K_old)
            if K_idx != K_old
                println("nope")
            end

        end
    end
end#=}}}=#

function compute_config_dict(fock_list, prob::RASCIAnsatz, spin="alpha")
#function compute_config_dict(fock_list_a, fock_list_b, prob::RASCIAnsatz)
    configs_dict = Dict{Int, Vector{Int32}}()#={{{=#
    #configs_dict_b = Dict{Int, Vector{Int32}}()
    configs = []
    #configs_b = []
    for j in 1:length(fock_list)
        graph_a = ActiveSpaceSolvers.RASCI.make_cat_graphs(fock_list[j], prob)
        if spin == "alpha"
            config_dict = ActiveSpaceSolvers.RASCI.old_dfs(prob.na, graph_a.connect, graph_a.weights, 1, graph_a.max)
        else
            config_dict = ActiveSpaceSolvers.RASCI.old_dfs(prob.nb, graph_a.connect, graph_a.weights, 1, graph_a.max)
        end
        #graph_b = ActiveSpaceSolvers.RASCI.make_cat_graphs(fock_list_b[j], prob)
        #config_dict_b = ActiveSpaceSolvers.RASCI.old_dfs(prob.nb, graph_a.connect, graph_a.weights, 1, graph_a.max)
        for x in keys(config_dict)
            push!(configs, x)
        end
        
        #for y in keys(config_dict_b)
        #    push!(configs_b, y)
        #end
    end
    
    for i in 1:length(configs)
        configs_dict[i] = configs[i]
    end
    
    #for z in 1:length(configs_b)
    #    configs_dict_b[z] = configs_b[z]
    #end
    return configs_dict#=}}}=#
end


function generate_spin_categories(prob::RASCIAnsatz)
    #categories = [(ih, ip), (ih, ip)...]{{{
    #max_p = prob.ras3_max
    #max_h = prob.ras_spaces[1]-prob.ras1_min
    categories = []

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
    #max_p = prob.ras3_max*2
    #max_h = (prob.ras_spaces[1]-prob.ras1_min)*2
    #max_p = prob.ras3_max
    #max_h = prob.ras_spaces[1]-prob.ras1_min

    connected = Vector{Vector{Int}}()
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
    return connected
end

function make_category_connections(categories, prob::RASCIAnsatz)
    #max_p = prob.ras3_max#={{{=#
    #max_h = prob.ras_spaces[1]-prob.ras1_min

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
        #start = (prob.ras_spaces[1], prob.na-prob.ras_spaces[1], 0)
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
    graph = ActiveSpaceSolvers.RASCI.RASCI_OlsenGraph(prob.no, sum(fock_list), prob.ras_spaces, prob.ras1_min, prob.ras3_max, max_val, vert, connect, weights)
    #max_dim = maximum(full)
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


function find_cat(idx::Int, categories::Vector{Spin_Categories})
#function find_cat(idx::Int, categories::Vector{Spin_Categories}, spin=1)
    #spin = 1 is alpha
    #spin = 2 is beta
    #this function will find the category that idx belongs to
    for cat in categories
        if idx in cat.idxs
            return cat
        else
            continue
        end
    end
    return 0
end


function sigma_one(prob::RASCIAnsatz, cats_a::Vector{Spin_Categories}, cats_b::Vector{Spin_Categories}, ints::InCoreInts, v)
    T = eltype(v[1])#={{{=#
    n_roots::Int = size(v,3)
    #v = reshape(v, prob.dima, prob.dimb)
    sigma_one = zeros(prob.dima, prob.dimb, n_roots)
    
    F = zeros(T, prob.dimb)
    gkl = get_gkl(ints, prob) 
    
    sigma_one = permutedims(sigma_one,[1,3,2])
    v = permutedims(v,[1,3,2])
    #final_cat = Spin_Categories()

    for Ib in 1:prob.dimb
        #println("Config: ", Ib)
        cat = find_cat(Ib, cats_b)
        fill!(F,0.0)
        comb_kl = 0
        comb_ij = 0
        for k in 1:prob.no, l in 1:prob.no
            K_idx = cat.lookup[l,k,Ib]
            #K_idx = cat.lookup[k,l,Ib]
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
                #J_idx = next_cat.lookup[i,j,K]
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
        #ActiveSpaceSolvers.FCI._ss_sum!(sigma_one, v, F, Ib)
    end
    sigma_one = permutedims(sigma_one,[1,3,2])
    v = permutedims(v,[1,3,2])
    return sigma_one#=}}}=#
end

#function sigma_two(prob::RASCIAnsatz, categories::Vector{Spin_Categories}, ints::InCoreInts, v)
function sigma_two(prob::RASCIAnsatz, cats_a::Vector{Spin_Categories}, cats_b::Vector{Spin_Categories}, ints::InCoreInts, v)
    T = eltype(v[1])#={{{=#
    n_roots::Int = size(v,3)
    sigma_two = zeros(prob.dima, prob.dimb, n_roots)
    
    F = zeros(T, prob.dima)
    gkl = get_gkl(ints, prob) 
    
    sigma_two = permutedims(sigma_two,[2,3,1])
    v = permutedims(v,[2,3,1])
    #final_cat = Spin_Categories()
    
    for Ib in 1:prob.dima
        cat = find_cat(Ib, cats_a)
        fill!(F,0.0)
        comb_kl = 0
        comb_ij = 0
        for k in 1:prob.no, l in 1:prob.no
            K_idx = cat.lookup[l,k,Ib]
            #K_idx = cat.lookup[1][k,l,Ib]
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
                #J_idx = next_cat.lookup[1][i,j,K]
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

function slow_sigma_three(prob::RASCIAnsatz, cats_a::Vector{Spin_Categories}, cats_b::Vector{Spin_Categories}, ints::InCoreInts, v)
    
    T = eltype(v[1])
    n_roots::Int = size(v,3)
    sigma_three = zeros(Float64, prob.dima, prob.dimb,n_roots)
    
    hkl = zeros(Float64, prob.no, prob.no)
    count = 0
    countsd = 0
    combo = []

    for Ia in 1:prob.dima
        cat = find_cat(Ia, cats_a)
        fill!(hkl, T(0.0))
        count = 0
        #countsd = 0
        #println("Ia: ", cat.hp)
        for catb in cat.connected
            for Ib in cats_b[catb].idxs
                countsd+=1
                for k in 1:prob.no, l in 1:prob.no
                    Ja = cat.lookup[l,k,Ia]
                    #Ja = cat.lookup[k,l,Ia]
                    Ja != 0 || continue
                    sign_kl = sign(Ja)
                    Ja = abs(Ja)
                    hkl .= ints.h2[:,:,k,l]
                    cat_a = find_cat(Ja, cats_a)
                    #println("Ja: ", cat_a.hp)
                    #for Ib in 1:prob.dimb
                    #catb = find_cat(Ib, cats_b)
                    #println("Ib: ", catb.hp)
                    #    if catb.idx in cat.connected
                    for i in 1:prob.no, j in 1:prob.no
                        Jb = cats_b[catb].lookup[j,i,Ib]
                        #Jb = catb.lookup[j,i,Ib]
                        Jb != 0 || continue
                        sign_ij = sign(Jb)
                        Jb = abs(Jb)
                        #if Jb in cat_a.beta_idxs
                        final_catb = find_cat(Jb, cats_b)
                        if final_catb.idx in cat_a.connected
                            push!(combo, [Ia, Ib, Ja, Jb])
                            count+=1
                            for si in 1:n_roots
                                sigma_three[Ia, Ib, si] += hkl[i,j]*v[Ja, Jb, si]*sign_ij*sign_kl
                            end
                            #end
                        end
                    end
                end
            end

        end
    end
    println(countsd)
    println(count)
    return sigma_three
end

function sigma_three(prob::RASCIAnsatz, categories::Vector{Spin_Categories}, ints::InCoreInts, v)
    T = eltype(v[1])#={{{=#
    n_roots::Int = size(v,3)
    sigma_three = zeros(Float64, prob.dima, prob.dimb,n_roots)
    
    hkl = zeros(Float64, prob.no, prob.no)
    FJb = zeros(T, prob.dimb)
    #Ckl = Array{T, 3} 
    Ckl = zeros(T, prob.dima, prob.dimb, n_roots)
    #Ckl = zeros(T, prob.dima, prob.dimb, n_roots)
    
    sigma_three = permutedims(sigma_three,[1,3,2])
    #v = permutedims(v,[1,3,2])

    for k in 1:prob.no, l in 1:prob.no
        L = Vector{Int}()
        R = Vector{Int}()
        sign_I = Vector{Int8}()
        cat_list = Vector{Spin_Categories}()
        @inbounds fill!(Ckl, T(0.0))
        #loop over all alpha configs in cat
        for I in 1:prob.dima
            cat = find_cat(I, categories, 1)
            Iidx = cat.lookup[1][k,l,I]
            if Iidx != 0
                push!(R,I)
                push!(L,abs(Iidx))
                push!(sign_I, sign(Iidx))
                push!(cat_list, cat)
            end
        end
        
        #Ckl = zeros(T, length(L), prob.dimb, n_roots)
        
        #Gather
        _gather!(Ckl, v, R, L, sign_I)
        Ckl = permutedims(Ckl,[1,3,2])
        #println(size(Ckl))
        
        hkl .= ints.h2[:,:,k,l]
#VI = zeros(T, prob.dima, n_roots)
        #VI = zeros(T, length(L), n_roots)
        for Ib in 1:prob.dimb
            cat_b = find_cat(Ib, categories, 2)
            @inbounds fill!(FJb, T(0.0))
            for i in 1:prob.no, j in 1:prob.no
                Jb = cat_b.lookup[2][i,j,Ib]
                Jb != 0 || continue
                sign_ij = sign(Jb)
                Jb = abs(Jb)
                @inbounds FJb[Jb] += sign_ij*hkl[i,j]
            end

            #return Ckl, FJb, VI, Ib, cat_list

            _ras_mult!(sigma_three, Ckl, FJb, Ib, cat_list, categories)
            #_ras_mult!(Ckl, FJb, VI, Ib, cat_list, categories)

            #Scatter
            #_scatter!(sigma_three, R, VI, Ib)
        end
        Ckl = permutedims(Ckl,[1,3,2])
    end
    sigma_three = permutedims(sigma_three,[1,3,2])
    return sigma_three#=}}}=#
end

function _ras_ss_sum!(sig::Array{T,3}, v::Array{T,3}, F::Vector{T},Ib::Int, cats_a::Vector{Spin_Categories}, cats_b::Vector{Spin_Categories}; sigma="one") where {T}
    #spin refers to the fack that Ib, and final cat are of spin alpha or beta
    #Ib, Jb are the active spin
    #Ia is the idle spin that is vectorized over, these will enforce which Jb are allowed (active spin summations)
    #nIa     = size(v)[1]
    n_roots = size(v)[2]
    #nJb     = size(v)[3]
    count = 0
    #
    #
    
    nIa     = size(v)[1]
    n_roots = size(v)[2]
    nJb     = size(v)[3]
    if sigma == "one"
        #current_cat = find_cat(Ib, cats_b)
        for catb in cats_b    
            #if abs(F[Jb]) > 1e-14 
            for cats in catb.connected
                #if cats in current_cat.connected
                for Ia in cats_a[cats].idxs
                    for Jb in catb.idxs
                        @inbounds @simd for si in 1:n_roots
                            #count+= 1
                            sig[Ia,si,Ib] += F[Jb]*v[Ia,si,Jb]
                        end
                    end
                end
                #end
            end
        end
    else
        for cata in cats_a    
            #if abs(F[Jb]) > 1e-14 
            for cats in cata.connected
                for Ia in cats_b[cats].idxs
                    for Jb in cata.idxs
                        @inbounds @simd for si in 1:n_roots
                            #count+= 1
                            sig[Ia,si,Ib] += F[Jb]*v[Ia,si,Jb]
                        end
                    end
                end
                #end
            end
        end
    end
    #println("SD Count: ", count, "\n")
end


function _ras_ss_sum!(sig::Array{T,3}, v::Array{T,3}, F::Vector{T},Ib::Int, categories::Vector{Spin_Categories}; sigma="one") where {T}
    #spin refers to the fack that Ib, and final cat are of spin alpha or beta{{{
    #Ib, Jb are the active spin
    #Ia is the idle spin that is vectorized over, these will enforce which Jb are allowed (active spin summations)
    #nIa     = size(v)[1]
    n_roots = size(v)[2]
    #nJb     = size(v)[3]
    #count = 0
    
    if sigma == "one"
        #alpha
        for cats in categories
            #beta
            for i in cats.connected
                for Ia in cats.alpha_idxs
                    for Jb in categories[i].beta_idxs
                        if Ib in cats[i].beta_idxs
                            #count+=1
                            for si in 1:n_roots
                                sig[Ia,si,Ib] += F[Jb]*v[Ia,si,Jb]
                            end
                        end
                    end
                end
            end
        end
    
    else
        #beta
        for cats in categories
            #alpha
            for i in cats.connected
                for Ia in cats.beta_idxs #beta even though labeled Ia
                    for Jb in categories[i].alpha_idxs #alpha even though labeled Jb
                        if Ib in categories[i].alpha_idxs #current alpha
                            #count+=1
                            for si in 1:n_roots
                                sig[Ia,si,Ib] += F[Jb]*v[Ia,si,Jb]
                            end
                        end
                    end
                end
            end
        end
    end
    #println(count)}}}
end

function _gather!(Ckl::Array{T,3}, v, R::Vector{Int}, L::Vector{Int}, sign_I::Vector{Int8}) where {T}
#function _gather!(Ckl::Array{T,3}, v, R::Vector{Int}, L::Vector{Int}, sign_I::Vector{Int8}, categories::Vector{Spin_Categories}) where {T}
    nI = length(L)#={{{=#
    #na = size(v)[1]
    n_roots = size(v)[3]
    ket_max = size(v)[2]
    
    @inbounds @simd for si in 1:n_roots
        for Jb in 1:ket_max
            for Li in 1:nI
                Ckl[R[Li],Jb,si] = v[L[Li], Jb,si] * sign_I[Li]
            end
        end
    end


    #@inbounds @simd for si in 1:n_roots
    #    for Jb in 1:ket_max
    #        for Li in 1:nI
    #            Ckl[Li,Jb,si] = v[L[Li], Jb,si] * sign_I[Li]
    #        end
    #    end
    #end#=}}}=#
end


function _ras_mult!(sig::Array{T,3}, Ckl::Array{T,3}, FJb::Array{T,1}, Ib::Int, a_cat::Vector{Spin_Categories}, categories::Vector{Spin_Categories}) where {T}

#function _ras_mult!(Ckl::Array{T,3}, FJb::Array{T,1}, VI::Array{T,2}, Ib::Int, a_cats::Vector{Int8}, categories::Vector{Spin_Categories}) where {T}
    
    #length(a_cats) == size(Ckl,1) || throw(DimensionMismatch)
    #VI .= 0
    #nI = size(Ckl)[1]
    n_roots::Int = size(Ckl)[2]
    #ket_max = size(FJb)[1]
    #tmp = 0.0
    count = 0

    #L = list of alpha configurations
    #alpha cats
    #for cats in categories
    #    #beta
    #    for i in cats.connected
    #        for Ia in cats.list_idxs[1]
    #            for Jb in categories[i].list_idxs[2]
    #                #count+=1
    #                #if abs(tmp) > 1e-14
    #                for si in 1:n_roots
    #                    #println(size(sig))
    #                    #println(size(Ckl))
    #                    sig[Ia, si, Ib] += FJb[Jb]*Ckl[Ia,si,Jb]
    #                    #sigma_three[Ia, Ib, si] += F[Jb]*Ckl[Ia,Jb,si]
    #                end
    #                #end
    #            end
    #        end
    #    end
    #end
    
    #alpha
    for cats in a_cat
        #beta
        for i in cats.connected
            for Ia in cats.alpha_idxs
                for Jb in categories[i].beta_idxs
                    if Ib in categories[i].beta_idxs
                        #count+=1
                        #if abs(tmp) > 1e-14
                        for si in 1:n_roots
                            #println(size(sig))
                            #println(size(Ckl))
                            sig[Ia, si, Ib] += FJb[Jb]*Ckl[Ia,si,Jb]
                            #sigma_three[Ia, Ib, si] += F[Jb]*Ckl[Ia,Jb,si]
                        end
                    end
                end
            end
        end
    end
    #println(count)
end

"""
    compute_S2_expval(prb::RASCIAnsatz)
- `prb`: RASCIAnsatz just defines the current CI ansatz (i.e., ras_spaces sector)
"""
function compute_S2_expval(v::Matrix, P::RASCIAnsatz, all_cats::Vector{Spin_Categories})
    a_categories = ActiveSpaceSolvers.RASCI.make_categories(P, spin="alpha")
    b_categories = ActiveSpaceSolvers.RASCI.make_categories(P, spin="beta")

    #categories = ActiveSpaceSolvers.RASCI.generate_spin_categories(P)
    fock_list_a, del_at = make_fock_from_categories(categories, P, "alpha")
    fock_list_b, del_at = make_fock_from_categories(categories, P, "beta")
    deleteat!(categories, del_at)
    len_cat = length(categories)
    connected = make_category_connections(categories, P)
    as, bs = compute_config_dict(fock_list_a, fock_list_b, P)
    as_old =  ActiveSpaceSolvers.RASCI.compute_configs(P)[1]
    bs_old = ActiveSpaceSolvers.RASCI.compute_configs(P)[2]
    rev_as = Dict(value => key for (key, value) in as)
    rev_bs = Dict(value => key for (key, value) in bs)
    for i in keys(rev_as)
        idx = as_old[i]
        rev_as[i] = idx
    end
    
    for i in keys(rev_bs)
        idx = bs_old[i]
        rev_bs[i] = idx
    end

    as = Dict(value => key for (key, value) in rev_as)
    bs = Dict(value => key for (key, value) in rev_bs)
    
    max_a = length(as)
    max_b = length(bs)

    
    all_cats = Vector{Spin_Categories}()
    for cat in 1:len_cat
        x = Spin_Categories(cat, categories[cat], connected[cat])
        push!(all_cats, x)
    end

    for i in 1:length(fock_list_a)
        graph_a = make_cat_graphs(fock_list_a[i], P)
        idxas = Vector{Int}()
        idxbs = Vector{Int}()
        idxas = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_a, 1, graph_a.max, idxas, rev_as) 
        graph_b = make_cat_graphs(fock_list_b[i], P)

        idxbs = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_b, 1, graph_b.max, idxbs, rev_bs) 
        all_cats[i].alpha_idxs = idxas
        all_cats[i].beta_idxs = idxbs
    end
    
    for j in 1:len_cat
        graph_a = make_cat_graphs(fock_list_a[j], P)
        graph_b = make_cat_graphs(fock_list_b[j], P)
        #config_dict = old_dfs(prob.na, graph_a.connect, graph_a.weights, 1, graph_a.max)
        lua = find_paths_indexes(all_cats, graph_a, max_a, rev_as, 1);
        lub = find_paths_indexes(all_cats, graph_b, max_b, rev_bs, 2);
        all_cats[j].lookup = [lua, lub]
    end

    
    nr = size(v,2)#={{{=#
    s2 = zeros(nr)
    
    #a_configs = compute_configs(P)[1]
    #b_configs = compute_configs(P)[2]
    
    #fill single excitation lookup tables
    #a_lookup = fill_lookup(P, a_configs, P.dima)
    #b_lookup = fill_lookup(P, b_configs, P.dimb)
    #beta_graph = RASCI_OlsenGraph(P.no, P.nb+1, P.ras_spaces, P.ras1_min, P.ras3_max)
    #bra_graph = RASCI_OlsenGraph(P.no, P.nb, P.ras_spaces, P.ras1_min, P.ras3_max)
    
    #alpha
    for cats in all_cats
        #beta
        for i in cats.connected
            for Ia in cats.alpha_idxs
                for Ib in all_cats[i].beta_idxs
                    K = Ia + (Ib-1)*P.dima

                    #Sz.Sz
                    for ai in as[Ia]
                        for aj in as[Ia]
                            if ai!= aj
                                for r in 1:nr
                                    s2[r] += 0.25 * v[K,r]*v[K,r]
                                end
                            end
                        end
                    end

                    for bi in bs[Ib]
                        for bj in bs[Ib]
                            if bi != bj
                                for r in 1:nr
                                    s2[r] += 0.25 * v[K,r]*v[K,r]
                                end
                            end
                        end
                    end

                    for ai in as[Ia]
                        for bj in bs[Ib]
                            if ai != bj
                                for r in 1:nr
                                    s2[r] -= .5 * v[K,r]*v[K,r] 
                                end
                            end
                        end
                    end

                    #Sp.Sm
                    for ai in as[Ia]
                        if ai in bs[Ib]
                        else
                            for r in 1:nr
                                s2[r] += .75 * v[K,r]*v[K,r] 
                            end
                        end
                    end

                    #Sm.Sp
                    for bi in bs[Ib]
                        if bi in as[Ia]
                        else
                            for r in 1:nr
                                s2[r] += .75 * v[K,r]*v[K,r] 
                            end
                        end
                    end

                    for ai in as[Ia]
                        for bj in bs[Ib]
                            if ai ∉ bs[Ib]
                                if bj ∉ as[Ia]
                                    La = cats.lookup[1][ai, bj, Ia] 
                                    La != 0 || continue
                                    sign_a = sign(La)
                                    La = abs(La)
                                    cat_a = find_cat(La, all_cats, 1)
                                    

                                    #lookup table annhilates then creates but we need create then annhilate
                                    #Lb = b_lookup[bj, ai, Kb[2]]
                                    #Lb != 0 || continue
                                    #sign_b = sign(Lb)
                                    signb, conf, idx = apply_creation!(bs[Ib], ai, rev_bs, all_cats)
                                    conf != 0 || continue
                                    sign_b, conf_ann, Lb = apply_annhilation!(conf, bj, rev_bs, all_cats, 2)
                                    conf_ann != 0 || continue
                                    sign_b = sign_b*signb
                                    cat_b = find_cat(Lb, all_cats, 2)
                                    Lb = abs(Lb)
                                    if cat_b.idx in cat_a.connected
                                        L = abs(La) + (abs(Lb)-1)*P.dima
                                        for r in 1:nr
                                            s2[r] += sign_a * sign_b * v[K,r] * v[L,r]
                                        end
                                    end

                                end
                            end
                        end
                    end
                end
            end
        end
    end
    return s2#=}}}=#
end
    
#sig1 = reshape(sig1, prob.dima*prob.dimb, nroots)
#A = sig1[:,1]
#B = []
#for element = A
#    if element != 0
#        push!(B, element)
#    end
#end
#ras_dim(h,p,hp) == size(B)

        
                    

        
    
                    









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

function find_paths_indexes(categories::Vector{Spin_Categories}, graph::ActiveSpaceSolvers.RASCI.RASCI_OlsenGraph, max_dim::Int, config_dict::Dict{Vector{Int32}, Int}, lookup::Array{Int, 3})#={{{=#
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

    









