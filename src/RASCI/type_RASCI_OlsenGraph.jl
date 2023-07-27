using StaticArrays

struct RASCI_OlsenGraph
    no::Int
    ne::Int
    spaces::SVector{3,Int}
    #ras1_min::Int
    #ras3_max::Int
    max::Int
    vert::Array{Int32}
    connect::Dict{Int32, Vector{Int32}}
    weights::Dict{Tuple{Int32, Int32}, Int32}
end

"""
    RASCI_OlsenGraph()

Creates an empty OlsenGraph type
"""
function RASCI_OlsenGraph()
    return RASCI_OlsenGraph(1,1,SVector(1,1,1), 1, 1, 1, Array{Int32}(undef, 0, 0), Dict{Int32, Vector{Int32}}(), Dict{Tuple{Int32, Int32}, Int32}())
end

"""
    RASCI_OlsenGraph(no, ne, spaces, ras1_min=0, ras3_max=spaces[3])

"""
function RASCI_OlsenGraph(no, ne, spaces, ras1_min=0, ras3_max=spaces[3])
    spaces = convert(SVector{3,Int},collect(spaces))
    x = make_ras_x(no, ne, spaces, ras1_min, ras3_max)
    y = make_ras_y(x)
    vert, max_val = make_vert_graph_ras(x)
    connect, weights = make_graph_dict(y, vert)
    return RASCI_OlsenGraph(no, ne, spaces, ras1_min, ras3_max, max_val, vert, connect, weights)
end

function calc_ndets(no, nelec, ras_spaces, ras1_min, ras3_max)
    x = make_ras_x(no, nelec, ras_spaces, ras1_min, ras3_max)
    dim_x = findmax(x)[1]
    return dim_x
end

"""
    old_dfs(nelecs, connect, weights, start, max, visited=Vector(zeros(max)), path=[], nodes=Dict{Vector{Int32}, Int64}())

Does a depth first search in the originial way I coded up, still in use
"""
function old_dfs(nelecs, connect, weights, start, max, visited=Vector(zeros(max)), path=[], nodes=Dict{Vector{Int32}, Int64}())
    # Returns a node dictionary where keys are configs and values are the indexes
    visited[start] = true#={{{=#
    push!(path, start)
    if start == max
        #get config,index, add to nodes dictonary
        index, config = get_index(nelecs, path, weights)
        #push!(idxs, index)
        #nodes[index] = config
        nodes[config] = index
    else
        for i in connect[start]
            if visited[i]==false
                old_dfs(nelecs, connect, weights,i,max,visited,path,nodes)
            end
        end
    end

    #remove current vertex from path and mark as unvisited
    pop!(path)
    visited[start]=false
    return nodes#=}}}=#
end

"""
    dfs_single_excitation(ket::RASCI_OlsenGraph, start, max, lu, cat_lu, categories, config_dict, visited=Vector(zeros(max)), path=[])

Does a depth-first search to find string configurations then applys a single excitation to fill the lookup table
"""
function dfs_single_excitation(ket::RASCI_OlsenGraph, start, max, lu, cat_lu, categories, config_dict, visited=Vector(zeros(max)), path=[])
    visited[start] = true#={{{=#
    push!(path, start)
    if start == max
        #get config,index, add to nodes dictonary
        index_dontuse, config = get_index(ket.ne, path, ket.weights)
        #index = config_dict[config]
        
        for orb in config
            for orb_c in 1:ket.no
                #if orb_c in config
                #    continue
                #end
                sgn, conf, idx_loc, idx = ActiveSpaceSolvers.RASCI.apply_single_excitation!(config, orb, orb_c, config_dict, categories)
                if conf == 0
                    continue
                end
                lu[orb, orb_c, idx_loc] = sgn*idx
                new_cat = find_cat(idx, categories)
                cat_lu[orb, orb_c, idx_loc] = new_cat.idx
            end
        end
    else
        for i in ket.connect[start]
            if visited[i]==false
                dfs_single_excitation(ket, i,max, lu, cat_lu, categories, config_dict, visited, path)
            end
        end
    end

    #remove current vertex from path and mark as unvisited
    pop!(path)
    visited[start]=false
    return lu, cat_lu#=}}}=#
end

"""
    dfs_a(ket::RASCI_OlsenGraph, start, max, lu, categories_ket, categories_bra, config_dict_ket, config_dict_bra, visited=Vector(zeros(max)), path=[])

Does a depth-first search to find string configurations then applys an annhilation operator to fill the lookup table
"""
function dfs_a(ket::RASCI_OlsenGraph, start, max, lu, categories_ket, categories_bra, config_dict_ket, config_dict_bra, visited=Vector(zeros(max)), path=[])
    visited[start] = true#={{{=#
    push!(path, start)
    if start == max
        #get config,index, add to nodes dictonary
        index_dontuse, config = get_index(ket.ne, path, ket.weights)
        #index = config_dict[config]
        
        for orb in config
            sgn, conf, idx_loc, idx = ActiveSpaceSolvers.RASCI.apply_a(config, orb, config_dict_ket, config_dict_bra, categories_ket, categories_bra)
            if conf == 0
                continue
            end
            lu[orb, idx_loc] = sgn*idx
        end
    else
        for i in ket.connect[start]
            if visited[i]==false
                dfs_a(ket, i,max, lu, categories_ket, categories_bra, config_dict_ket, config_dict_bra, visited, path)
            end
        end
    end

    #remove current vertex from path and mark as unvisited
    pop!(path)
    visited[start]=false
    return lu#=}}}=#
end

"""
    dfs_c(ket::RASCI_OlsenGraph, start, max, lu, categories_ket, categories_bra, config_dict_ket, config_dict_bra, visited=Vector(zeros(max)), path=[])

Does a depth-first search to find string configurations then applys a creation operator to fill the lookup table
"""
function dfs_c(ket::RASCI_OlsenGraph, start, max, lu, categories_ket, categories_bra, config_dict_ket, config_dict_bra, visited=Vector(zeros(max)), path=[])
    visited[start] = true#={{{=#
    push!(path, start)
    if start == max
        #get config,index, add to nodes dictonary
        index_dontuse, config = get_index(ket.ne, path, ket.weights)
        #index = config_dict[config]
        
        for orb_c in 1:ket.no
            if orb_c in config
                continue
            end
            sgn, conf, idx_loc, idx = ActiveSpaceSolvers.RASCI.apply_c(config, orb_c, config_dict_ket, config_dict_bra, categories_ket, categories_bra)
            if conf == 0
                continue
            end
            lu[orb_c, idx_loc] = sgn*idx
        end
    else
        for i in ket.connect[start]
            if visited[i]==false
                dfs_c(ket, i,max, lu, categories_ket, categories_bra, config_dict_ket, config_dict_bra, visited, path)
            end
        end
    end

    #remove current vertex from path and mark as unvisited
    pop!(path)
    visited[start]=false
    return lu#=}}}=#
end

"""
    dfs_ca(ket::RASCI_OlsenGraph, start, max, lu, categories_ket, categories_bra, config_dict_ket, config_dict_bra, visited=Vector(zeros(max)), path=[])

Does a depth-first search to find string configurations then applys a creation-annhilation pair of operators to fill the lookup table
"""
function dfs_ca(ket::RASCI_OlsenGraph, start, max, lu, categories_ket, categories_bra, config_dict_ket, config_dict_bra, visited=Vector(zeros(max)), path=[])
    visited[start] = true#={{{=#
    push!(path, start)
    if start == max
        index_dontuse, config = get_index(ket.ne, path, ket.weights)
        
        for orb in config
            for orb_c in 1:ket.no
                sgn, conf, idx_loc, idx = ActiveSpaceSolvers.RASCI.apply_ca(config, orb, orb_c, config_dict_ket, config_dict_bra, categories_ket, categories_bra)
                if conf == 0
                    continue
                end
                lu[orb, orb_c, idx_loc] = sgn*idx
            end
        end
    else
        for i in ket.connect[start]
            if visited[i]==false
                dfs_ca(ket, i,max, lu, categories_ket, categories_bra, config_dict_ket, config_dict_bra, visited, path)
            end
        end
    end

    #remove current vertex from path and mark as unvisited
    pop!(path)
    visited[start]=false
    return lu#=}}}=#
end

"""
    dfs_cc(ket::RASCI_OlsenGraph, start, max, lu, categories_ket, categories_bra, config_dict_ket, config_dict_bra, visited=Vector(zeros(max)), path=[])

Does a depth-first search to find string configurations then applys a creation-creation pair of operators to fill the lookup table
"""
function dfs_cc(ket::RASCI_OlsenGraph, start, max, lu, categories_ket, categories_bra, config_dict_ket, config_dict_bra, visited=Vector(zeros(max)), path=[])
    visited[start] = true#={{{=#
    push!(path, start)
    if start == max
        #get config,index, add to nodes dictonary
        index_dontuse, config = get_index(ket.ne, path, ket.weights)
        #index = config_dict[config]
        
        for orb_c in 1:ket.no
            if orb_c in config
                continue
            end
            for orb_cc in 1:ket.no
                if orb_cc in config && orb_cc == orb_c
                    continue
                end
                sgn, conf, idx_loc, idx = ActiveSpaceSolvers.RASCI.apply_cc(config, orb_c, orb_cc, config_dict_ket, config_dict_bra, categories_ket, categories_bra)
                if conf == 0
                    continue
                end
                lu[orb_c, orb_cc, idx_loc] = sgn*idx
            end
        end
    else
        for i in ket.connect[start]
            if visited[i]==false
                dfs_cc(ket, i,max, lu, categories_ket, categories_bra, config_dict_ket, config_dict_bra, visited, path)
            end
        end
    end

    #remove current vertex from path and mark as unvisited
    pop!(path)
    visited[start]=false
    return lu#=}}}=#
end

"""
    dfs_cca(ket::RASCI_OlsenGraph, start, max, lu, categories_ket, categories_bra, config_dict_ket, config_dict_bra, visited=Vector(zeros(max)), path=[])

Does a depth-first search to find string configurations then applys a creation-creation-annhilation set of operators to fill the lookup table
"""
function dfs_cca(ket::RASCI_OlsenGraph, start, max, lu, categories_ket, categories_bra, config_dict_ket, config_dict_bra, visited=Vector(zeros(max)), path=[])
    visited[start] = true#={{{=#
    push!(path, start)
    if start == max
        #get config,index, add to nodes dictonary
        index_dontuse, config = get_index(ket.ne, path, ket.weights)
        #index = config_dict[config]

        for orb_a in config
            for orb_c in 1:ket.no
                #orb_c and orb_a can be the same for diagonal
                #if orb_c in config
                #    continue
                #end
                for orb_cc in 1:ket.no
                    if orb_cc in config && orb_cc == orb_c
                        continue
                    end
                    sgn, conf, idx_loc, idx = ActiveSpaceSolvers.RASCI.apply_cca(config, orb_a, orb_c, orb_cc, config_dict_ket, config_dict_bra, categories_ket, categories_bra)
                    if conf == 0
                        continue
                    end
                    lu[orb_a, orb_c, orb_cc, idx_loc] = sgn*idx
                end
            end
        end
    else
        for i in ket.connect[start]
            if visited[i]==false
                dfs_cca(ket, i,max, lu, categories_ket, categories_bra, config_dict_ket, config_dict_bra, visited, path)
            end
        end
    end

    #remove current vertex from path and mark as unvisited
    pop!(path)
    visited[start]=false
    return lu#=}}}=#
end

"""
    dfs_ccaa(ket::RASCI_OlsenGraph, start, max, lu, categories_ket, categories_bra, config_dict_ket, config_dict_bra, visited=Vector(zeros(max)), path=[])

Does a depth-first search to find string configurations then applys a creation-creation-annhilation-annhilation set of operators to fill the lookup table
"""
function dfs_ccaa(ket::RASCI_OlsenGraph, start, max, lu, categories_ket, categories_bra, config_dict_ket, config_dict_bra, visited=Vector(zeros(max)), path=[])
    visited[start] = true#={{{=#
    push!(path, start)
    if start == max
        #get config,index, add to nodes dictonary
        index_dontuse, config = get_index(ket.ne, path, ket.weights)
        #index = config_dict[config]

        for orb_a in config
            for orb_aa in config
                if orb_aa != orb_a
                    for orb_c in 1:ket.no
                        #orb_c and orb_a can be the same for diagonal
                        #if orb_c in config
                        #    continue
                        #end
                        for orb_cc in 1:ket.no
                            #if orb_cc in config && orb_cc == orb_c
                            #    continue
                            #end
                            if orb_cc != orb_c
                                sgn, conf, idx_loc, idx = ActiveSpaceSolvers.RASCI.apply_ccaa(config, orb_a, orb_aa, orb_c, orb_cc, config_dict_ket, config_dict_bra, categories_ket, categories_bra)
                                if conf == 0
                                    continue
                                end
                                lu[orb_a, orb_aa, orb_c, orb_cc, idx_loc] = sgn*idx
                            end
                        end
                    end
                end
            end
        end
    else
        for i in ket.connect[start]
            if visited[i]==false
                dfs_ccaa(ket, i,max, lu, categories_ket, categories_bra, config_dict_ket, config_dict_bra, visited, path)
            end
        end
    end

    #remove current vertex from path and mark as unvisited
    pop!(path)
    visited[start]=false
    return lu#=}}}=#
end

"""
    dfs_fill_idxs(ket::RASCI_OlsenGraph, start, max, idxs, config_dict, visited=Vector(zeros(max)), path=[])

Used to filld idxs information during the creation of the HP_Category structs
"""
function dfs_fill_idxs(ket::RASCI_OlsenGraph, start, max, idxs, config_dict, visited=Vector(zeros(max)), path=[])
    visited[start] = true#={{{=#
    push!(path, start)
    if start == max
        #get config,index, add to nodes dictonary
        index_dontuse, config = get_index(ket.ne, path, ket.weights)
        index = config_dict[config]
        append!(idxs, index)
        
    else
        for i in ket.connect[start]
            if visited[i]==false
                dfs_fill_idxs(ket, i,max, idxs, config_dict, visited, path)
            end
        end
    end

    #remove current vertex from path and mark as unvisited
    pop!(path)
    visited[start]=false
    return idxs#=}}}=#
end

"""
    get_path_then_index(config, graph::RASCI_OlsenGraph)

Takes a configuration and converts it to a path taken through the olsen graph and returns a list of vertices
from the vert graph. Then, takes the path and computes the index for that configuration.

#Arguments
- `config`: Vector of occuppied orbitals for single configuration
- `graph`: OlsenGraph type

#Returns
- `index`: Int index from olsen graph, reverse-lexical ordering olsen graph
"""
function get_path_then_index(config, graph::RASCI_OlsenGraph)
    #get the path somehow then get index{{{
    start = [1,1]
    path = []
    push!(path, 1)
    move_down = Int
    orb_count = 0

    #check to see if config is empty
    if isempty(config) == false
        for orb in config
            move_down = orb-orb_count-1
            if move_down != 0
                for down in 1:move_down
                    orb_count += 1
                    start = [start[1]+1, start[2]]
                    value = graph.vert[start[1], start[2]]
                    push!(path, value)
                end
            end
            #move right
            start = [start[1], start[2]+1]
            value = graph.vert[start[1], start[2]]
            push!(path, value)
            orb_count += 1
            continue
        end

        #add remaining vertices that arent occupied to path
        orbs_left = graph.no-orb_count

        if orbs_left != 1
            #check if we need to move down or right
            if start[1] == size(graph.vert)[1]
                #move down
                for i in 1:orbs_left
                    start = [start[1], start[2]+1]
                    value = graph.vert[start[1], start[2]]
                    push!(path, value)
                end
            end

            if start[2] == size(graph.vert)[2]
                #move right
                for j in 1:orbs_left
                    start = [start[1]+1, start[2]]
                    value = graph.vert[start[1], start[2]]
                    push!(path, value)
                end
            end

        else
            push!(path, graph.max)
        end
        
        index = 1 
        count = 1
        for i in 1:length(path)-1
            if (path[i],path[i+1]) in keys(graph.weights)
                index += graph.weights[(path[i],path[i+1])]
            end
        end

    #config is empty
    else
        path = vec(graph.vert)
        index = 1
    end
    return index#=}}}=#
end

"""
    make_ras_x(norbs, nelec, ras_spaces::SVector{3, Int}, ras1_min=0, ras3_max=ras_spaces[3])

Makes x matrix in the GRMS method to help with indexing and finding configurations
"""
function make_ras_x(norbs, nelec, ras_spaces::SVector{3, Int}, ras1_min=0, ras3_max=ras_spaces[3])
    n_unocc = (norbs-nelec)+1#={{{=#
    x = zeros(Int, n_unocc, nelec+1)
    if ras1_min == 0 && ras3_max==ras_spaces[3]
        #do fci graph
        #fill first row and columns
        if size(x,2) != 0
            x[:,1] .= 1
        end

        if size(x,1) != 0
            x[1,:] .= 1
        end


        for i in 2:nelec+1
            for j in 2:n_unocc
                x[j, i] = x[j-1, i] + x[j, i-1]
            end
        end
        return x
    else
        if ras1_min == 0
            x[:,1] .= 1
        end


        if n_unocc == norbs+1
            x[:,1] .=1
            return x
        end


        #meaning if dim of prob  = 1, only one possible config
        if n_unocc == 1
            x[1,:].=1
            return x
        end

        x[1,:].=1
        loc = [1,1]
        #ras_spaces = (3,3,3)

        #RAS1
        if ras1_min == 0
            h = 1
        else
            h = ras_spaces[1]-ras1_min
        end
        for spot in 1:h
            loc[1] += 1
            update_x!(x, loc)
        end
        p = ras_spaces[1]-h
        loc[2] += p

        #RAS2
        p2 = nelec-ras1_min-ras3_max
        h2 = ras_spaces[2] - p2
        for spot in 1:h2
            loc[1] += 1
            #check
            if loc[1] > size(x)[1]
                return x
            else
                update_x!(x, loc) #updates everything at loc and to the right
            end
            #update_x!(x, loc) #updates everything at loc and to the right
        end
        loc[2] += p2


        #RAS3
        h3 = ras_spaces[3] - ras3_max
        if h3 == 0
            h3 = 1
        end

        for spot in 1:h3
            loc[1] += 1
            #check
            if loc[1] > size(x)[1]
                return x
            else
                update_x!(x, loc) #updates everything at loc and to the right
            end
        end#=}}}=#
    end
    return x
end

function update_x!(x, loc)
    row = loc[1]#={{{=#
    for column in loc[2]:size(x)[2]
        if column == 1
            x[row, column] = x[row-1, column]
        else
            x[row, column] = x[row-1, column] + x[row, column-1]
        end
    end#=}}}=#
end

"""
    make_ras_y(x)

Makes y matrix from x matrix in GRMS method
"""
function make_ras_y(x)
    y = x#={{{=#
    y = vcat(zeros(Int, (1, size(x)[2])), x)
    y = y[1:size(x)[1], :]
    for i in 1:size(y)[1]
        for j in 1:size(y)[2]
            if x[i,j] == 0
                y[i,j] = 0
            end
        end
    end
    return y#=}}}=#
end

"""
    make_graph_dict(y,vert)

Used in the older version of the depth first search algorithm
"""
function make_graph_dict(y,vert)
    connect = Dict{Int32, Vector{Int32}}() #key: node, value: ones its connected to{{{
    weights = Dict{Tuple{Int32, Int32}, Int32}()     #key: Tuple(node1, node2), value: arc weight between nodes 1 and 2
    for row in 1:size(y)[1]
        for column in 1:size(y)[2]
            #at last row and column
            if row==size(y)[1] && column==size(y)[2]
                return connect, weights
            
            #at non existent node (RAS graphs)
            elseif vert[row,column] == 0
                continue
            
            #at last row or no node present (RAS graphs)
            elseif row == size(y)[1] || vert[row+1,column]==0
                connect[vert[row,column]]=[vert[row,column+1]]
                #check if moving right is a node (RAS graphs)
                if vert[row,column+1] != 0
                    weights[vert[row,column], vert[row,column+1]] = y[row, column+1]
                end

            #at last column or no node present (RAS graphs)
            elseif column == size(y)[2] || vert[row,column+1]==0
                connect[vert[row,column]]=[vert[row+1, column]]
            

            else
                connect[vert[row,column]]=[vert[row,column+1],vert[row+1,column]]
                #check if moving right is a node (RAS graphs)
                if vert[row,column+1] != 0
                    weights[vert[row,column],vert[row,column+1]] = y[row,column+1]
                end
            end
        end
    end
    return connect, weights#=}}}=#
end

"""
    make_vert_graph_ras(x)

Used in the older version of the depth-first search algorithm
"""
function make_vert_graph_ras(x)
    vert = Array{Int16}(zeros(size(x)))#={{{=#
    count = 1
    for row in 1:size(x)[1]
        for column in 1:size(x)[2]
            if x[row,column] != 0
                vert[row,column] = count
                count += 1
            end
        end
    end
    max_val = findmax(vert)[1]
    return vert, max_val#=}}}=#
end

"""
    get_index(nelecs, path, weights)

"""
function get_index(nelecs, path, weights)
    index = 1 #={{{=#
    config = Vector{Int32}(zeros(nelecs))
    count = 1

    for i in 1:length(path)-1
        if (path[i],path[i+1]) in keys(weights)
            index += weights[(path[i],path[i+1])]
            config[count]=i
            count += 1
        end
    end
    return index, config#=}}}=#
end

# Returns a node dictionary where keys are configs and values are the indexes
"""
    make_ras_dict(y,vert)

"""
function make_ras_dict(y,vert)
    graph = Dict()#={{{=#
    for row in 1:size(y)[1]
        for column in 1:size(y)[2]
            #at last row and column
            if row==size(y)[1] && column==size(y)[2]
                return graph
            
            #at non existent node (RAS graphs)
            elseif vert[row,column] == 0
                continue
            
            #at last row or no node present (RAS graphs)
            elseif row == size(y)[1] || vert[row+1,column]==0
                graph[vert[row,column]]=Set([vert[row,column+1]])
                #check if moving right is a node (RAS graphs)
                if vert[row,column+1] != 0
                    graph[vert[row,column],vert[row,column+1]] = y[row,column+1]
                end

            #at last column or no node present (RAS graphs)
            elseif column == size(y)[2] || vert[row,column+1]==0
                graph[vert[row,column]]=Set([vert[row+1, column]])

            else
                graph[vert[row,column]]=Set([vert[row,column+1],vert[row+1,column]])
                #check if moving right is a node (RAS graphs)
                if vert[row,column+1] != 0
                    graph[vert[row,column],vert[row,column+1]] = y[row,column+1]
                end
            end
        end
    end
    #max = findmax(ras_vert)[1]
    #println("max: ", max)
    return graph#=}}}=#
end

"""
    get_index_ras(nelecs, path, graph)

"""
function get_index_ras(nelecs, path, graph)
    index = 1#={{{=#
    config = Vector{Int}(zeros(nelecs))
    count = 1
    for i in 1:length(path)-1
        if (path[i],path[i+1]) in keys(graph)
            index += graph[(path[i],path[i+1])]
            config[count]=i
            count += 1
        end
    end
    return index, config#=}}}=#
end

