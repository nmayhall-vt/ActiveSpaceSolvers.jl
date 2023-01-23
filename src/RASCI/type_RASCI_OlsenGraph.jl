using StaticArrays

struct RASCI_OlsenGraph
    no::Int
    ne::Int
    spaces::SVector{3,Int}
    ras1_min::Int
    ras3_max::Int
    max::Int
    vert::Array{Int32}
    connect::Dict{Int32, Vector{Int32}}
    weights::Dict{Tuple{Int32, Int32}, Int32}
end

function RASCI_OlsenGraph()
    return RASCI_OlsenGraph(1,1,SVector(1,1,1), 1, 1, 1, Array{Int32}(undef, 0, 0), Dict{Int32, Vector{Int32}}(), Dict{Tuple{Int32, Int32}, Int32}())
end

function RASCI_OlsenGraph(no, ne, spaces, ras1_min=1, ras3_max=2)
    spaces = convert(SVector{3,Int},collect(spaces))
    x = make_ras_x(no, ne, spaces, ras1_min, ras3_max)
    y = make_ras_y(x)
    vert, max_val = make_vert_graph_ras(x)
    connect, weights = make_graph_dict(y, vert)
    return RASCI_OlsenGraph(no, ne, spaces, ras1_min, ras3_max, max_val, vert, connect, weights)
end

function calc_ndets(no, nelec, fock, ras1_min, ras3_max)
    x = make_ras_x(no, nelec, fock, ras1_min, ras3_max)
    dim_x = findmax(x)[1]
    return dim_x
end

function dfs_creation(graph::RASCI_OlsenGraph, n1::RASCI_OlsenGraph, start, max, lu, lus, must_obey::Bool, visited=Vector(zeros(max)), path=[])
    visited[start] = true#={{{=#
    push!(path, start)
    if start == max
        #get config,index, add to nodes dictonary
        index, config = get_index(graph.ne, path, graph.weights)

        for orb in 1:graph.no
            if orb in config
                continue
            else
                signa, conf = ActiveSpaceSolvers.RASCI.apply_creation!(config, orb, n1, must_obey)
                if conf == 0
                    continue
                end

                idxa = get_path_then_index(conf, n1)
                lu[index, orb] = idxa
                lus[index, orb] = signa
            end
        end
    else
        for i in graph.connect[start]
            if visited[i]==false
                dfs_creation(graph, n1, i,max, lu, lus, must_obey, visited, path)
            end
        end
    end

    #remove current vertex from path and mark as unvisited
    pop!(path)
    visited[start]=false
    return lu, lus#=}}}=#
end

function dfs_annhilation(graph::RASCI_OlsenGraph, n1::RASCI_OlsenGraph, start, max, lu, lus, must_obey::Bool, visited=Vector(zeros(max)), path=[])
    visited[start] = true#={{{=#
    push!(path, start)
    if start == max
        #get config,index, add to nodes dictonary
        index, config = get_index(graph.ne, path, graph.weights)
        for orb in config
            signa, conf = ActiveSpaceSolvers.RASCI.apply_annhilation!(config, orb, n1, must_obey)
            if conf == 0
                continue
            end
            idxa = get_path_then_index(conf, n1)
            lu[index, orb] = idxa
            lus[index, orb] = signa
        end
    else
        for i in graph.connect[start]
            if visited[i]==false
                dfs_annhilation(graph, n1, i,max, lu, lus, must_obey, visited, path)
            end
        end
    end

    #remove current vertex from path and mark as unvisited
    pop!(path)
    visited[start]=false
    return lu, lus#=}}}=#
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

function get_annhilation(no, ne, n0::RASCI_OlsenGraph, must_obey::Bool)
    if ne-1 < 0#={{{=#
        error("Can not apply annhiliation")
    end

    n1 = ActiveSpaceSolvers.RASCI.RASCI_OlsenGraph(no, ne-1, n0.spaces, 0, n0.spaces[3])
    dim = ActiveSpaceSolvers.RASCI.calc_ndets(no, ne, n1.spaces, n1.ras1_min, n1.ras3_max)
    lu = zeros(Int64, dim, no)
    lus = zeros(Int8, dim, no)
    lu, lus = dfs_annhilation(n0, n1, 1, n0.max, lu, lus, must_obey)
    return lu, lus#=}}}=#
end

function get_creation(no, ne, g_oneless::RASCI_OlsenGraph, g_org::RASCI_OlsenGraph, must_obey::Bool)
    #={{{=#
    n0 = RASCI_OlsenGraph(no, ne, g_oneless.spaces, g_oneless.ras1_min, g_oneless.ras3_max)

    if must_obey == true
        n1 = RASCI_OlsenGraph(no, ne+1, g_org.spaces, g_org.ras1_min, g_org.ras3_max)
    else
        n1 = RASCI_OlsenGraph(no, ne+1, g_org.spaces, 0, g_org.spaces[3])
    end
    
    dim = calc_ndets(no, ne, g_oneless.spaces, 0, g_oneless.spaces[3])
    lu = zeros(Int64, dim, no)
    lus = zeros(Int8, dim, no)
    lu, lus = dfs_creation(n0,  n1, 1, n0.max, lu, lus, must_obey)
    return lu, lus#=}}}=#
end

function fill_lu(norb::Int, nelec::Int, g::RASCI_OlsenGraph)
    if nelec > 0 #={{{=#
        a_lu, a_lus = get_annhilation(norb, nelec, g, false)
        g_oneless = RASCI_OlsenGraph(norb, nelec-1, g.spaces, 0, g.spaces[3])
        c_lu, c_lus = get_creation(norb, nelec-1, g_oneless, g, true)
    else
        return nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing
    end

    if nelec-1 > 0
        g_oneless = RASCI_OlsenGraph(norb, nelec-1, g.spaces, 0, g.spaces[3])
        aa_lu, aa_lus = get_annhilation(norb, nelec-1, g_oneless, false)
        g_twoless = RASCI_OlsenGraph(norb, nelec-2, g.spaces, 0, g.spaces[3])
        cc_lu, cc_lus = get_creation(norb, nelec-2, g_twoless, g, false)
    else
        return a_lu, a_lus, nothing, nothing, c_lu, c_lus, nothing, nothing
    end
    return a_lu, a_lus, aa_lu, aa_lus, c_lu, c_lus, cc_lu, cc_lus#=}}}=#
end

function make_ras_x(norbs, nelec, fock::SVector{3, Int}, ras1_min=1, ras3_max=2)
    n_unocc = (norbs-nelec)+1#={{{=#
    x = zeros(Int, n_unocc, nelec+1)

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
    #fock = (3,3,3)
    
    #RAS1
    h = fock[1]-ras1_min
    for spot in 1:h
        loc[1] += 1
        update_x!(x, loc)
    end
    p = fock[1]-h
    loc[2] += p

    #RAS2
    p2 = nelec-ras1_min-ras3_max
    h2 = fock[2] - p2
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
    h3 = fock[3] - ras3_max
    for spot in 1:h3
        loc[1] += 1
        #check
        if loc[1] > size(x)[1]
            return x
        else
            update_x!(x, loc) #updates everything at loc and to the right
        end
        #update_x!(x, loc) #updates everything at loc and to the right
    end#=}}}=#
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

function get_index(nelecs, path, weights)
    index = 1 #={{{=#
    config = Vector{Int8}(zeros(nelecs))
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

function dfs_ras(nelecs, graph, start, max, visited=Vector(zeros(max)), path=[], nodes=Dict())
    visited[start] = true#={{{=#
    push!(path, start)
    if start == max
        #get config,index, add to nodes dictonary
        index, config = get_index_ras(nelecs, path, graph)
        nodes[config] = index
    else
        for i in graph[start]
            if visited[i]==false
                dfs_ras(nelecs,graph,i,max,visited,path,nodes)
            end
        end
    end

    #remove current vertex from path and mark as unvisited
    pop!(path)
    visited[start]=false
    return nodes#=}}}=#
end

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

