using JLD2
using InCoreIntegrals
using StaticArrays


struct Spin_Pair
    pair::Tuple{Int, Int} #(alpha HP category, beta HP category)
    ashift::Int #shift of dim of Spin Pairs that come before 
    bshift::Int #shift of dim of Spin Pairs that come before 
    dim::Int
end

function make_spin_pairs(prob::RASCIAnsatz)
    spin_pairs = Vector{Spin_Pair}()
    a_categories = ActiveSpaceSolvers.RASCI.make_categories(prob, spin="alpha")
    b_categories = ActiveSpaceSolvers.RASCI.make_categories(prob, spin="beta")
    ashift = 0
    for i in 1:length(a_categories)
        dima = length(a_categories[i].idxs)
        for j in a_categories[i].connected 
            bshift = 0
            if j >= 2
                for m in 1:j-1
                    bshift += length(b_categories[m].idxs)
                end
            end


            dimb = length(b_categories[j].idxs)
            x = Spin_Pair((i, j), ashift, bshift, dima*dimb)
            push!(spin_pairs, x)
        end
        ashift += dima
    end
    return spin_pairs, a_categories, b_categories
end

function make_spin_pairs(prob::RASCIAnsatz, a_categories::Vector{<:HP_Category}, b_categories::Vector{<:HP_Category})
    spin_pairs = Vector{Spin_Pair}()
    ashift = 0
    for i in 1:length(a_categories)
        dima = length(a_categories[i].idxs)
        for j in a_categories[i].connected 
            bshift = 0
            if j >= 2
                for m in 1:j-1
                    bshift += length(b_categories[m].idxs)
                end
            end

            dimb = length(b_categories[j].idxs)
            x = Spin_Pair((i, j), ashift, bshift, dima*dimb)
            push!(spin_pairs, x)
        end
        ashift += dima
    end
    return spin_pairs
end

function find_spin_pair(spin_pairs::Vector{Spin_Pair}, current::Tuple{Int,Int})
    for i in 1:length(spin_pairs)
        if spin_pairs[i].pair == current 
            return i
        else
            continue
        end
    end
    return 0
end

function find_spin_pair(spin_pairs::Vector{Spin_Pair}, current::Int, spin="beta")
    if spin == "beta"
        for i in 1:length(spin_pairs)
            if spin_pairs[i].pair[2] == current
                return i
            else
                continue
            end
        end
    else
        for j in 1:length(spin_pairs)
            if spin_pairs[j].pair[1] == current
                return j
            else
                continue
            end
        end
    end
end


