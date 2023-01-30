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
#using RDM

"""
    build_H_matrix(ints::InCoreInts, p::RASCIAnsatz)
"""
function build_H_matrix(ints::InCoreInts, p::RASCIAnsatz)
    Hmat = zeros(p.dim, p.dim)#={{{=#
    #if closed shell only compute single spin
    if p.na == p.nb 
        a_configs = compute_configs(p)[1]
        b_configs = compute_configs(p)[2]

        #fill single excitation lookup tables
        #a_lookup = fill_lookup(p, a_configs, p.dima)
        #b_lookup = fill_lookup(p, b_configs, p.dimb)
        
        a_lookup_ov = fill_lookup_ov(p, a_configs, p.dima)
        b_lookup_ov = fill_lookup_ov(p, b_configs, p.dimb)
        
        #compute diag terms of sigma 1 and sigma 2
        Ha_diag = precompute_spin_diag_terms(a_configs, p.na, p.dima, ints)
        Hb_diag = precompute_spin_diag_terms(b_configs, p.nb, p.dimb, ints)
        
        #compute off diag terms of sigma1 and sigma2
        Ha = compute_ss_terms_full(a_configs, a_lookup_ov, p.dima, p.no, p.na, ints) + Ha_diag
        Hb = compute_ss_terms_full(b_configs, b_lookup_ov, p.dimb, p.no, p.nb, ints) + Hb_diag
        
        Hmat .+= kron(SMatrix{p.dimb, p.dimb, UInt8}(Matrix{UInt8}(I,p.dimb,p.dimb)),  Ha)
        Hmat .+= kron(Hb, SMatrix{p.dima, p.dima, UInt8}(Matrix{UInt8}(I,p.dima,p.dima)))
        
        #compute both diag and off diag terms for sigma3 (mixed spin sigma)
        Hmat .+= compute_ab_terms_full(ints, p, a_configs, b_configs, a_lookup_ov, b_lookup_ov)
        #Nick has the following in his build_H_matrix function??
        #Hmat = .5*(Hmat+Hmat')
        #Hmat += 1.0I*ints.h0
    
    #if open shell must compute alpha and beta separately
    else 
        a_configs = compute_configs(p)[1]
        b_configs = compute_configs(p)[2]
    
        #fill single excitation lookup tables
        #a_lookup = fill_lookup(p, a_configs, p.dima)
        #b_lookup = fill_lookup(p, b_configs, p.dimb)
        a_lookup_ov = fill_lookup_ov(p, a_configs, p.dima)
        b_lookup_ov = fill_lookup_ov(p, b_configs, p.dimb)

        #compute diag terms of sigma 1 and sigma 2
        Ha_diag = precompute_spin_diag_terms(a_configs, p.na, p.dima, ints)
        Hb_diag = precompute_spin_diag_terms(b_configs, p.nb, p.dimb, ints)
        
        #compute off diag terms of sigma1 and sigma2
        Ha = compute_ss_terms_full(a_configs, a_lookup_ov, p.dima, p.no, p.na, ints) + Ha_diag
        Hb = compute_ss_terms_full(b_configs, b_lookup_ov, p.dimb, p.no, p.nb, ints) + Hb_diag
        
        Hmat .+= kron(SMatrix{p.dimb, p.dimb, UInt8}(Matrix{UInt8}(I,p.dimb,p.dimb)),  Ha)
        Hmat .+= kron(Hb, SMatrix{p.dima, p.dima, UInt8}(Matrix{UInt8}(I,p.dima,p.dima)))
        
        #compute both diag and off diag terms for sigma3 (mixed spin sigma)
        Hmat .+= compute_ab_terms_full(ints, p, a_configs, b_configs, a_lookup_ov, b_lookup_ov)
        
        #Nick has the following in his build_H_matrix function??
        #Hmat = .5*(Hmat+Hmat')
        #Hmat += 1.0I*ints.h0
    end#=}}}=#
    return Hmat
end

"""
    precompute_spin_diag_terms(configs, nelecs, dim, ints::InCoreInts)
"""
function precompute_spin_diag_terms(configs, nelecs, dim, ints::InCoreInts)
    Hout = zeros(dim, dim)#={{{=#
    for I in configs
        config = I[1]
        idx = I[2]
        for i in 1:nelecs
            Hout[idx, idx] += ints.h1[config[i], config[i]]
            for j in i+1:nelecs
                Hout[idx,idx] += ints.h2[config[i], config[i], config[j], config[j]]
                Hout[idx,idx] -= ints.h2[config[i], config[j], config[i], config[j]]
            end
        end
    end#=}}}=#
    return Hout
end

"""
    compute_ss_terms_full(configs, lookup, dim, norbs, nelecs, ints::InCoreInts)
"""
function compute_ss_terms_full(configs, lookup, dim, norbs, nelecs, ints::InCoreInts)
    Ha = zeros(dim, dim)#={{{=#
    F = zeros(dim)
    for I in configs
        fill!(F, 0.0)
        config = I[1]
        I_idx = I[2]
        #orbs = [1:norbs;]
        vir = filter!(x->!(x in config), [1:norbs;])
        
        #single excitation
        for k in config
            for l in vir
                single_idx = lookup[k,l,I_idx]
                if single_idx == 0
                    continue
                end
                sign_s = sign(single_idx)
                F[abs(single_idx)] += ints.h1[k,l]*sign_s
                for m in config
                    if m!=k
                        F[abs(single_idx)] += sign_s*(ints.h2[k,l,m,m]-ints.h2[k,m,l,m])
                    end
                end
            end
        end
        
        #double excitation
        for k in config
            for i in config
                if i>k
                    for l in vir
                        for j in vir
                            if j>l
                                single, sorted_s, sign_s = excit_config(config, k,l)
                                double, sorted_d, sign_d = excit_config(sorted_s, i,j)
                                #idx = configs[sorted_d]
                                if haskey(configs, sorted_d)
                                    idx = configs[sorted_d]
                                else
                                    continue
                                end
                                if sign_d == sign_s
                                    @inbounds F[idx] += (ints.h2[i,j,k,l] - ints.h2[i,l,j,k]) 

                                else
                                    @inbounds F[idx] -= (ints.h2[i,j,k,l] - ints.h2[i,l,j,k]) 
                                end
                            end
                        end
                    end
                end
            end
        end
        Ha[:,I_idx] .= F
    end#=}}}=#
    return Ha
end

"""
    get_gkl(ints::InCoreInts, prob::RASCIAnsatz)
"""
function get_gkl(ints::InCoreInts, prob::RASCIAnsatz)
    hkl = zeros(prob.no, prob.no)#={{{=#
    hkl .= ints.h1
    gkl = zeros(prob.no, prob.no)
    for k in 1:prob.no
        for l in 1:prob.no
            gkl[k,l] += hkl[k,l]
            x = 0
            for j in 1:prob.no
                if j < k
                    x += ints.h2[k,j,j,l]
                end
            end
            gkl[k,l] -= x
            if k >= l 
                if k == l
                    delta = 1
                else
                    delta = 0
                end
                gkl[k,l] -= ints.h2[k,k,k,l]*1/(1+delta)
            end
        end
    end#=}}}=#
    return gkl
end

"""
    compute_sigma_one(b_configs, b_lookup, ci_vector, ints::InCoreInts, prob::RASCIAnsatz)
"""
function compute_sigma_one(b_configs, b_lookup, ci_vector, ints::InCoreInts, prob::RASCIAnsatz)
    ## bb σ1(Iα, Iβ){{{
    sigma_one = zeros(prob.dima, prob.dimb)
    ci_vector = reshape(ci_vector, prob.dima, prob.dimb)
    
    F = zeros(prob.dimb)
    gkl = get_gkl(ints, prob) 
    
    for I_b in b_configs
        I_idx = I_b[2]
        I_config = I_b[1]
        fill!(F,0.0)
        comb_kl = 0
        comb_ij = 0
        ##single excitation
        for k in 1:prob.no
            for l in 1:prob.no
                K_idx = b_lookup[l,k,I_idx]
                #K_idx = b_lookup[k,l,I_idx]
                if K_idx == 0
                    continue
                end
                sign_kl = sign(K_idx)
                K = abs(K_idx)
               @inbounds F[K] += sign_kl*gkl[k,l]
                comb_kl = (k-1)*prob.no + l
                
                #double excitation
                for i in 1:prob.no
                    for j in 1:prob.no
                        comb_ij = (i-1)*prob.no + j
                        if comb_ij < comb_kl
                            continue
                        end

                        J_idx = b_lookup[j,i,K]
                        #J_idx = b_lookup[i,j,K]
                        if J_idx == 0 
                            continue
                        end

                        sign_ij = sign(J_idx)
                        J = abs(J_idx)
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
            end
        end
        
        #for a in 1:prob.dima
        #    for b in 1:prob.dimb
        #        @inbounds sigma_one[a, I_idx] += F[b]*ci_vector[a,b]
        #    end
        #end
        
        scr = ci_vector*F
        sigma_one[:, I_idx] .+= scr
    end#=}}}=#
    return sigma_one
end

"""
    compute_sigma_two(a_configs, a_lookup, ci_vector, ints::InCoreInts, prob::RASCIAnsatz)
"""
function compute_sigma_two(a_configs, a_lookup, ci_vector, ints::InCoreInts, prob::RASCIAnsatz)
    ## aa σ2(Iα, Iβ){{{
    sigma_two = zeros(prob.dima, prob.dimb)
    ci_vector = reshape(ci_vector, prob.dima, prob.dimb)
    
    F = zeros(prob.dima)
    gkl = get_gkl(ints, prob) 
    
    for I_a in a_configs
        I_idx = I_a[2]
        I_config = I_a[1]
        fill!(F,0.0)
        comb_kl = 0
        comb_ij = 0

        ##single excitation
        for k in 1:prob.no
            for l in 1:prob.no
                K_idx = a_lookup[l,k,I_idx]
                #K_idx = a_lookup[k,l,I_idx]
                if K_idx == 0
                    continue
                end
                sign_kl = sign(K_idx)
                K = abs(K_idx)
                @inbounds F[K] += sign_kl*gkl[k,l]
                comb_kl = (k-1)*prob.no + l
                
                #double excitation
                for i in 1:prob.no
                    for j in 1:prob.no
                        comb_ij = (i-1)*prob.no + j
                        if comb_ij < comb_kl
                            continue
                        end

                        J_idx = a_lookup[j,i,K]
                        #J_idx = a_lookup[i,j,K]
                        if J_idx == 0 
                            continue
                        end

                        sign_ij = sign(J_idx)
                        J = abs(J_idx)

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
            end
        end
    
        #for a in 1:prob.dima
        #    for b in 1:prob.dimb
        #        @inbounds sigma_two[I_idx,b] += F[a]*ci_vector[a,b]
        #    end
        #end

        #scr = zeros(prob.dimb)
        #@tensor begin
        #    scr[b] = F[a]*ci_vector[a,b]
        #end
        #sigma_two[I_idx,:] .+= scr

        scr = F'*ci_vector
        sigma_two[I_idx,:] .+= scr'
    end#=}}}=#
    return sigma_two
end

"""
    compute_sigma_three(a_configs, b_configs, a_lookup, b_lookup, ci_vector, ints::InCoreInts, prob::RASCIAnsatz)
"""
function compute_sigma_three(a_configs, b_configs, a_lookup, b_lookup, ci_vector, ints::InCoreInts, prob::RASCIAnsatz)
    sigma_three = zeros(prob.dima, prob.dimb)#={{{=#
    #ci_vector = reshape(ci_vector, prob.dimb, prob.dima)
    ci_vector = reshape(ci_vector, prob.dima, prob.dimb)
    
    F = zeros(prob.dimb)
    hkl = zeros(prob.no, prob.no)
    
    for k in 1:prob.no
        for l in 1:prob.no
            L = Vector{Int32}()
            R = Vector{Int32}()
            sign_I = Vector{Int32}()
            # compute all k->l excitations of alpha ket
            # config R -> config L with a k->l excitation
            for I in a_configs
                Iidx = a_lookup[k,l,I[2]]
                if Iidx != 0
                    push!(R,I[2])
                    push!(L,abs(Iidx))
                    push!(sign_I, sign(Iidx))
                end
            end

            #Gather
            # .* is vectorized multiplication
            #Ckl = ci_vector[:, L]
            #Ckl_prime = Ckl .* sign_I'
            Ckl = ci_vector[L, :]
            Ckl_prime = sign_I .* Ckl
            #@views hkl = ints.h2[:,:,k,l]

            hkl .= ints.h2[:,:,k,l]
            
            #look over beta configs
            for Ib in b_configs

                fill!(F,0.0)
                #@code_warntype _update_F!(hkl, F, b_lookup, Ib, prob)
                #error("here")
                #@views hkl = ints.h2[:,:,k,l]
                _update_F!(hkl, F, b_lookup, Ib[2], prob)
                
                #VI = Ckl*F.*sign_I
                #VI = Ckl_prime'*F
                VI = Ckl_prime*F
                
                sigma_three[R, Ib[2]] .+= VI
                
                #Scatter
                #for Li in 1:length(VI)
                #    sigma_three[R[Li], Ib[2]] += VI[Li]
                #end
            end
        end
    end
#    @save "src/data/h8_sigma3_old.jld2" sigma_three
    return sigma_three#=}}}=#
end

function _update_F!(hkl, F, b_lookup::Array{Int,3}, Ib, prob)
    for j in 1:prob.no#={{{=#
        #jkl_idx = j-1 + (k-1)*prob.no + (l-1)*prob.no*prob.no 
        for i in 1:prob.no

            #ijkl_idx = (i-1) + jkl_idx*prob.no + 1

            Jb = b_lookup[i,j,Ib]
            if Jb == 0
                continue
            end
            sign_ij = sign(Jb)
            J = abs(Jb)
            @inbounds F[J] += sign_ij*hkl[i,j]
            #@inbounds F[J] += sign_ij*ints.h2[ijkl_idx]
            #F[J] += sign_ij*ints.h2[i,j,k,l]
        end
    end#=}}}=#
end

"""
    compute_ab_terms_full(ints::InCoreInts, prob::RASCIAnsatz, a_configs, b_configs, a_lookup, b_lookup)
"""
function compute_ab_terms_full(ints::InCoreInts, prob::RASCIAnsatz, a_configs, b_configs, a_lookup, b_lookup)
    Hmat = zeros(prob.dim, prob.dim)#={{{=#
    
    for Ka in a_configs
        Ka_idx = Ka[2]
        Ka_config = Ka[1]
        orbsa = [1:prob.no;]
        vira = filter!(x->!(x in Ka_config), orbsa)
        for Kb in b_configs
            Kb_idx = Kb[2]
            Kb_config = Kb[1]
            orbsb = [1:prob.no;]
            virb = filter!(x->!(x in Kb_config), orbsb)
            K = Ka_idx + (Kb_idx-1)*prob.dima
            
            #diagonal part
            for l in Kb_config
                for n in Ka_config
                    Hmat[K, K] += ints.h2[n,n,l,l]
                end
            end
            
            #excit alpha only
            for p in Ka_config
                for q in vira
                    #a_single, sort_a, sign_a = excit_config(Ka_config, p,q)
                    idxa = a_lookup[p,q,Ka_idx]
                    if idxa == 0
                        continue
                    end

                    sign_a = sign(idxa)
                    Kprime = abs(idxa) + (Kb_idx-1)*prob.dima
                    #alpha beta <ii|jj>
                    for m in Kb_config
                        Hmat[K,Kprime]+=sign_a*ints.h2[p,q,m,m]
                    end
                end
            end

            #excit beta only
            for r in Kb_config
                for s in virb
                    #b_single, sort_b, sign_b = excit_config(Kb_config, r,s)
                    idxb = b_lookup[r,s,Kb_idx]
                    if idxb == 0
                        continue
                    end

                    sign_b = sign(idxb)
                    Lprime = Ka_idx + (abs(idxb)-1)*prob.dima
                    
                    #alpha beta <ii|jj>
                    for n in Ka_config
                        Hmat[K,Lprime]+=sign_b*ints.h2[r,s,n,n]
                    end
                end
            end

            #excit alpha and beta
            for p in Ka_config
                for q in vira
                    #a_single, sort_a, sign_a = excit_config(Ka_config, p,q)
                    idxa = a_lookup[p,q,Ka_idx]
                    if idxa == 0
                        continue
                    end
                    sign_a = sign(idxa)
                    for r in Kb_config
                        for s in virb
                            #b_single, sort_b, sign_b = excit_config(Kb_config, r,s)
                            idxb = b_lookup[r,s,Kb_idx]
                            if idxb == 0
                                continue
                            end
                            sign_b = sign(idxb)
                            L = abs(idxa) + (abs(idxb)-1)*prob.dima
                            Hmat[K,L] += sign_a*sign_b*(ints.h2[p,q,r,s])
                        end
                    end
                end
            end
        end
    end#=}}}=#
    return Hmat#=}}}=#
end

"""
    fill_lookup(prob::RASCIAnsatz, configs, dim_s)
"""
function fill_lookup(prob::RASCIAnsatz, configs, dim_s)
    lookup_table = zeros(Int64,prob.no, prob.no, dim_s)#={{{=#
    orbs = [1:prob.no;]
    ras1, ras2, ras3 = make_rasorbs(prob.fock[1], prob.fock[2], prob.fock[3], prob.no)
    for i in configs
        vir = filter!(x->!(x in i[1]), [1:prob.no;])
        for p in i[1]
            for q in 1:prob.no
                if p == q
                    lookup_table[p,q,i[2]] = i[2]
                end

                if q in vir
                    new_config, sorted_config, sign_s = excit_config(i[1], p, q)
                    
                    #CHECK POINT for excitation
                    ras1_test = length(findall(in(sorted_config),ras1))
                    if ras1_test < prob.ras1_min
                        continue
                    end

                    ras3_test = length(findall(in(sorted_config),ras3))
                    if ras3_test > prob.ras3_max
                        continue
                    end
                    
                    idx = configs[new_config]
                    lookup_table[p,q,i[2]] = sign_s*idx
                end
            end
        end
    end#=}}}=#
    return lookup_table
end

function fill_lookup_ov(prob::RASCIAnsatz, configs, dim_s)
    lookup_table = zeros(Int64,prob.no, prob.no, dim_s)#={{{=#
    orbs = [1:prob.no;]
    ras1, ras2, ras3 = make_rasorbs(prob.fock[1], prob.fock[2], prob.fock[3], prob.no)
    for i in configs
        vir = filter!(x->!(x in i[1]), [1:prob.no;])
        for p in i[1]
            for q in vir
                new_config, sorted_config, sign_s = excit_config(i[1], p, q)
                
                #CHECK POINT for excitation
                ras1_test = length(findall(in(sorted_config),ras1))
                if ras1_test < prob.ras1_min
                    continue
                end

                ras3_test = length(findall(in(sorted_config),ras3))
                if ras3_test > prob.ras3_max
                    continue
                end

                idx = configs[sorted_config]
                lookup_table[p,q,i[2]] = sign_s*idx
            end
        end
    end#=}}}=#
    return lookup_table
end

"""
    excit_config(config, i, j)
"""
function excit_config(config, i, j)
    #apply annhilation/creation operator pair to the string{{{
    #get new index of string and store in lookup table
    #config is orbital indexing in ascending order
    #positions [i,a] meaning electron in orb_i went to orb_a
    #config = SVector
    
    spot = first(findall(x->x==i, config))
    new = Vector(config)
    new[spot] = j
    count, arr = bubble_sort(new)
    if iseven(count)
        sign = 1
    else
        sign = -1
    end#=}}}=#
    return new, arr, sign
end

"""
    bubble_sort(arr)
"""
function bubble_sort(arr)
    len = length(arr) #={{{=#
    count = 0
    # Traverse through all array elements
    for i = 1:len-1
        for j = 2:len
        # Last i elements are already in place
        # Swap if the element found is greater
            if arr[j-1] > arr[j] 
                count += 1
                tmp = arr[j-1]
                arr[j-1] = arr[j]
                arr[j] = tmp
            end
        end
    end
    return count, arr#=}}}=#
end
    
"""
    make_rasorbs(rasi_orbs, rasiii_orbs, norbs, frozen_core=false)
"""
function make_rasorbs(rasi_orbs, rasii_orbs, rasiii_orbs, norbs, frozen_core=false)
    if frozen_core==false#={{{=#
        i_orbs = [1:1:rasi_orbs;]
        start2 = rasi_orbs+1
        end2 = start2+rasii_orbs-1
        ii_orbs = [start2:1:end2;]
        start = norbs-rasiii_orbs+1
        iii_orbs = [start:1:norbs;]
        return i_orbs, ii_orbs, iii_orbs
    end#=}}}=#
end

"""
    compute_configs(prob::RASCIAnsatz)
"""
function compute_configs(prob::RASCIAnsatz)
    if prob.na == prob.nb#={{{=#
        a_configs = ras_get_all_configs(prob.na, prob)
        return (a_configs, a_configs)
    else 
        a_configs = ras_get_all_configs(prob.na, prob)
        b_configs = ras_get_all_configs(prob.nb, prob)
        return (a_configs, b_configs)
    end#=}}}=#
end

"""
    get_all_configs(x, y, nelecs)
"""
function ras_get_all_configs(nelecs, prob::RASCIAnsatz)
    ras_olsen = ActiveSpaceSolvers.RASCI.RASCI_OlsenGraph(prob.no, nelecs, prob.fock, prob.ras1_min, prob.ras3_max)#={{{=#
    config_dict = old_dfs(nelecs, ras_olsen.connect, ras_olsen.weights, 1, ras_olsen.max)
    return config_dict#=}}}=#
end

# Returns a node dictionary where keys are configs and values are the indexes
function old_dfs(nelecs, connect, weights, start, max, visited=Vector(zeros(max)), path=[], nodes=Dict{Vector{Int32}, Int64}())
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
    apply_annhilation(config, orb_index)
"""
function apply_annhilation!(config, orb_index, graph::RASCI_OlsenGraph, must_obey::Bool)
    spot = first(findall(x->x==orb_index, config))#={{{=#
    new = Vector(config)
    i_orbs, ii_orbs, iii_orbs = make_rasorbs(graph.spaces[1], graph.spaces[2], graph.spaces[3], graph.no)
    
    splice!(new, spot)
    #need to check ras1_min parameters
    if must_obey == true && graph.ras1_min!=0
        if length(intersect(Set(i_orbs), Set(new))) < graph.ras1_min
            #println("Unallowed annhilation application, Goes outside of RAS space")
            return 1, 0
        end
    end

    sign = 1 
    if spot % 2 != 1
        sign = -1
    end
    return sign, new#=}}}=#
end

"""
    apply_creation(config, orb_index)
"""
function apply_creation!(config, orb_index, graph::RASCI_OlsenGraph, must_obey::Bool)
    insert_here = 1#={{{=#
    new = Vector(config)
    i_orbs, ii_orbs, iii_orbs = make_rasorbs(graph.spaces[1], graph.spaces[2], graph.spaces[3], graph.no)

    if isempty(config)
        new = [orb_index]
        return 1, new
    end
    
    for i in 1:length(config)
        if config[i] > orb_index
            insert_here = i
            break
        else
            insert_here += 1
        end
    end
    
    sign = 1
    if insert_here % 2 != 1
        sign = -1
    end

    
    insert!(new, insert_here, orb_index)
    if must_obey == true && graph.ras1_min!=0
        if length(intersect(Set(i_orbs), Set(new))) < graph.ras1_min
            #println("Unallowed annhilation application, Goes outside of RAS space")
            return 1, 0
        end
    end

    
    #need to check ras3_max parameters
    if orb_index in iii_orbs
        if length(intersect(Set(iii_orbs), Set(new))) > graph.ras3_max
            #println("Unallowed creation application, Goes outside of RAS space")
            return 1, 0
        end
    end

    return sign, new#=}}}=#
end

"""
    compute_1rdm(p::RASCIAnsatz, v::Vector)

# Arguments
- `p`: RASCIAnsatz just defines the current RASCI problem
- `v` : CI vector from RASCI solution
"""
function compute_1rdm(prob::RASCIAnsatz, v::Vector)
    vnew = reshape(v, prob.dima, prob.dimb)
    rdm1a = zeros(prob.no, prob.no)
    rdm1b = zeros(prob.no, prob.no)
    ga = ActiveSpaceSolvers.RASCI.RASCI_OlsenGraph(prob.no, prob.na, prob.fock, prob.ras1_min, prob.ras3_max)
    gb = ActiveSpaceSolvers.RASCI.RASCI_OlsenGraph(prob.no, prob.nb, prob.fock, prob.ras1_min, prob.ras3_max)
    a_lu_a, a_lus_a, aa_lu_a, aa_lus_a, c_lu_a, c_lus_a, cc_lu_a, cc_lus_a = ActiveSpaceSolvers.RASCI.fill_lu(prob.no, prob.na, ga)
    a_lu_b, a_lus_b, aa_lu_b, aa_lus_b, c_lu_b, c_lus_b, cc_lu_b, cc_lus_b = ActiveSpaceSolvers.RASCI.fill_lu(prob.no, prob.nb, gb)
    i_orbs, ii_orbs, iii_orbs = make_rasorbs(prob.fock[1], prob.fock[2], prob.fock[3], prob.no)

    #alpha p'q
    if a_lu_a != nothing
        for K in 1:size(a_lu_a)[1]
            for q in 1:prob.no
                for p in 1:prob.no
                    idxa = a_lu_a[K,q]
                    if idxa == 0
                        continue
                    end
                    signa = a_lus_a[K,q]
                    idxc = c_lu_a[idxa, p]
                    if idxc == 0
                        continue
                    end
                    signc = c_lus_a[idxa, p]
                    sign = signa*signc
                    @views rdm1a[q,p]+= sign*dot(vnew[idxc,:], vnew[K,:])
                end
            end
        end
    end

    #beta r's
    if a_lu_b != nothing
        for J in 1:size(a_lu_b)[1]
            for s in 1:prob.no
                idxa = a_lu_b[J, s]
                if idxa == 0
                    continue
                end
                signa = a_lus_b[J, s]
                for r in 1:prob.no
                    idxc = c_lu_b[idxa, r]
                    if idxc == 0
                        continue
                    end
                    signc = c_lus_b[idxa, r]
                    sign = signa*signc
                    @views rdm1b[s, r] += sign*dot(vnew[:, idxc], vnew[:, J])
                end
            end
        end
    end
    return rdm1a, rdm1b
    #return RDM1(rdm1a, rdm1b)
end

"""
    compute_1rdm_2rdm(p::RASCIAnsatz, v::Vector)

# Arguments
- `p`: RASCIAnsatz just defines teh current RASCI problem 
- `v` : CI vector from RASCI solution
"""
function compute_1rdm_2rdm(prob::RASCIAnsatz, v::Vector)
    vnew = reshape(v, prob.dima, prob.dimb)
    rdm1a, rdm1b = compute_1rdm(prob, v)
    rdm2aa = zeros(prob.no, prob.no, prob.no, prob.no)
    rdm2bb = zeros(prob.no, prob.no, prob.no, prob.no)
    rdm2ab = zeros(prob.no, prob.no, prob.no, prob.no)
    ga = ActiveSpaceSolvers.RASCI.RASCI_OlsenGraph(prob.no, prob.na, prob.fock, prob.ras1_min, prob.ras3_max)
    gb = ActiveSpaceSolvers.RASCI.RASCI_OlsenGraph(prob.no, prob.nb, prob.fock, prob.ras1_min, prob.ras3_max)
    
    a_lu_a, a_lus_a, aa_lu_a, aa_lus_a, c_lu_a, c_lus_a, cc_lu_a, cc_lus_a = ActiveSpaceSolvers.RASCI.fill_lu(prob.no, prob.na,ga)
    a_lu_b, a_lus_b, aa_lu_b, aa_lus_b, c_lu_b, c_lus_b, cc_lu_b, cc_lus_b = ActiveSpaceSolvers.RASCI.fill_lu(prob.no, prob.nb,gb)

    #alpha alpha p'r'sq
    if aa_lu_a != nothing
        for K in 1:size(a_lu_a)[1]
            for q in 1:prob.no
                for s in 1:prob.no
                    for r in 1:prob.no
                        for p in 1:prob.no
                            idxa = a_lu_a[K,q]
                            if idxa == 0
                                continue
                            end
                            signa = a_lus_a[K,q]
                            idxaa = aa_lu_a[idxa, s]
                            if idxaa == 0
                                continue
                            end
                            signaa = aa_lus_a[idxa,s]
                            idxcc = cc_lu_a[idxaa, r]
                            if idxcc == 0
                                continue
                            end
                            signcc = cc_lus_a[idxaa, r]
                            idxc = c_lu_a[idxcc, p]
                            if idxc == 0
                                continue
                            end
                            signc = c_lus_a[idxcc, p]
                            sign = signa*signc*signaa*signcc
                            @views rdm2aa[p,q,r,s] += sign*dot(vnew[idxc,:], vnew[K,:])
                        end
                    end
                end
            end
        end
    end

    #beta beta p'r'sq
    if aa_lu_b != nothing
        for J in 1:size(a_lu_b)[1]
            for q in 1:prob.no
                idxa = a_lu_b[J,q]
                if idxa == 0
                    continue
                end
                signa = a_lus_b[J,q]
                for s in 1:prob.no
                    idxaa = aa_lu_b[idxa, s]
                    if idxaa == 0
                        continue
                    end
                    signaa = aa_lus_b[idxa,s]
                    for r in 1:prob.no
                        idxcc = cc_lu_b[idxaa, r]
                        if idxcc == 0
                            continue
                        end
                        signcc = cc_lus_b[idxaa, r]
                        for p in 1:prob.no
                            idxc = c_lu_b[idxcc, p]
                            if idxc == 0
                                continue
                            end
                            signc = c_lus_b[idxcc, p]
                            sign = signa*signc*signaa*signcc
                            @views rdm2bb[p,q,r,s]+= sign*dot(vnew[:,idxc], vnew[:,J])
                        end
                    end
                end
            end
        end
    end
    
    #alpha beta p'r'sq
    if a_lu_b != nothing && a_lu_a != nothing
        for K in 1:size(a_lu_a)[1]
            for q in 1:prob.no
                idxa_a = a_lu_a[K, q]
                if idxa_a == 0
                    continue
                end
                signa_a = a_lus_a[K,q]
                for p in 1:prob.no
                    idxc_a = c_lu_a[idxa_a, p]
                    if idxc_a == 0
                        continue
                    end
                    signc_a = c_lus_a[idxa_a, p]

                    for J in 1:size(a_lu_b)[1]
                        for s in 1:prob.no
                            idxa_b = a_lu_b[J, s]
                            if idxa_b == 0
                                continue
                            end
                            signa_b = a_lus_b[J,s]
                            for r in 1:prob.no
                                idxc_b = c_lu_b[idxa_b, r]
                                if idxc_b == 0
                                    continue
                                end
                                signc_b = c_lus_b[idxa_b, r]
                                sign = signa_a*signc_a*signa_b*signc_b
                                @views rdm2ab[p,q,r,s]+= sign*vnew[idxc_a,idxc_b]*vnew[K,J]
                            end
                        end
                    end
                end
            end
        end
    end

    return rdm1a, rdm1b, rdm2aa, rdm2bb, rdm2ab
    #return RDM1(rdm1a, rdm1b), RDM2(rdm2aa, rdm2ab, rdm2bb)
end








    
        


