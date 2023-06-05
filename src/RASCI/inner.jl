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
        Hmat = .5*(Hmat+Hmat')
        Hmat += 1.0I*ints.h0
    
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
        Hmat = .5*(Hmat+Hmat')
        Hmat += 1.0I*ints.h0
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

function _ras_ss_sum!(sig::Array{T,3}, v::Array{T,3}, F::Vector{T},Ib::Int, allowed::Dict{Int, Vector{Int}}) where {T}
    nIa     = size(v)[1]
    n_roots = size(v)[2]
    nJb     = size(v)[3]


    for Jb in 1:nJb
        if abs(F[Jb]) > 1e-14 
            @inbounds @simd for si in 1:n_roots
                #for Kb in allowed
                for Ia in 1:nIa
                    if Jb in allowed[Ia]
                        sig[Ia,si,Ib] += F[Jb]*v[Ia,si,Jb]
                    end
                end
            end
        end
    end
end

function restrict_strings(prob::RASCIAnsatz, current::Dict{Vector{Int32}, Int64}; key_spin="alpha")
    allowed = Dict{Int, Vector{Int}}()
    if key_spin == "alpha"
        key_configs = ActiveSpaceSolvers.RASCI.compute_configs(prob)[1]
    else
        key_configs = ActiveSpaceSolvers.RASCI.compute_configs(prob)[2]
    end
    ras1, ras2, ras3 = make_rasorbs(prob.ras_spaces[1], prob.ras_spaces[2], prob.ras_spaces[3], prob.no)
    
    for Ia in key_configs
        ras1_test = length(findall(in(Ia[1]),ras1))
        ras3_test = length(findall(in(Ia[1]),ras3))
        tmp = []

        for Ib in current
            num_ras1 = length(findall(in(Ib[1]),ras1))
            num_ras3 = length(findall(in(Ib[1]),ras3))
            if (num_ras1+ras1_test)>=prob.ras1_min && (num_ras3+ras3_test) <= prob.ras3_max
                push!(tmp, Ib[2])
            end
        end
        allowed[Ia[2]] = tmp
    end
    return allowed
end

function restrict_opp_spin(prob::RASCIAnsatz, curr_config::Vector{Int32}; opp_spin="alpha")
    allowed = Vector{Int}()
    ras1, ras2, ras3 = make_rasorbs(prob.ras_spaces[1], prob.ras_spaces[2], prob.ras_spaces[3], prob.no)
    
    ras1_test = length(findall(in(curr_config),ras1))
    ras3_test = length(findall(in(curr_config),ras3))
    allowed_ras1 = prob.ras1_min - ras1_test
    allowed_ras3 = prob.ras3_max - ras3_test
    
    if opp_spin == "alpha"
        opp_configs = ActiveSpaceSolvers.RASCI.compute_configs(prob)[1]
    else
        opp_configs = ActiveSpaceSolvers.RASCI.compute_configs(prob)[2]
    end

    for I in opp_configs
        config = I[1]
num_ras1 = length(findall(in(config),ras1))
        num_ras3 = length(findall(in(config),ras3))
        if num_ras1 >= allowed_ras1
            if num_ras3 <= allowed_ras3
                append!(allowed, I[2])
            end
        end
    end

    return allowed
end



"""
    compute_sigma_one(b_configs, b_lookup, v, ints::InCoreInts, prob::RASCIAnsatz)
"""
function compute_sigma_one(b_configs::Dict{Vector{Int32}, Int64}, b_lookup::Array{Int64, 3}, v, ints::InCoreInts, prob::RASCIAnsatz)
    ## bb σ1(Iα, Iβ){{{
    T = eltype(v[1])
    n_roots::Int = size(v,3)
    sigma_one = zeros(prob.dima, prob.dimb, n_roots)
    #v = reshape(v, prob.dima, prob.dimb)
    
    F = zeros(T, prob.dimb)
    gkl = get_gkl(ints, prob) 
    
    sigma_one = permutedims(sigma_one,[1,3,2])
    v = permutedims(v,[1,3,2])
    
    allowed = restrict_strings(prob, b_configs, key_spin="alpha")
    
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
        #        @inbounds sigma_one[a, I_idx] += F[b]*v[a,b]
        #    end
        #end
        
        #scr = v*F
        #sigma_one[:, I_idx] .+= scr
        
        #allowed = restrict_strings(prob, b_configs, spin="alpha")
        #allowed = restrict_opp_spin(prob, I_config, opp_spin="alpha")
        #_ras_ss_sum!(sigma_one, v, F, I_idx, allowed)


        ActiveSpaceSolvers.FCI._ss_sum!(sigma_one, v, F, I_idx)
    end#=}}}=#
    
    sigma_one = permutedims(sigma_one,[1,3,2])
    v = permutedims(v,[1,3,2])
    return sigma_one
end


"""
    compute_sigma_two(a_configs, a_lookup, v, ints::InCoreInts, prob::RASCIAnsatz)
"""
function compute_sigma_two(a_configs::Dict{Vector{Int32}, Int64}, a_lookup::Array{Int64,3}, v, ints::InCoreInts, prob::RASCIAnsatz)
    ## aa σ2(Iα, Iβ){{{
    T = eltype(v[1])
    n_roots::Int = size(v,3)
    sigma_two = zeros(prob.dima, prob.dimb, n_roots)
    
    F = zeros(prob.dima)
    gkl = get_gkl(ints, prob) 
    
    sigma_two = permutedims(sigma_two,[2,3,1])
    v = permutedims(v,[2,3,1])
    
    allowed = restrict_strings(prob, a_configs, key_spin="beta")
    
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
        #        @inbounds sigma_two[I_idx,b] += F[a]*v[a,b]
        #    end
        #end

        #scr = zeros(prob.dimb)
        #@tensor begin
        #    scr[b] = F[a]*v[a,b]
        #end
        #sigma_two[I_idx,:] .+= scr

        #scr = F'*v
        #sigma_two[I_idx,:] .+= scr'
        #_ras_ss_sum!(sigma_two, v, F, I_idx, allowed)
        ActiveSpaceSolvers.FCI._ss_sum_Ia!(sigma_two, v, F, I_idx)
    
    end#=}}}=#
    
    sigma_two = permutedims(sigma_two,[3,1,2])
    v = permutedims(v,[3,1,2])
    
    return sigma_two
end

function slow_compute_sigma_three(a_configs::Dict{Vector{Int32}, Int64}, b_configs::Dict{Vector{Int32}, Int64}, a_lookup::Array{Int64, 3}, b_lookup::Array{Int64, 3}, v, ints::InCoreInts, prob::RASCIAnsatz)
    #v = reshape(v, prob.dima, prob.dimb)
    
    T = eltype(v[1])
    n_roots::Int = size(v,3)
    sigma_three = zeros(Float64, prob.dima, prob.dimb,n_roots)
    
    hkl = zeros(Float64, prob.no, prob.no)
    FJb = zeros(T, prob.dimb)
    Ckl = Array{T, 3} 

    rev_as = Dict(value => key for (key, value) in a_configs)
    rev_bs = Dict(value => key for (key, value) in b_configs)
    ras1, ras2, ras3 = make_rasorbs(prob.ras_spaces[1], prob.ras_spaces[2], prob.ras_spaces[3], prob.no)
    hkl = zeros(Float64, prob.no, prob.no)

    for Ia in a_configs
        for k in 1:prob.no,  l in 1:prob.no
            Ja = a_lookup[k,l,Ia[2]]
            Ja != 0 || continue
            sign_kl = sign(Ja)
            Ja = abs(Ja)
            a_ras1 = length(findall(in(rev_as[Ja]),ras1))
            a_ras3 = length(findall(in(rev_as[Ja]),ras3))
            hkl .= ints.h2[:,:,k,l]

            for Ib in b_configs
                for i in 1:prob.no, j in 1:prob.no
                    Jb = b_lookup[i,j,Ib[2]]
                    Jb != 0 || continue
                    sign_ij = sign(Jb)
                    Jb = abs(Jb)
                    b_ras1 = length(findall(in(rev_bs[Jb]),ras1))
                    b_ras3 = length(findall(in(rev_bs[Jb]),ras3))
                    if (a_ras1+b_ras1)>=prob.ras1_min && (a_ras3+b_ras3) <= prob.ras3_max
                        for si in 1:n_roots
                            sigma_three[Ia[2], Ib[2], si] += hkl[i,j]*v[Ja, Jb, si]*sign_ij*sign_kl
                        end
                    end
                end
            end
        end
    end
    return sigma_three
end



        #Ckl = zeros(T, length(L), prob.dimb, n_roots)
        #
        #
        ##Gather
        ##_gather!(Ckl, v, L, sign_I)
        #_gather!(Ckl, v, L, sign_I, global_allowed_b)
        #
        #hkl .= ints.h2[:,:,k,l]
        #VI = zeros(T, length(L), n_roots)

        ##look over beta configs
        #for Ib in b_configs
        #    allowed_a = Vector{Int}()
        #    @inbounds fill!(FJb, T(0.0))
        #    #@code_warntype _update_F!(hkl, F, b_lookup, Ib, prob)
        #    _update_F!(hkl, FJb, b_lookup, Ib[2], prob, allowed_a, b_configs)

        #    #VI = Ckl*FJb
        #    #@tensor begin
        #    #    VI[I,s] = Ckl[I,J,s]*FJb[J]
        #    #end
        #    _ras_mult!(Ckl, FJb, VI, allowed_a)

        #    #Scatter
        #    _scatter!(sigma_three, R, VI, Ib[2])
        #end

"""
    compute_sigma_three(a_configs, b_configs, a_lookup, b_lookup, v, ints::InCoreInts, prob::RASCIAnsatz)
"""
function compute_sigma_three(a_configs::Dict{Vector{Int32}, Int64}, b_configs::Dict{Vector{Int32}, Int64}, a_lookup::Array{Int64, 3}, b_lookup::Array{Int64, 3}, v, ints::InCoreInts, prob::RASCIAnsatz)
    #v = reshape(v, prob.dima, prob.dimb){{{
    
    T = eltype(v[1])
    n_roots::Int = size(v,3)
    sigma_three = zeros(Float64, prob.dima, prob.dimb,n_roots)
    
    hkl = zeros(Float64, prob.no, prob.no)
    FJb = zeros(T, prob.dimb)
    Ckl = Array{T, 3} 

    #short loop to show dim of Ckl for various RASCI Ansantz
    #Ckl_size = zeros(Int, prob.no, prob.no)
    #for k in 1:prob.no,  l in 1:prob.no
    #    count = 0
    #    for I in a_configs
    #        Iidx = a_lookup[k,l,I[2]]
    #        if Iidx != 0
    #            count += 1
    #        end
    #    Ckl_size[k, l] = count
    #    end
    #end
    
    for k in 1:prob.no,  l in 1:prob.no
        L = Vector{Int}()
        R = Vector{Int}()
        sign_I = Vector{Int8}()
        #
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

        Ckl = zeros(T, length(L), prob.dimb, n_roots)
        
        
        #Gather
        _gather!(Ckl, v, L, sign_I)
        
        hkl .= ints.h2[:,:,k,l]
        VI = zeros(T, length(L), n_roots)

        #look over beta configs
        for Ib in b_configs

            @inbounds fill!(FJb, T(0.0))
            #@code_warntype _update_F!(hkl, F, b_lookup, Ib, prob)
            _update_F!(hkl, FJb, b_lookup, Ib[2], prob)

            #VI = Ckl*FJb
            #@tensor begin
            #    VI[I,s] = Ckl[I,J,s]*FJb[J]
            #end
            ActiveSpaceSolvers.FCI._mult!(Ckl, FJb, VI)

            #Scatter
            _scatter!(sigma_three, R, VI, Ib[2])
        end
    end
    return sigma_three#=}}}=#
end

function _ras_mult!(Ckl::Array{T,3}, FJb::Array{T,1}, VI::Array{T,2}, allowed::Vector{Int}) where {T}
    #={{{=#
    VI .= 0
    nI = size(Ckl)[1]
    n_roots::Int = size(Ckl)[3]
    ket_max = size(FJb)[1]
    tmp = 0.0
    for si in 1:n_roots
        @views V = VI[:,si]
        for Jb in 1:ket_max
            tmp = FJb[Jb]
            if abs(tmp) > 1e-14
                @inbounds @simd for I in allowed
                #@inbounds @simd for I in 1:nI
                    VI[I,si] += tmp*Ckl[I,Jb,si]
                end
                #@views LinearAlgebra.axpy!(tmp, Ckl[:,Jb,si], VI[:,si])
                #@inbounds VI[:,si] .+= tmp .* Ckl[:,Jb,si]
                #@inbounds @views @. VI[:,si] += tmp * Ckl[:,Jb,si]
            end
        end
    end
end
#=}}}=#

function _gather!(Ckl::Array{T,3}, v, L::Vector{Int}, sign_I::Vector{Int8}, allowed::Vector{Vector{Int}}) where {T}
    nI = length(L)#={{{=#
    n_roots = size(v)[3]
    ket_max = size(v)[2]
    @inbounds @simd for si in 1:n_roots
        for Li in 1:nI
            for Jb in allowed[Li]
                Ckl[Li,Jb,si] = v[L[Li], Jb,si] * sign_I[Li]
            end
        end
    end#=}}}=#
end

function _gather!(Ckl::Array{T,3}, v, L::Vector{Int}, sign_I::Vector{Int8}) where {T}
    nI = length(L)#={{{=#
    n_roots = size(v)[3]
    ket_max = size(v)[2]
    @inbounds @simd for si in 1:n_roots
        for Jb in 1:ket_max
            for Li in 1:nI
                Ckl[Li,Jb,si] = v[L[Li], Jb,si] * sign_I[Li]
            end
        end
    end#=}}}=#
end

function _scatter!(sigma_three::Array{T, 3}, R::Vector{Int}, VI::Array{T, 2}, Ib::Int) where {T}
    #sigma_three[R, Ib[2]] .+= VI{{{
    n_roots = size(sigma_three)[3]
    
    #what nick uses
    for si in 1:n_roots
    #@inbounds @simd for si in 1:n_roots
        for Li in 1:size(VI,1)
            sigma_three[R[Li], Ib, si] += VI[Li,si]
        end
    end
    
    #Fastest for me
    #@views a = sigma_three[:, Ib]
    #for (Li,L) in enumerate(VI)
    #    @inbounds a[R[Li]] += L
    #end}}}
end
#end

function _update_F!(hkl::Array{T,2}, FJb::Vector{T}, b_lookup::Array{Int,3}, Ib::Int, prob::RASCIAnsatz, allowed::Vector{Int}, b_configs::Dict{Vector{Int32}, Int64}) where {T}
    i::Int = 1#={{{=#
    j::Int = 1
    Jb::Int = 1
    sign_ij::T = 1.0
    for j in 1:prob.no
        #jkl_idx = j-1 + (k-1)*prob.no + (l-1)*prob.no*prob.no 
        for i in 1:prob.no
            #ijkl_idx = (i-1) + jkl_idx*prob.no + 1
            Jb = b_lookup[i,j,Ib]
            if Jb == 0
                continue
            end
            
            sign_ij = sign(Jb)
            Jb = abs(Jb)
            
            for (k,v) in b_configs if v==Jb
                    allowed = restrict_opp_spin(prob, k, opp_spin="alpha")
                end
            end
            

            @inbounds FJb[Jb] += sign_ij*hkl[i,j]
        end
    end#=}}}=#
end

function _update_F!(hkl::Array{T,2}, FJb::Vector{T}, b_lookup::Array{Int,3}, Ib::Int, prob::RASCIAnsatz) where {T}
    i::Int = 1#={{{=#
    j::Int = 1
    Jb::Int = 1
    sign_ij::T = 1.0
    for j in 1:prob.no
        #jkl_idx = j-1 + (k-1)*prob.no + (l-1)*prob.no*prob.no 
        for i in 1:prob.no
            #ijkl_idx = (i-1) + jkl_idx*prob.no + 1
            Jb = b_lookup[i,j,Ib]
            if Jb == 0
                continue
            end
            sign_ij = sign(Jb)
            Jb = abs(Jb)
            @inbounds FJb[Jb] += sign_ij*hkl[i,j]
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
    ras1, ras2, ras3 = make_rasorbs(prob.ras_spaces[1], prob.ras_spaces[2], prob.ras_spaces[3], prob.no)
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
    ras1, ras2, ras3 = make_rasorbs(prob.ras_spaces[1], prob.ras_spaces[2], prob.ras_spaces[3], prob.no)
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
    ras_olsen = ActiveSpaceSolvers.RASCI.RASCI_OlsenGraph(prob.no, nelecs, prob.ras_spaces, prob.ras1_min, prob.ras3_max)#={{{=#
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
"""
function apply_single_excitation!(config, a_orb, c_orb, config_dict, categories::Vector{HP_Category_CA})
    #println("Config: ", config)
    #println(a_orb, " : ", c_orb)
    idx_org = config_dict[config]
    cat_org = find_cat(idx_org, categories)
    idx_local = findfirst(item -> item == idx_org, cat_org.idxs)

    spot = first(findall(x->x==a_orb, config))#={{{=#
    new = Vector(config)
    splice!(new, spot)
    
    sign_a = 1 
    if spot % 2 != 1
        sign_a = -1
    end
    
    if c_orb in new
        return 1, 0, 0,0
    end

    insert_here = 1
    new2 = Vector(new)

    if isempty(new)
        new2 = [c_orb]
        sign_c = 1
        #println("New config: ", new2)
        
        if haskey(config_dict, new2);
            idx = config_dict[new2]
        else
            return 1, 0, 0,0
        end
        cat = find_cat(idx, categories)
        if cat == 0
            return 1, 0,0,0
        end

    else
        for i in 1:length(new)
            if new[i] > c_orb
                insert_here = i
                break
            else
                insert_here += 1
            end
        end

        insert!(new2, insert_here, c_orb)
        #println("New config: ", new2)

        if haskey(config_dict, new2);
            idx = config_dict[new2]
        else
            return 1, 0, 0,0
        end
        cat = find_cat(idx, categories)
        if cat == 0
            return 1, 0,0,0
        end


        sign_c = 1
        if insert_here % 2 != 1
            sign_c = -1
        end
    end

    return sign_c*sign_a, new2, idx_local, idx#=}}}=#
    #return sign_c*sign_a, new2, idx#=}}}=#
end



"""
    apply_annhilation(config, orb_index)
"""
function apply_annhilation!(config, orb_index, config_dict, categories::Vector{HP_Category_CA})
    spot = first(findall(x->x==orb_index, config))#={{{=#
    new = Vector(config)
    
    splice!(new, spot)
    if haskey(config_dict, new)
        idx = config_dict[new]
    else
        return 1, 0, 0
    end


    cat = find_cat(idx, categories)
    if cat == 0
        #println("not allowed")
        return 1, 0, 0
    end

    sign = 1 
    if spot % 2 != 1
        sign = -1
    end
    return sign, new, idx#=}}}=#
end

"""
    apply_creation(config, orb_index)
"""
function apply_creation!(config, orb_index, config_dict)
    insert_here = 1#={{{=#
    new = Vector(config)

    if isempty(config)
        new = [orb_index]
    else
        for i in 1:length(config)
            if config[i] > orb_index
                insert_here = i
                break
            else
                insert_here += 1
            end
        end

        insert!(new, insert_here, orb_index)
    end
    
    sign = 1
    if insert_here % 2 != 1
        sign = -1
    end
    idx = 1

    return sign, new, idx#=}}}=#
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

function apply_a(config, orb, config_dict_ket, config_dict_bra, cats_ket::Vector{<:HP_Category}, cats_bra::Vector{<:HP_Category})
    idx_org = config_dict_ket[config]
    cat_org = find_cat(idx_org, cats_ket)
    idx_local = findfirst(item -> item == idx_org, cat_org.idxs)

    spot = first(findall(x->x==orb, config))
    new = Vector(config)
    
    splice!(new, spot)
    if haskey(config_dict_bra, new)
        idx = config_dict_bra[new]
    else
        return 1, 0, 0, 0
    end

    cat = find_cat(idx, cats_bra)
    if cat == 0
        return 1, 0, 0, 0
    end

    sign = 1 
    if spot % 2 != 1
        sign = -1
    end
    return sign, new, idx_local, idx
end

function apply_c(config, orb, config_dict_ket, config_dict_bra, cats_ket::Vector{<:HP_Category}, cats_bra::Vector{<:HP_Category})
    idx_org = config_dict_ket[config]
    cat_org = find_cat(idx_org, cats_ket)
    idx_local = findfirst(item -> item == idx_org, cat_org.idxs)

    new = Vector(config)
    insert_here = 1
    
    if isempty(config)
        new = [orb]
        sign_c = 1
        
        if haskey(config_dict_bra, new);
            idx = config_dict[new]
        else
            return 1, 0, 0,0
        end
        cat = find_cat(idx, cats_bra)
        if cat == 0
            return 1, 0,0,0
        end

    else
        for i in 1:length(config)
            if config[i] > orb
                insert_here = i
                break
            else
                insert_here += 1
            end
        end

        insert!(new, insert_here, orb)
        if haskey(config_dict_bra, new);
            idx = config_dict_bra[new]
        else
            return 1, 0, 0,0
        end
        cat = find_cat(idx, cats_bra)
        if cat == 0
            return 1, 0,0,0
        end

        sign_c = 1
        if insert_here % 2 != 1
            sign_c = -1
        end
    end
    return sign_c, new, idx_local, idx#=}}}=#
end

function apply_cc(config, orb, orb2, config_dict_ket, config_dict_bra, cats_ket::Vector{<:HP_Category}, cats_bra::Vector{<:HP_Category})
    idx_org = config_dict_ket[config]
    cat_org = find_cat(idx_org, cats_ket)
    idx_local = findfirst(item -> item == idx_org, cat_org.idxs)

    new = Vector(config)
    insert_here = 1
    sign_c = 1
    
    if isempty(config)
        new = [orb]
        sign_c = 1

    else
        for i in 1:length(config)
            if config[i] > orb
                insert_here = i
                break
            else
                insert_here += 1
            end
        end

        insert!(new, insert_here, orb)

        if insert_here % 2 != 1
            sign_c = -1
        end
    end

    insert_here = 1
    new2 = Vector(new)
    for i in 1:length(new)
        if new[i] > orb2
            insert_here = i
            break
        else
            insert_here += 1
        end
    end

    insert!(new2, insert_here, orb2)

    sign_cc = 1
    if insert_here % 2 != 1
        sign_cc = -1
    end
    
    if haskey(config_dict_bra, new2);
        idx = config_dict_bra[new2]
    else
        return 1, 0, 0,0
    end
    cat = find_cat(idx, cats_bra)
    if cat == 0
        return 1, 0,0,0
    end

    return sign_c*sign_cc, new2, idx_local, idx#=}}}=#
end

function apply_cca(config, orb_a, orb, orb2, config_dict_ket, config_dict_bra, cats_ket::Vector{<:HP_Category}, cats_bra::Vector{<:HP_Category})
    idx_org = config_dict_ket[config]
    cat_org = find_cat(idx_org, cats_ket)
    idx_local = findfirst(item -> item == idx_org, cat_org.idxs)

    #apply annhilation
    spot = first(findall(x->x==orb_a, config))
    new_a = Vector(config)
    splice!(new_a, spot)
    sign_a = 1
    if spot % 2 != 1
        sign_a = -1
    end
    
    #apply first creation
    new = Vector(new_a)
    insert_here = 1
    sign_c = 1
    
    if isempty(new_a)
        new = [orb]
        sign_c = 1

    else
        for i in 1:length(new_a)
            if new_a[i] > orb
                insert_here = i
                break
            else
                insert_here += 1
            end
        end

        insert!(new, insert_here, orb)

        if insert_here % 2 != 1
            sign_c = -1
        end
    end
    
    #apply second creation
    insert_here = 1
    new2 = Vector(new)
    for i in 1:length(new)
        if new[i] > orb2
            insert_here = i
            break
        else
            insert_here += 1
        end
    end

    insert!(new2, insert_here, orb2)

    sign_cc = 1
    if insert_here % 2 != 1
        sign_cc = -1
    end
    
    if haskey(config_dict_bra, new2);
        idx = config_dict_bra[new2]
    else
        return 1, 0, 0,0
    end
    cat = find_cat(idx, cats_bra)
    if cat == 0
        return 1, 0,0,0
    end

    return sign_a*sign_c*sign_cc, new2, idx_local, idx#=}}}=#
end

function apply_ccaa(config, orb_a, orb_aa, orb, orb2, config_dict_ket, config_dict_bra, cats_ket::Vector{<:HP_Category}, cats_bra::Vector{<:HP_Category})
    idx_org = config_dict_ket[config]
    cat_org = find_cat(idx_org, cats_ket)
    idx_local = findfirst(item -> item == idx_org, cat_org.idxs)

    #apply first annhilation
    spot = first(findall(x->x==orb_a, config))
    new_a = Vector(config)
    splice!(new_a, spot)
    sign_a = 1
    if spot % 2 != 1
        sign_a = -1
    end
    
    #apply second annhilation
    spota = first(findall(x->x==orb_aa, new_a))
    new_aa = Vector(new_a)
    splice!(new_aa, spota)
    sign_aa = 1
    if spota % 2 != 1
        sign_aa = -1
    end
    
    #apply first creation
    new = Vector(new_aa)
    insert_here = 1
    sign_c = 1
    
    if isempty(new_aa)
        new = [orb]
        sign_c = 1

    else
        for i in 1:length(new_aa)
            if new_aa[i] > orb
                insert_here = i
                break
            else
                insert_here += 1
            end
        end

        insert!(new, insert_here, orb)

        if insert_here % 2 != 1
            sign_c = -1
        end
    end
    
    #apply second creation
    insert_here = 1
    new2 = Vector(new)
    for i in 1:length(new)
        if new[i] > orb2
            insert_here = i
            break
        else
            insert_here += 1
        end
    end

    insert!(new2, insert_here, orb2)

    sign_cc = 1
    if insert_here % 2 != 1
        sign_cc = -1
    end
    
    if haskey(config_dict_bra, new2);
        idx = config_dict_bra[new2]
    else
        return 1, 0, 0,0
    end
    cat = find_cat(idx, cats_bra)
    if cat == 0
        return 1, 0,0,0
    end

    return sign_a*sign_aa*sign_c*sign_cc, new2, idx_local, idx#=}}}=#
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
    ga = ActiveSpaceSolvers.RASCI.RASCI_OlsenGraph(prob.no, prob.na, prob.ras_spaces, prob.ras1_min, prob.ras3_max)
    gb = ActiveSpaceSolvers.RASCI.RASCI_OlsenGraph(prob.no, prob.nb, prob.ras_spaces, prob.ras1_min, prob.ras3_max)
    a_lu_a, a_lus_a, aa_lu_a, aa_lus_a, c_lu_a, c_lus_a, cc_lu_a, cc_lus_a = ActiveSpaceSolvers.RASCI.fill_lu(prob.no, prob.na, ga)
    a_lu_b, a_lus_b, aa_lu_b, aa_lus_b, c_lu_b, c_lus_b, cc_lu_b, cc_lus_b = ActiveSpaceSolvers.RASCI.fill_lu(prob.no, prob.nb, gb)
    i_orbs, ii_orbs, iii_orbs = make_rasorbs(prob.ras_spaces[1], prob.ras_spaces[2], prob.ras_spaces[3], prob.no)

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
    ga = ActiveSpaceSolvers.RASCI.RASCI_OlsenGraph(prob.no, prob.na, prob.ras_spaces, prob.ras1_min, prob.ras3_max)
    gb = ActiveSpaceSolvers.RASCI.RASCI_OlsenGraph(prob.no, prob.nb, prob.ras_spaces, prob.ras1_min, prob.ras3_max)
    
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








    
        



