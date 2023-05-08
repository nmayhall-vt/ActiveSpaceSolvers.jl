using JLD2
using InCoreIntegrals
using StaticArrays

struct Spin_Pair
    pair::Tuple{HP_Category, HP_Category} #(alpha HP category, beta HP category)
    dim::Int #alpha_HP.dima*beta_HP.dimb
    shift::Int #shift of dim of Spin Pairs that come before 
end

function Spin_Pair(pair::Tuple{HP_Category, HP_Category})
    dim = length(pair[1].idxs)*length(pair[2].idxs)
    shift = 0
    return Spin_Pair(pair, dim, shift)
end

function make_spin_pairs(prob::RASCIAnstaz)
    spin_pairs = Vector{Spin_Pair}()
    a_categories = ActiveSpaceSolvers.RASCI.make_categories(prob, spin="alpha")
    b_categories = ActiveSpaceSolvers.RASCI.make_categories(prob, spin="beta")
    return spin_pairs
end

function sigma_one_pair(prob::RASCIAnsatz, spin_pairs::Vector{Spin_Pair}, ints::InCoreInts, C)
    gkl = get_gkl(ints, prob) 
    for μ in spin_pairs
        for Ib in 1:size(μ[2].lookup,3)
            comb_kl = 0
            comb_ij = 0
            for k in 1:prob.no, l in 1:prob.no
                Kb = cats_b[μ[2]].lookup[l,k,Ib]
                Kb != 0 || continue
                sign_kl = sign(Kb)
                #ν = find which spin category Kb lives
                #check if (μ[1].idx, ν) is a spin pair || continue
                # need to find spin pair that is (μ[1].idx, ν)
                Kb_local = Kb-cats_b[ν].shift
                comb_kl = (k-1)*prob.no + l
                for Ia in 1:length(cats_b[μ[1]].idxs)
                    for r in 1:nroots
                        sigma_one[μ][Ia, Ib, r] += sign_kl*gkl[k,l]*C[ν][Ia, Kb_local, r]
                    end
                end

                for i in 1:prob.no, j in 1:prob.no
                    comb_ij = (i-1)*prob.no + j
                    if comb_ij < comb_kl
                        continue
                    end
                    Jb = ν[2].lookup[j,i,Kb_local]
                    Jb != 0 || continue
                    sign_ij = sign(Jb)
                    #λ = find which spin cat Jb lives
                    Jb_local = Jb-λ.shift
                    for Ia in 1:length(μ[1].idxs)
                        if comb_kl == comb_ij
                            delta = 1
                        else
                            delta = 0
                        end

                        for r in 1:nroots
                            sigma_one[μ][Ia, Ib, r] += ints.h2[i,j,k,l]*sign_kl*sign_ij*1(1+delta)*C[λ][Ia, Jb_local, r]
                        end
                    end
                end
            end
        end
    end
    return sigma_one
end

function sigma_two_pair()
end






    

