using QCBase


"""
        generate_cluster_fock_ansatze( ref_fock, 
                                        clusters::Vector{MOCluster}, 
                                        init_cluster_ansatz::Vector{}, 
                                        delta_elec=zeros(length(clusters)), 
                                        verbose=0) 

Generates all possible fock sectors that are reachable for the given delta_elec for each cluster
"""
function generate_cluster_fock_ansatze( ref_fock, 
                                        clusters::Vector{MOCluster}, 
                                        init_cluster_ansatz::Vector{}, 
                                        delta_elec=zeros(length(clusters)), 
                                        verbose=0) 
    ansatze = Vector{Vector{Ansatz}}()
    length(delta_elec) == length(clusters) || error("length(delta_elec) != length(clusters)") 

    for i in 1:length(clusters)
        verbose == 0 || display(clusters[i])
        delta_e_i = delta_elec[i] 
        
        #
        # Get list of Fock-space sectors for current cluster
        #
        ni = ref_fock[i][1] + ref_fock[i][2]  # number of electrons in cluster i
        sectors = []
        max_e = 2*length(clusters[i])
        min_e = 0
        for nj in ni-delta_e_i:ni+delta_e_i
        
            nj <= max_e || continue
            nj >= min_e || continue

            naj = nj÷2 + nj%2
            nbj = nj÷2
            if typeof(init_cluster_ansatz[i]) ==  FCIAnsatz
                ansatz_i = FCIAnsatz(init_cluster_ansatz[i].no, Int(naj), Int(nbj))
                push!(sectors, ansatz_i)
            elseif typeof(init_cluster_ansatz[i]) == RASCIAnsatz
                ansatz_i = RASCIAnsatz(init_cluster_ansatz[i].no, naj, nbj, init_cluster_ansatz[i].ras_spaces, max_h=init_cluster_ansatz[i].max_h, max_p=init_cluster_ansatz[i].max_p)
                display(ansatz_i)
                push!(sectors, ansatz_i)
            else
                error("No ansatz defined")
            end
        end
        append!(ansatze, [sectors])
    end
    return ansatze
end

"""
    invariant_orbital_rotations(init_cluster_ansatz::Vector{})

Generates a list of all pairs of orbitals that have invariant orbital rotations for each cluster
"""
function invariant_orbital_rotations(init_cluster_ansatz::Vector{})
    invar_pairs = []
    for i in init_cluster_ansatz
        if typeof(i) == FCIAnsatz
            #return all pairs of orbs since all are invariant
            pairs = []
            for a in 1:i.no
                for b in a+1:i.no
                    push!(pairs, (a,b))
                end
            end
            push!(invar_pairs, pairs)

        else
            #return pairs of orbs within each ras subspace
            ras1, ras2, ras3 = ActiveSpaceSolvers.RASCI.make_rasorbs(i.ras_spaces[1], i.ras_spaces[2], i.ras_spaces[3], i.no)
            println(ras1, " ", ras2, " ", ras3)
            pairs = []
            for a in 1:length(ras1)
                for b in a+1:length(ras1)
                    push!(pairs, (ras1[a],ras1[b]))
                end
            end
            
            for c in 1:length(ras2)
                for d in c+1:length(ras2)
                    push!(pairs, (ras2[c],ras2[d]))
                end
            end
            
            for e in 1:length(ras3)
                for f in e+1:length(ras3)
                    push!(pairs, (ras3[e],ras3[f]))
                end
            end
            push!(invar_pairs, pairs)
        end
    end
    return invar_pairs
end

