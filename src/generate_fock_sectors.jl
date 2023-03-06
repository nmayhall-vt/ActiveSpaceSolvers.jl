using QCBase


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
                ansatz_i = RASCIAnsatz(init_cluster_ansatz[i].no, naj, nbj, init_cluster_ansatz[i].ras_spaces, init_cluster_ansatz[i].ras1_min, init_cluster_ansatz[i].ras3_max)
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

