using ActiveSpaceSolvers

function generate_single_index_lookup(bra::RASCIAnsatz, ket::RASCIAnsatz, spin::String)
    if spin == "alpha"
        a_ket_graph = RASCI_OlsenGraph(ket.no, ket.na, ket.fock, ket.ras1_min, ket.ras3_max)
        a_bra_graph = RASCI_OlsenGraph(bra.no, bra.na, bra.fock, bra.ras1_min, bra.ras3_max)
        dim_a = calc_ndets(bra.no, bra.na, a_bra_graph.spaces, a_bra_graph.ras1_min, a_bra_graph.ras3_max)
        lu = zeros(Int, ket.dima, bra.no)
        lu_sign = zeros(Int8, ket.dima, bra.no)
        
        if bra.na - ket.na > 0
            #alpha creation
            println("Apply α creation")
            lu, lu_sign = dfs_creation(a_ket_graph, a_bra_graph, 1, a_ket_graph.max, lu, lu_sign, true)
            #b_lu, b_sign = dfs_no_operation(b_ket_graph, b_bra_graph, 1, b_ket_graph.max, b_lu, b_sign, true)
        end

        if bra.na - ket.na < 0 
            #alpha annhilation
            println("Apply α annhilation")
            lu, lu_sign = dfs_annhilation(a_ket_graph, a_bra_graph, 1, a_ket_graph.max, lu, lu_sign, true)
            #b_lu, b_sign = dfs_no_operation(b_ket_graph, b_bra_graph, 1, b_ket_graph.max, b_lu, b_sign, true)
        end

    else
        b_ket_graph = RASCI_OlsenGraph(ket.no, ket.nb, ket.fock, ket.ras1_min, ket.ras3_max)
        b_bra_graph = RASCI_OlsenGraph(bra.no, bra.nb, bra.fock, bra.ras1_min, bra.ras3_max)
        dim_b = calc_ndets(bra.no, bra.nb, b_bra_graph.spaces, b_bra_graph.ras1_min, b_bra_graph.ras3_max)
        lu = zeros(Int, ket.dimb, bra.no)
        lu_sign = zeros(Int8, ket.dimb, bra.no)
    
        if bra.nb - ket.nb > 0
            #doing beta creation
            println("Apply β creation")
            lu, lu_sign = dfs_creation(b_ket_graph, b_bra_graph, 1, b_ket_graph.max, lu, lu_sign, true)
        end
        
        if bra.nb - ket.nb < 0
            #doing beta annhilation
            println("Apply β annhilation")
            lu, lu_sign = dfs_annhilation(b_ket_graph, b_bra_graph, 1, b_ket_graph.max, lu, lu_sign, true)
        end
    end

    return lu, lu_sign
    #return a_lu, a_sign, b_lu, b_sign
end

