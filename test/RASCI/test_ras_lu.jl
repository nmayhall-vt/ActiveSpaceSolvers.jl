using ActiveSpaceSolvers
using Test
using JLD2

@load "/Users/nicole/My Drive/code/ActiveSpaceSolvers.jl/test/RASCI/ras_h6/_ras_solution_info.jld2"
#prob = p

#function test_ras_lu(prob::ActiveSpaceSolvers.RASCI.RASCIAnsatz)
@testset "RAS Lookup Tables" begin
    a_configs = ActiveSpaceSolvers.RASCI.compute_configs(prob)[1]
    b_configs = ActiveSpaceSolvers.RASCI.compute_configs(prob)[2]
    
    #Create/Annihlate lookup table
    a_lookup = ActiveSpaceSolvers.RASCI.fill_lookup(prob, a_configs, prob.dima)
    b_lookup = ActiveSpaceSolvers.RASCI.fill_lookup(prob, b_configs, prob.dimb)

    ga = ActiveSpaceSolvers.RASCI.RASCI_OlsenGraph(prob.no, prob.na, prob.ras_spaces, prob.ras1_min, prob.ras3_max)
    gb = ActiveSpaceSolvers.RASCI.RASCI_OlsenGraph(prob.no, prob.nb, prob.ras_spaces, prob.ras1_min, prob.ras3_max)
    a, as, aa, aas, c, cs, cc, ccs = ActiveSpaceSolvers.RASCI.fill_lu(prob.no, prob.na, ga)
    a_b, as_b, aa_b, aas_b, c_b, cs_b, cc_b, ccs_b = ActiveSpaceSolvers.RASCI.fill_lu(prob.no,prob.nb,gb)
    dima = ActiveSpaceSolvers.RASCI.calc_ndets(prob.no, prob.na, prob.ras_spaces, prob.ras1_min, prob.ras3_max)
    dimb = ActiveSpaceSolvers.RASCI.calc_ndets(prob.no, prob.nb, prob.ras_spaces, prob.ras1_min, prob.ras3_max)

    # Alpha
    # p'r'sq = p'qr's if q != r 
    # check if at least 1 alpha electron
    @testset "Alpha Single Excitation Lookup" begin
        if a != nothing
            #loop through configs
            #for K in 1:size(a)[1]
            for K in 1:dima
                #do single excitation p'q
                for q in 1:prob.no
                    idxa = a[K, q]
                    if idxa == 0
                        continue
                    end
                    for p in 1:prob.no
                        idx = a_lookup[q, p, K]
                        idxc = c[idxa, p]
                        if abs(idx) != idxc
                            println("q, p, K;\n", q, " ", p, " ", K)
                        end
                        @test abs(idx) == idxc
                    end
                end
            end
        end
    end

    @testset "Alpha Double Excitation Lookup" begin
        #check if at least 2 alpha electrons
        if aa != nothing
            #for K in 1:size(a)[1]
            for K in 1:dima
                for q in 1:prob.no
                    for s in 1:prob.no
                        for r in 1:prob.no
                            idx1 = a_lookup[s, r, K]
                            if idx1 == 0
                                continue
                            end
                            for p in 1:prob.no
                                idx2 = a_lookup[q, p, abs(idx1)]
                                idxa_double = a[K, q]
                                if idxa_double == 0
                                    continue
                                end
                                idxaa_double = aa[idxa_double, s]
                                if idxaa_double == 0
                                    continue
                                end
                                idxcc_double = cc[idxaa_double, r]
                                if idxcc_double == 0
                                    continue
                                end

                                idxc_double = c[idxcc_double, p]
                                @test idxc_double == abs(idx2)
                            end
                        end
                    end
                end
            end
        end
    end

    # Beta
    # p'r'sq = p'qr's if q != r 
    # check if at least 1 alpha electron
    @testset "Beta Single Excitation lu tables" begin
        if a_b != nothing
            #loop through configs
            #for K in 1:size(a_b)[1]
            for K in 1:dimb
                #do single excitation p'q
                for s in 1:prob.no
                    for r in 1:prob.no
                        idx = b_lookup[s, r, K]
                        idxa = a_b[K, s]
                        if idxa == 0
                            continue
                        end

                        idxc = c_b[idxa, r]
                        @test abs(idx) == idxc
                    end
                end
            end
        end
    end
    
    @testset "Beta Double Excitation lu tables" begin
        #check if at least 2 alpha electrons
        if aa_b != nothing
            #for K in 1:size(a_b)[1]
            for K in 1:dimb
                for q in 1:prob.no
                    for s in 1:prob.no
                        for r in 1:prob.no
                            idx1 = b_lookup[s, r, K]
                            if idx1 == 0
                                continue
                            end
                            for p in 1:prob.no
                                idx2 = b_lookup[q, p, abs(idx1)]
                                idxa_double = a_b[K, q]
                                if idxa_double == 0
                                    continue
                                end
                                idxaa_double = aa_b[idxa_double, s]
                                if idxaa_double == 0
                                    continue
                                end
                                idxcc_double = cc_b[idxaa_double, r]
                                if idxcc_double == 0
                                    continue
                                end
                                idxc_double = c_b[idxcc_double, p]
                                @test idxc_double == abs(idx2)
                            end
                        end
                    end
                end
            end
        end
    end
end

function test_ras_lu(prob::ActiveSpaceSolvers.RASCI.RASCIAnsatz, ga::ActiveSpaceSolvers.RASCI.RASCI_OlsenGraph)
    a_configs = ActiveSpaceSolvers.RASCI.compute_configs(prob)[1]
    b_configs = ActiveSpaceSolvers.RASCI.compute_configs(prob)[2]
    
    #Create/Annihlate lookup table
    a_lookup = ActiveSpaceSolvers.RASCI.fill_lookup(prob, a_configs, dima)
    b_lookup = ActiveSpaceSolvers.RASCI.fill_lookup(prob, b_configs, dimb)

    #ga = ActiveSpaceSolvers.RASCI.RASCI_OlsenGraph(prob.no, prob.na, prob.ras_spaces, prob.ras1_min, prob.ras3_max)
    gb = ActiveSpaceSolvers.RASCI.RASCI_OlsenGraph(prob.no, prob.nb, prob.ras_spaces, prob.ras1_min, prob.ras3_max)
    a, as, aa, aas, c, cs, cc, ccs = ActiveSpaceSolvers.RASCI.fill_lu(prob.no, prob.na, ga)
    a_b, as_b, aa_b, aas_b, c_b, cs_b, cc_b, ccs_b = ActiveSpaceSolvers.RASCI.fill_lu(prob.no,prob.nb,gb)

    # Alpha
    # p'r'sq = p'qr's if q != r 
    # check if at least 1 alpha electron
    @testset "Alpha Single Excitation Lookup" begin
        if a != nothing
            #loop through configs
            for K in 1:size(a)[1]
                #do single excitation p'q
                for q in 1:prob.no
                    idxa = a[K, q]
                    if idxa == 0
                        continue
                    end
                    for p in 1:prob.no
                        idx = a_lookup[q, p, K]
                        idxc = c[idxa, p]
                        if abs(idx) != idxc
                            println("q, p, K;\n", q, " ", p, " ", K)
                        end
                        @test abs(idx) == idxc
                    end
                end
            end
        end
    end

    @testset "Alpha Double Excitation Lookup" begin
        #check if at least 2 alpha electrons
        if aa != nothing
            for K in 1:size(a)[1]
                for q in 1:prob.no
                    for s in 1:prob.no
                        for r in 1:prob.no
                            idx1 = a_lookup[s, r, K]
                            if idx1 == 0
                                continue
                            end
                            for p in 1:prob.no
                                idx2 = a_lookup[q, p, abs(idx1)]
                                idxa_double = a[K, q]
                                if idxa_double == 0
                                    continue
                                end
                                idxaa_double = aa[idxa_double, s]
                                if idxaa_double == 0
                                    continue
                                end
                                idxcc_double = cc[idxaa_double, r]
                                if idxcc_double == 0
                                    continue
                                end

                                idxc_double = c[idxcc_double, p]
                                @test idxc_double == abs(idx2)
                            end
                        end
                    end
                end
            end
        end
    end

    # Beta
    # p'r'sq = p'qr's if q != r 
    # check if at least 1 alpha electron
    @testset "Beta Single Excitation lu tables" begin
        if a_b != nothing
            #loop through configs
            for K in 1:size(a_b)[1]
                #do single excitation p'q
                for s in 1:prob.no
                    for r in 1:prob.no
                        idx = b_lookup[s, r, K]
                        idxa = a_b[K, s]
                        if idxa == 0
                            continue
                        end

                        idxc = c_b[idxa, r]
                        @test abs(idx) == idxc
                    end
                end
            end
        end
    end
    
    @testset "Beta Double Excitation lu tables" begin
        #check if at least 2 alpha electrons
        if aa_b != nothing
            for K in 1:size(a_b)[1]
                for q in 1:prob.no
                    for s in 1:prob.no
                        for r in 1:prob.no
                            idx1 = b_lookup[s, r, K]
                            if idx1 == 0
                                continue
                            end
                            for p in 1:prob.no
                                idx2 = b_lookup[q, p, abs(idx1)]
                                idxa_double = a_b[K, q]
                                if idxa_double == 0
                                    continue
                                end
                                idxaa_double = aa_b[idxa_double, s]
                                if idxaa_double == 0
                                    continue
                                end
                                idxcc_double = cc_b[idxaa_double, r]
                                if idxcc_double == 0
                                    continue
                                end
                                idxc_double = c_b[idxcc_double, p]
                                @test idxc_double == abs(idx2)
                            end
                        end
                    end
                end
            end
        end
    end
end




