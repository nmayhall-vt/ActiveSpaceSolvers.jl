using ActiveSpaceSolvers
using Test
using JLD2

@load "test_tdm_cca_abb.jld2"
@load "test_tdm_cca_aba.jld2"
@load "test_tdm_cca_bbb.jld2"
@load "test_tdm_cca_aaa.jld2"

@testset "RASCI TDMs" begin
    @testset "cca_abb" begin
        tdm_cca_abb_2 = ActiveSpaceSolvers.compute_operator_cca_abb(ras_bra_sol, ras_ket_sol);
        @test isapprox(tdm_cca_abb_2, tdm_cca_abb, atol=1e-12)
    end

    @testset "cca_aba" begin
        tdm_cca_aba_2 = ActiveSpaceSolvers.compute_operator_cca_aba(ras_bra_aba_sol, ras_ket_aba_sol);
        @test isapprox(tdm_cca_aba_2, tdm_cca_aba, atol=1e-12)
    end

    @testset "cca_bbb" begin
        tdm_cca_bbb_2 = ActiveSpaceSolvers.compute_operator_cca_bbb(ras_bra_bbb_sol, ras_ket_bbb_sol);
        @test isapprox(tdm_cca_bbb_2, tdm_cca_bbb, atol=1e-12)
    end

    @testset "cca_aaa" begin
        tdm_cca_aaa_2 = ActiveSpaceSolvers.compute_operator_cca_aaa(ras_bra_aaa_sol, ras_ket_aaa_sol);
        @test isapprox(tdm_cca_aaa_2, tdm_cca_aaa, atol=1e-12)
    end

end





