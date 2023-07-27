using QCBase
using ActiveSpaceSolvers
using InCoreIntegrals
using LinearAlgebra
using Printf
using Test
using Arpack
using NPZ
using JLD2

@load "ras_h6/_ras_solution.jld2"

#v = abs.(v)

@testset "RASCI (H6, 3α, 3β), Davidson=false" begin
    solution = ActiveSpaceSolvers.solve(ints, ras, solver)
    eval = solution.energies
    @test isapprox(eval, ras_sol.energies, atol=10e-13)
end

@testset "RASCI expval of S^2" begin
    s2_new = ActiveSpaceSolvers.RASCI.compute_S2_expval(ras_sol.vectors, ras)
    @test isapprox(s2_new, s2, atol=10e-14)
end

