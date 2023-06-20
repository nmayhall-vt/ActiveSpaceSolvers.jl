using QCBase
using ActiveSpaceSolvers
using InCoreIntegrals
using LinearAlgebra
using Printf
using Test
using Arpack
using NPZ
using JLD2

#@load "/Users/nicole/My Drive/code/ActiveSpaceSolvers.jl/test/RASCI/ras_h6/_integrals.jld2"
@load "/Users/nicole/My Drive/code/ActiveSpaceSolvers.jl/test/RASCI/ras_h6/_ras_solution.jld2"

#v = abs.(v)

#@testset "RAS Precompute same spin blocks (H6, 3α, 3β), Davidson=true" begin
#    evals, evecs = fci.RASCI.solve(ints, p, ci_vec, 100, 1, 1e-8, true, true)
#    evecs = abs.(evecs[:,1])
#    eval = evals[1]
#    #@test isapprox(evecs, v, atol=10e-10)
#    @test isapprox(eval, e[1], atol=10e-13)
#end
#
#@testset "RAS Precompute same spin blocks (H6, 3α, 3β), Davidson=false" begin
#    evals, evecs = fci.RASCI.solve(ints, p, ci_vec, 100, 1, 1e-8, true, false)
#    evecs = abs.(evecs[1])
#    eval = evals[1]
#    #@test isapprox(evecs, v, atol=10e-10)
#    @test isapprox(eval, e[1], atol=10e-13)
#end
#
#@testset "RAS Do not precompute same spin blocks (H6, 3α, 3β), Davidson=true" begin
#    evals, evecs = fci.RASCI.solve(ints, p, ci_vec, 100, 1, 1e-8, false, true)
#    evecs = abs.(evecs[:,1])
#    eval = evals[1]
#    #@test isapprox(evecs, v, atol=10e-10)
#    @test isapprox(eval, e[1], atol=10e-13)
#end

@testset "RASCI (H6, 3α, 3β), Davidson=false" begin
    solution = ActiveSpaceSolvers.solve(ints, ras, solver)
    eval = solution.energies
    @test isapprox(eval, ras_sol.energies, atol=10e-13)
end

@testset "RASCI expval of S^2" begin
    s2_new = ActiveSpaceSolvers.RASCI.compute_S2_expval(ras_sol.vectors, ras)
    @test isapprox(s2_new, s2, atol=10e-14)
end

