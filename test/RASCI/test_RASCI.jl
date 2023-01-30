using QCBase
using ActiveSpaceSolvers
using InCoreIntegrals
using LinearAlgebra
using Printf
using Test
using Arpack
using NPZ
using JLD2

@load "ras_h6/_integrals.jld2"
@load "ras_h6/_ras_solution_info.jld2"

v = abs.(v)

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
    solver = SolverSettings(nroots=1, tol=1e-8, maxiter=100)
    solution = ActiveSpaceSolvers.solve(ints, prob, solver)
    eval = solution.energies
    #v = solution.vectors
    #eval = eval[1]+ints.h0
    eval = eval[1]
    #v = v[:,1]
    @test isapprox(eval, e[1], atol=10e-13)
end
