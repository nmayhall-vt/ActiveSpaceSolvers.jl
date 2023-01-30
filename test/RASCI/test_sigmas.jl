using QCBase
using ActiveSpaceSolvers
using InCoreIntegrals
using LinearAlgebra
using Printf
using Test
using Arpack
using NPZ
using JLD2
using StatProfilerHTML

@load "ras_h6/sigma_data.jld2"

a_configs = ActiveSpaceSolvers.RASCI.compute_configs(ras)[1]
b_configs = ActiveSpaceSolvers.RASCI.compute_configs(ras)[2]

#fill single excitation lookup tables
a_lookup = ActiveSpaceSolvers.RASCI.fill_lookup(ras, a_configs, ras.dima)
b_lookup = ActiveSpaceSolvers.RASCI.fill_lookup(ras, b_configs, ras.dimb)

sigma1 = vec(ActiveSpaceSolvers.RASCI.compute_sigma_one(b_configs, b_lookup, ci, ints, ras))
sigma2 = vec(ActiveSpaceSolvers.RASCI.compute_sigma_two(a_configs, a_lookup, ci, ints, ras))
@time sigma3 = vec(ActiveSpaceSolvers.RASCI.compute_sigma_three(a_configs, b_configs, a_lookup, b_lookup, ci, ints, ras))
@profilehtml sigma3 = vec(ActiveSpaceSolvers.RASCI.compute_sigma_three(a_configs, b_configs, a_lookup, b_lookup, ci, ints, ras))

@testset "Initial σ1, σ2, σ3 build (H6, 3α, 3β)" begin
    @test isapprox(sig1, sigma1, atol=10e-12)
    @test isapprox(sig2, sigma2, atol=10e-12)
    @test isapprox(sig3, sigma3, atol=10e-10)
end
