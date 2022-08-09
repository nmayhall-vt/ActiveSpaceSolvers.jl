using ActiveSpaceSolvers
using LinearAlgebra
using Printf
using Test
using Arpack
using NPZ

#@testset "FCI" begin
   
    h0 = npzread("h6_sto3g/h0.npy")
    h1 = npzread("h6_sto3g/h1.npy")
    h2 = npzread("h6_sto3g/h2.npy")

    n_elec_a = 3
    n_elec_b = 3

    norb = size(h1,1)
    problem = FCIProblem(norb, n_elec_a, n_elec_b)

    display(problem)

    Hmat = build_H_matrix(ints, problem)
    #@time e,v = eigs(Hmat, nev = 10, which=:SR)
    #e = real(e)
    #for ei in e
    #    @printf(" Energy: %12.8f\n",ei+ints.h0)
    #end
    #@test isapprox(e[1], e_fci , atol=1e-10)
#end

