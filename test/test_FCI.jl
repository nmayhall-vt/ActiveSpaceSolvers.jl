using ActiveSpaceSolvers
using InCoreIntegrals
using LinearAlgebra
using Printf
using Test
using Arpack
using NPZ

@testset "FCI" begin
   
    h0 = npzread("h6_sto3g/h0.npy")
    h1 = npzread("h6_sto3g/h1.npy")
    h2 = npzread("h6_sto3g/h2.npy")


    ints = InCoreInts(h0, h1, h2)
    n_elec_a = 3
    n_elec_b = 3

    norb = size(h1,1)
    problem = FCIProblem(norb, n_elec_a, n_elec_b)

    display(problem)

    # test build_H_matrix
    Hmat = build_H_matrix(ints, problem)
    @time e,v = eigs(Hmat, nev = 10, which=:SR)
    e .+= ints.h0
    ref_e = -3.155304800477 # from pyscf in generate_integrals.py
    @test isapprox(e[1], ref_e , atol=1e-10)
        
    e, v = eigs(problem, ints, 3)
   
    ref = [-3.1553048004765447, -2.849049024311162, -2.5973991507252805]
    @test all(isapprox.(e.+ints.h0, ref, atol=1e-10))
    println(e.+ints.h0)


    Hmap = LinearMap(ints, problem)
    e = v' * Matrix(Hmap * v)
    display(e)
    @test all(isapprox.(diag(e.+ints.h0), ref, atol=1e-10))

    solver = ArpackSolver(nroots=3)
    println(solver)
    solution = solve(problem, ints, solver)
    display(solution)
    @test all(isapprox.(solution.energies.+ints.h0, ref, atol=1e-10))

    # string stuff
    #display(ActiveSpaceSolvers.StringCI.string_to_index("110010"))
    @test ActiveSpaceSolvers.StringCI.string_to_index("110010") == 19
end
