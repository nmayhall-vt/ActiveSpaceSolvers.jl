using QCBase
using ActiveSpaceSolvers
using InCoreIntegrals
using LinearAlgebra
using Printf
using Test
using Arpack
using NPZ

#function run()
@testset "h4" begin
 
    h0 = npzread("h4_ccpvdz/h0.npy")
    h1 = npzread("h4_ccpvdz/h1.npy")
    h2 = npzread("h4_ccpvdz/h2.npy")


    ints = InCoreInts(h0, h1, h2)

    ints = subset(ints, 1:8)

    #k = rand(8,8);
    #U = exp(k-k')
    #ints = orbital_rotation(ints,U)
    n_elec_a = 2
    n_elec_b = 2

    norb = n_orb(ints)
    ansatz = FCIAnsatz(norb, n_elec_a, n_elec_b)

    display(ansatz)

    # test build_H_matrix
    #@test isapprox(e[1], ref_e , atol=1e-10)
  
    ref = [-2.22833756, -1.9548734, -1.84003465]
    
    Hmap = LinearMap(ints, ansatz)
    e, v = eigs(Hmap, nev=3, which=:SR)
    println(e)
    #println(ref)
    #@test all(isapprox.(e, ref, atol=1e-10))


    Hmap = LinearMap(ints, ansatz)
    e = v' * Matrix(Hmap * v)
    display(e)
    #@test all(isapprox.(diag(e), ref, atol=1e-10))

    solver = SolverSettings(nroots=10, package="arpack")
    println(solver)
    solution = solve(ints, ansatz, solver)
    display(solution)

    S2 = build_S2_matrix(solution.ansatz)
    s2a = diag(solution.vectors' * S2 * solution.vectors)
    s2 = compute_s2(solution)

    for r in 1:solver.nroots
        display(round(s2[r]))
        @test isapprox(s2a[r], s2[r], atol=1e-10)
    end
    @test round(s2[1]) == 0
    @test round(s2[2]) == 2
    @test round(s2[3]) == 0
    @test round(s2[4]) == 2
    


end
#run()
