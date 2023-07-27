using QCBase
using ActiveSpaceSolvers
using InCoreIntegrals
using LinearAlgebra
using Printf
using Test
using Arpack
using NPZ
using TensorOperations
    
    
function run_ras(norbs, na, nb, ras1min, ras3max)
    h0 = npzread("integrals_h0.npy")
    h1 = npzread("integrals_h1.npy")
    h2 = npzread("integrals_h2.npy")

    ints = InCoreInts(h0, h1, h2)

    ansatz = RASCIAnsatz(norbs, na, nb, (6,6,6), ras1min, ras3max)
    display(ansatz)
    solver = SolverSettings(nroots=1, tol=1e-6, maxiter=120)
    @time solution = ActiveSpaceSolvers.solve(ints, ansatz, solver)
end



