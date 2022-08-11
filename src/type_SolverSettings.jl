using ActiveSpaceSolvers
using Arpack
#using KrylovKit 

"""
This type contains the solver settings information needed to solve the problem. 
    
    - nroots::Int
    - tol::Float64
    - maxiter::Int
    - verbose::Int
    - package::String

"""
struct  SolverSettings 
    nroots::Int
    tol::Float64
    maxiter::Int
    verbose::Int
    package::String
end


"""
    SolverSettings(;nroots=1, tol=1e-8, maxiter=100, verbose=0, package="arpack")

Default value constructor
"""
function SolverSettings(;nroots=1, tol=1e-8, maxiter=100, verbose=0, package="arpack")
    return SolverSettings(nroots, tol, maxiter, verbose, package)
end


"""
    solve(ints::InCoreInts{T}, ansatz::A, S::SolverSettings; v0=nothing) where {T, A<:Ansatz}

Get the energies and eigenstates (stored as a `Solution{A,T}` type), for the Hamiltonian (defined 
by `ints`) with the wavefunction approximated by the ansatz (defined by `ansatz`), and passing the 
solver settings (defined by `S`) to the solver.
"""
function solve(ints::InCoreInts{T}, ansatz::A, S::SolverSettings; v0=nothing) where {T, A<:Ansatz}

    #e = Vector{T}([])
    #v = Matrix{T}([])

    if lowercase(S.package) == "arpack"

        Hmap = LinearMap(ints, ansatz)

        if v0 == nothing
            e,v = Arpack.eigs(Hmap, nev = S.nroots, which=:SR, tol=S.tol)
        else
            e,v = Arpack.eigs(Hmap, v0=v0[:,1], nev = S.nroots, which=:SR, tol=S.tol)
        end
        return Solution{A,T}(ansatz, e, v)


    elseif lowercase(S.package) == "krylovkit"
        
        error("NYI")

#        Hmap = LinOp(ints, ansatz)
#
#        if v0 == nothing
#            e, v, info = KrylovKit.eigsolve(Hmap, S.nroots, :SR, 
#                                            verbosity   = S.verbose, 
#                                            maxiter     = S.maxiter, 
#                                            #krylovdim  = 20, 
#                                            issymmetric = issymmetric(Hmap), 
#                                            ishermitian = ishermitian(Hmap), 
#                                            eager       = true,
#                                            tol         = S.tol)
#            v = hcat(v[1:R]...)
#        else
#            error("huh")     
#            e, v, info = KrylovKit.eigsolve(Hmap, v0, S.nroots, :SR, 
#                                            verbosity   = S.verbose, 
#                                            maxiter     = S.maxiter, 
#                                            #krylovdim  = 20, 
#                                            issymmetric = issymmetric(Hmap), 
#                                            ishermitian = ishermitian(Hmap), 
#                                            eager       = true,
#                                            tol         = S.tol)
#            v = hcat(v[1:R]...)
#        end
#        if S.verbose > 0
#            @printf(" Number of matvecs performed: %5i\n", info.numops)
#            @printf(" Number of subspace restarts: %5i\n", info.numiter)
#        end
#        return Solution{P,T}(ansatz, e, v)
    end
end

