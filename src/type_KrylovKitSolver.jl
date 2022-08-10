using KrylovKit 

"""
This type contains the information needed to solve the problem. 

    - problem::P
    - vectors::Matrix{T}

"""
struct KrylovKitSolver <: Solver 
    nroots::Int
    tol::Float64
    maxiter::Int
    verbose::Int
end



function KrylovKitSolver(;nroots=1, tol=1e-8, maxiter=100, verbose=0)
    return KrylovKitSolver(nroots, tol, maxiter, verbose)
end

function solve(problem::P, ints::InCoreInts{T}, S::KrylovKitSolver; v0=nothing) where {T, P<:Problem}

    Hmap = LinOp(ints, problem)

    e = Vector{T}([])
    v = Matrix{T}([])
    if v0 == nothing
        e, v, info = KrylovKit.eigsolve(Hmap, S.nroots, :SR, 
                                        verbosity=  verbose, 
                                        maxiter=    max_iter, 
                                        #krylovdim=20, 
                                        issymmetric=issymmetric(Hmap), 
                                        ishermitian=issymmetric(Hmap), 
                                        eager = true,
                                        tol = S.tol)
        v = hcat(v[1:R]...)
    else
        e, v, info = KrylovKit.eigsolve(Hmap, v0, S.nroots, :SR, 
                                        verbosity=  verbose, 
                                        maxiter=    max_iter, 
                                        #krylovdim=20, 
                                        issymmetric=issymmetric(Hmap), 
                                        ishermitian=issymmetric(Hmap), 
                                        eager = true,
                                        tol = S.tol)
        v = hcat(v[1:R]...)
    end
    return Solution{P,T}(problem, e, v)
end

