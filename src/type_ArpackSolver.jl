using Arpack

"""
This type contains the information needed to solve the problem. 

    - problem::P
    - vectors::Matrix{T}

"""
struct ArpackSolver <: Solver 
    nroots::Int
    tol::Float64
    maxiter::Int
end



function ArpackSolver(;nroots=1, tol=1e-8, maxiter=100)
    return ArpackSolver(nroots, tol, maxiter)
end

function solve(problem::P, ints::InCoreInts{T}, S::ArpackSolver; v0=nothing) where {T, P<:Problem}
    
    Hmap = LinearMap(ints, problem)
    
    if v0 == nothing
        e,v = Arpack.eigs(Hmap, nev = S.nroots, which=:SR, tol=S.tol)
    else
        e,v = Arpack.eigs(Hmap, v0=v0[:,1], nev = S.nroots, which=:SR, tol=S.tol)
    end
    return Solution{P,T}(problem, e, v)
end
