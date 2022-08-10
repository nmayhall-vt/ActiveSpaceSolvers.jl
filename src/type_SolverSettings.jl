using ActiveSpaceSolvers
using Arpack
#using KrylovKit 

"""
This type contains the solver settings information needed to solve the problem. 


"""
struct  SolverSettings 
    nroots::Int
    tol::Float64
    maxiter::Int
    verbose::Int
    package::String
end



function SolverSettings(;nroots=1, tol=1e-8, maxiter=100, verbose=0, package="arpack")
    return SolverSettings(nroots, tol, maxiter, verbose, package)
end

function solve(problem::P, ints::InCoreInts{T}, S::SolverSettings; v0=nothing) where {T, P<:Problem}

    #e = Vector{T}([])
    #v = Matrix{T}([])

    if lowercase(S.package) == "arpack"

        Hmap = LinearMap(ints, problem)

        if v0 == nothing
            e,v = Arpack.eigs(Hmap, nev = S.nroots, which=:SR, tol=S.tol)
        else
            e,v = Arpack.eigs(Hmap, v0=v0[:,1], nev = S.nroots, which=:SR, tol=S.tol)
        end
        return Solution{P,T}(problem, e, v)


    elseif lowercase(S.package) == "krylovkit"
        
        error("NYI")

#        Hmap = LinOp(ints, problem)
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
#        return Solution{P,T}(problem, e, v)
    end
end

