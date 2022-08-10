using Printf

"""
This type contains both the Problem and the resulting Basis vectors.

    - problem::P
    - vectors::Matrix{T}

"""
struct Solution{P,T} 
    problem::P
    energies::Vector{T}
    vectors::Matrix{T}
end


Base.size(S::Solution) = size(S.vectors)

function Base.display(S::Solution)
    println()
    println(" Energies of Solution")
    display(S.problem)
    @printf(" %5s %12s\n", "State", "Energy")
    @printf("-------------------\n")
    for i in 1:length(S.energies)
        @printf(" %5i %16.12f\n", i, S.energies[i])
    end
    println()
end
