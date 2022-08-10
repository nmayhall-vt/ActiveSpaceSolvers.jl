
"""
This type contains both the Problem and the resulting Basis vectors, 
i.e., how to find eigenstates of the hamiltonian

    - problem::P
    - vectors::Matrix{T}

"""
struct Solutions{P,T} <: AbstractMatrix{T} 
    problem::P
    vectors::Matrix{T}
end
