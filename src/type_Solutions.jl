
"""
This abstract type contains both the Problem and the resulting Basis vectors, 
i.e., how to find eigenstates of the hamiltonian
"""
struct Solutions{P,T} <: AbstractMatrix{T} 
    problem::P
    vectors::Matrix{T}
end
