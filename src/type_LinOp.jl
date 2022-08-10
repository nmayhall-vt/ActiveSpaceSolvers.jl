import LinearAlgebra: issymmetric, ishermitian

"""
This is just a cheap alternative to LinearMap that lets us have more control over things
    
    - matvec
    - dim::Int
    - sym::Bool
    - hermitian::Bool
"""
struct LinOp{T} <: AbstractMatrix{T} 
    matvec
    dim::Int
    sym::Bool
    hermitian::Bool
end

Base.size(lop::LinOp{T}) where {T} = return (lop.dim,lop.dim)
Base.:(*)(lop::LinOp{T}, v::AbstractVector{T}) where {T} = return lop.matvec(v)
Base.:(*)(lop::LinOp{T}, v::AbstractMatrix{T}) where {T} = return lop.matvec(v)
issymmetric(lop::LinOp)  = return lop.sym
ishermitian(lop::LinOp)  = return lop.hermitian
