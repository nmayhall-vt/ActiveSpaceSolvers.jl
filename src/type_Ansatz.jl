using ActiveSpaceSolvers

"""
This abstract type contains all the metadata that defines a CI basis
for example, if we subtype a `FCIAnsatz`, then this contains the metadata
to diagonalize the Hamiltonian in the FCI basis.  
Diagonalization of H in a RASCI determinant basis would then be a different 
subtype.
In the future we may wish to extend this further to things like coupled cluster 
CIPSI, or MPS but not sure yet. 
"""
abstract type Ansatz end

Base.length(a::Ansatz) = a.dim
