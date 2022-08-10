using ActiveSpaceSolvers


struct FCIAnsatz <: Ansatz 
    no::Int  # number of orbitals
    na::Int  # number of alpha
    nb::Int  # number of beta
    dima::Int 
    dimb::Int 
    dim::Int
    converged::Bool
    restarted::Bool
    iteration::Int
    algorithm::String   #  options: direct/davidson
    n_roots::Int
end

function FCIAnsatz(no, na, nb)
    na <= no || throw(DimensionMismatch)
    nb <= no || throw(DimensionMismatch)
    dima = calc_nchk(no,na)
    dimb = calc_nchk(no,nb)
    return FCIAnsatz(no, na, nb, dima, dimb, dima*dimb, false, false, 1, "direct", 1)
end

function display(p::FCIAnsatz)
    @printf(" FCIAnsatz:: #Orbs = %-3i #α = %-2i #β = %-2i Dimension: %-9i\n",p.no,p.na,p.nb,p.dim)
end

