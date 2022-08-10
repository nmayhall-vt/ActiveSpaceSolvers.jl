using ActiveSpaceSolvers
import LinearMaps

"""
    LinearMap(ints, prb::FCIAnsatz)

Get LinearMap with takes a vector and returns action of H on that vector

# Arguments
- ints: `InCoreInts` object
- prb:  `FCIAnsatz` object
"""
function LinearMaps.LinearMap(ints::InCoreInts, prb::FCIAnsatz)
    #={{{=#
    ket_a = DeterminantString(prb.no, prb.na)
    ket_b = DeterminantString(prb.no, prb.nb)

    #@btime lookup_a = $fill_ca_lookup2($ket_a)
    lookup_a = fill_ca_lookup2(ket_a)
    lookup_b = fill_ca_lookup2(ket_b)
    iters = 0
    function mymatvec(v)
        iters += 1
        #@printf(" Iter: %4i\n", iters)
        nr = 0
        if length(size(v)) == 1
            nr = 1
            v = reshape(v,ket_a.max*ket_b.max, nr)
        else 
            nr = size(v)[2]
        end
        v = reshape(v, ket_a.max, ket_b.max, nr)
        sig = compute_ab_terms2(v, ints, prb, lookup_a, lookup_b)
        sig += compute_ss_terms2(v, ints, prb, lookup_a, lookup_b)

        v = reshape(v, ket_a.max*ket_b.max, nr)
        sig = reshape(sig, ket_a.max*ket_b.max, nr)
        return sig 
    end
    return LinearMap(mymatvec, prb.dim, prb.dim; issymmetric=true, ismutating=false, ishermitian=true)
end
#=}}}=#


"""
    build_H_matrix(ints, P::FCIAnsatz)

Build the Hamiltonian defined by `ints` in the Slater Determinant Basis 
in the sector of Fock space specified by `P`
"""
function ActiveSpaceSolvers.build_H_matrix(ints, P::FCIAnsatz)
#={{{=#
    T = eltype(ints.h0)

    Hmat = zeros(P.dim, P.dim)

    Hdiag_a = precompute_spin_diag_terms(ints,P,P.na)
    Hdiag_b = precompute_spin_diag_terms(ints,P,P.nb)
    # 
    #   Create ci_strings
    ket_a = DeterminantString(P.no, P.na)
    ket_b = DeterminantString(P.no, P.nb)
    bra_a = DeterminantString(P.no, P.na)
    bra_b = DeterminantString(P.no, P.nb)
    #   
    #   Add spin diagonal components
    Hmat += kron(Matrix(1.0I, P.dimb, P.dimb), Hdiag_a)
    Hmat += kron(Hdiag_b, Matrix(1.0I, P.dima, P.dima))
    #
    #   Add opposite spin term (todo: make this reasonably efficient)
    Hmat += compute_ab_terms_full(ints, P, T=T)
    
    Hmat = .5*(Hmat+Hmat')

    return Hmat
end
#=}}}=#
