module EigenValueSolversArpackExt

using ArgCheck: @argcheck
using EigenValueSolvers: EigenValueSolvers, EigArpack, sortselect

import Arpack

function EigenValueSolvers.eigsolve(l::EigArpack, J, nev::Int; kwargs...)
    @argcheck J isa AbstractMatrix
    n = min(nev, size(J, 1))

    # In shift-invert mode the target refers to the transformed problem, where the
    # eigenvalues closest to `sigma` are the dominant ones. `l.which` is applied afterwards.
    which = isnothing(l.sigma) ? l.which : :LM

    λ, ϕ, nconv, _, nmult = Arpack.eigs(J; nev = n, which = which, sigma = l.sigma, l.kwargs...)
    nconv < n && @warn "only $nconv of the $n requested eigenvalues converged using Arpack.eigs"

    values, vectors = sortselect(λ, ϕ, n, l.which)
    return values, vectors, nconv ≥ n, nmult
end

function EigenValueSolvers.geneigsolve(l::EigArpack, A, B, nev::Int; kwargs...)
    @argcheck A isa AbstractMatrix
    @argcheck B isa AbstractMatrix
    n = min(nev, size(A, 1))

    which = isnothing(l.sigma) ? l.which : :LM

    λ, ϕ, nconv, _, nmult = Arpack.eigs(A, B; nev = n, which = which, sigma = l.sigma, l.kwargs...)
    nconv < n && @warn "only $nconv of the $n requested eigenvalues converged using Arpack.eigs"

    values, vectors = sortselect(λ, ϕ, n, l.which)
    return values, vectors, nconv ≥ n, nmult
end

end
