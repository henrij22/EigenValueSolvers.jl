module EigenValueSolversGenericArpackExt

using ArgCheck: @argcheck
using EigenValueSolvers: EigenValueSolvers, EigGenericArpack, sortselect

import GenericArpack

# Symmetric ARPACK selects by *algebraic* value under the names `:LA`/`:SA`, and has no
# notion of `:LR`/`:SR` -- for a real spectrum these mean the same thing. `:LM`/`:SM` carry
# over unchanged, and `EigGenericArpack` rejects `:LI`/`:SI` already.
function target(which::Symbol)
    which === :LR && return :LA
    which === :SR && return :SA
    return which
end

function symeigs(l::EigGenericArpack, matrices, nev::Int)
    A = first(matrices)
    @argcheck all(M -> M isa AbstractMatrix, matrices)
    n = min(nev, size(A, 1))

    # ARPACK reports the operator applications it needed in `nopx`.
    stats = GenericArpack.ArpackStats()

    # Without `failonmaxiter = false` GenericArpack throws instead of returning the
    # eigenpairs that did converge, which would defeat the `converged` return value.
    result = GenericArpack.symeigs(
        matrices..., n; which = target(l.which), stats = stats, failonmaxiter = false, l.kwargs...
    )

    nconv = length(result.values)
    nconv < n && @warn "only $nconv of the $n requested eigenvalues converged using GenericArpack.symeigs"

    values, vectors = sortselect(result.values, result.vectors, n, l.which)
    return values, vectors, nconv ≥ n, stats.nopx
end

EigenValueSolvers.eigsolve(l::EigGenericArpack, J, nev::Int; kwargs...) = symeigs(l, (J,), nev)
EigenValueSolvers.geneigsolve(l::EigGenericArpack, A, B, nev::Int; kwargs...) = symeigs(l, (A, B), nev)

end
