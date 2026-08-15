module EigenValueSolversIterativeSolversExt

using ArgCheck: @argcheck
using EigenValueSolvers: EigenValueSolvers, EigLOBPCG, sortselect

import IterativeSolvers

# LOBPCG picks eigenvalues by algebraic value, which is exactly the `:LR`/`:SR` distinction.
# `EigLOBPCG` rejects every other target already, so this covers all reachable cases.
largest(l::EigLOBPCG) = l.which === :LR

function kwargs(l::EigLOBPCG)
    tolerance = isnothing(l.tol) ? NamedTuple() : (tol = l.tol,)
    return (maxiter = l.maxiter, P = l.P, C = l.C, tolerance..., l.kwargs...)
end

# `B === nothing` is how IterativeSolvers spells the standard eigenvalue problem.
function lobpcg(l::EigLOBPCG, A, B, nev::Int)
    @argcheck A isa AbstractMatrix
    @argcheck isnothing(B) || B isa AbstractMatrix
    n = min(nev, size(A, 1))

    # IterativeSolvers `throw`s a bare `String` here, which is neither catchable by type nor
    # consistent with the rest of this package, so we check the condition up front.
    size(A, 1) < 3n && throw(
        ArgumentError(
            "LOBPCG is unstable unless the matrix is at least three times as large as `nev`, " *
                "got size $(size(A, 1)) for nev = $n; use `EigDefault` for a problem this small"
        )
    )

    result = if isnothing(l.X₀)
        IterativeSolvers.lobpcg(A, B, largest(l), n; kwargs(l)...)
    else
        IterativeSolvers.lobpcg(A, B, largest(l), l.X₀, n; kwargs(l)...)
    end

    result.converged ||
        @warn "LOBPCG did not converge within $(result.iterations) iterations, the residual norms are $(result.residual_norms)"

    values, vectors = sortselect(result.λ, result.X, n, l.which)
    return values, vectors, result.converged, result.iterations
end

EigenValueSolvers.eigsolve(l::EigLOBPCG, J, nev::Int; kwargs...) = lobpcg(l, J, nothing, nev)
EigenValueSolvers.geneigsolve(l::EigLOBPCG, A, B, nev::Int; kwargs...) = lobpcg(l, A, B, nev)

end
