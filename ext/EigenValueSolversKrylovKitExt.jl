module EigenValueSolversKrylovKitExt

using EigenValueSolvers: EigenValueSolvers, EigKrylovKit, sortselect

import KrylovKit

function kwargs(l::EigKrylovKit)
    return (
        verbosity = l.verbose,
        krylovdim = l.dim,
        maxiter = l.maxiter,
        tol = l.tol,
        issymmetric = l.issymmetric,
        ishermitian = l.ishermitian,
    )
end

function EigenValueSolvers.eigsolve(l::EigKrylovKit, J, nev::Int; kw...)
    if J isa AbstractMatrix && isnothing(l.x₀)
        n = min(nev, size(J, 1))
        λ, ϕ, info = KrylovKit.eigsolve(J, n, l.which; kwargs(l)...)
    else
        isnothing(l.x₀) && throw(
            ArgumentError("EigKrylovKit needs a starting vector `x₀` if `J` is not an `AbstractMatrix`")
        )
        n = min(nev, length(l.x₀))
        λ, ϕ, info = KrylovKit.eigsolve(J, l.x₀, n, l.which; kwargs(l)...)
    end

    info.converged < n &&
        @warn "only $(info.converged) of the $n requested eigenvalues converged using KrylovKit.eigsolve"

    values, vectors = sortselect(λ, ϕ, n, l.which)
    return values, vectors, info.converged ≥ n, info.numops
end

function EigenValueSolvers.geneigsolve(l::EigKrylovKit, A, B, nev::Int; kw...)
    n = min(nev, size(A, 1))
    λ, ϕ, info = KrylovKit.geneigsolve((A, B), n, l.which; isposdef = l.isposdef, kwargs(l)...)

    info.converged < n &&
        @warn "only $(info.converged) of the $n requested eigenvalues converged using KrylovKit.geneigsolve"

    values, vectors = sortselect(λ, ϕ, n, l.which)
    return values, vectors, info.converged ≥ n, info.numops
end

end
