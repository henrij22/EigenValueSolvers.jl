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

function needsstartvector(name)
    return throw(
        ArgumentError("EigKrylovKit needs a starting vector `x₀` if `$name` is not an `AbstractMatrix`")
    )
end

function EigenValueSolvers.eigsolve(l::EigKrylovKit, J, nev::Int; kw...)
    if J isa AbstractMatrix && isnothing(l.x₀)
        n = min(nev, size(J, 1))
        λ, ϕ, info = KrylovKit.eigsolve(J, n, l.which; kwargs(l)...)
    else
        isnothing(l.x₀) && needsstartvector("J")
        n = min(nev, length(l.x₀))
        λ, ϕ, info = KrylovKit.eigsolve(J, l.x₀, n, l.which; kwargs(l)...)
    end

    info.converged < n &&
        @warn "only $(info.converged) of the $n requested eigenvalues converged using KrylovKit.eigsolve"

    values, vectors = sortselect(λ, ϕ, n, l.which)
    return values, vectors, info.converged ≥ n, info.numops
end

function EigenValueSolvers.geneigsolve(l::EigKrylovKit, A, B, nev::Int; kw...)
    kws = (isposdef = l.isposdef, kwargs(l)...)

    # As for the standard problem, `A` and `B` may be arbitrary operators, in which case
    # KrylovKit cannot generate a starting vector itself and `x₀` has to be supplied.
    if A isa AbstractMatrix && isnothing(l.x₀)
        n = min(nev, size(A, 1))
        λ, ϕ, info = KrylovKit.geneigsolve((A, B), n, l.which; kws...)
    else
        isnothing(l.x₀) && needsstartvector("A")
        n = min(nev, length(l.x₀))
        λ, ϕ, info = KrylovKit.geneigsolve((A, B), l.x₀, n, l.which; kws...)
    end

    info.converged < n &&
        @warn "only $(info.converged) of the $n requested eigenvalues converged using KrylovKit.geneigsolve"

    values, vectors = sortselect(λ, ϕ, n, l.which)
    return values, vectors, info.converged ≥ n, info.numops
end

end
