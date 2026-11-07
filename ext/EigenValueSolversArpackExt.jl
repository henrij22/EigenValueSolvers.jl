module EigenValueSolversArpackExt

using ArgCheck: @argcheck
using EigenValueSolvers: EigenValueSolvers, EigArpack, ShiftInvert, sortselect
using LinearAlgebra: LinearAlgebra, isposdef, issymmetric, lu

import Arpack

# ARPACK sorts :LI/:SI of real problems by |imag| (dsortc). The spectrum is symmetric to the
# real axis, so the 2nev eigenvalues of largest |imag| contain the wanted ones and
# `sortselect` picks them. That fails if 2nev had to be cut at the n - 2 ARPACK can compute.
isrealimag(which::Symbol, matrices...) = which in (:LI, :SI) && all(M -> eltype(M) <: Real, matrices)

function imagnev(nev::Int, n::Int)
    2nev ≤ n - 2 || throw(
        ArgumentError(
            "EigArpack needs 2nev ≤ n - 2 for `:LI`/`:SI` of real problems, got nev = $nev for " *
                "n = $n, use `EigDefault` for problems this small"
        )
    )
    return 2nev
end

symmetricimag(which::Symbol) = ArgumentError(
    "EigArpack: the target `:$which` is meaningless for a real symmetric problem " *
        "(with positive definite `B` if generalized), all its eigenvalues are real"
)

# The generalized modes of ARPACK need a positive definite `B`, so solve B⁻¹A⋅x = λ⋅x instead.
function inverseB(A, B, nev::Int, kwargs)
    n = size(A, 1)
    op = ShiftInvert(lu(B), A, n, Vector{promote_type(eltype(A), eltype(B))}(undef, n))
    λ, ϕ, nconv, _, nmult = Arpack.eigs(op; nev = nev, which = :LI, kwargs...)
    return λ, ϕ, nconv, nmult
end

# `Arpack.eigs(A, B; sigma)` swaps :LM and :SM for generalized problems by default. The
# symmetric case is therefore solved with `explicittransform = :none`, all others with the
# shift-invert transformation (A - σB)⁻¹B applied explicitly. Their eigenvalues are complex.
function shiftinvert(A, B, σ, nev::Int, kwargs)
    isreal = eltype(A) <: Real && eltype(B) <: Real
    @argcheck !isreal || σ isa Real "complex shifts are not supported for real problems"
    if isreal && issymmetric(A) && isposdef(B)
        λ, ϕ, nconv, _, nmult = Arpack.eigs(
            A, B; nev = nev, which = :LM, sigma = σ, explicittransform = :none, kwargs...
        )
        return λ, ϕ, nconv, nmult
    end
    M = A - σ * B
    n = size(A, 1)
    op = ShiftInvert(lu(M), B, n, Vector{eltype(M)}(undef, n))
    μ, ϕ, nconv, _, nmult = Arpack.eigs(op; nev = nev, which = :LM, kwargs...)
    return (@. σ + inv(μ)), ϕ, nconv, nmult
end

function EigenValueSolvers.eigsolve(l::EigArpack, J, nev::Int; kwargs...)
    @argcheck J isa AbstractMatrix
    n = min(nev, size(J, 1))

    if !isnothing(l.sigma)
        # In shift-invert mode the target refers to the transformed problem, where the
        # eigenvalues closest to `sigma` are the dominant ones. `l.which` is applied afterwards.
        which, narpack = :LM, n
    elseif isrealimag(l.which, J)
        issymmetric(J) && throw(symmetricimag(l.which))
        which, narpack = :LI, imagnev(n, size(J, 1))
    else
        which, narpack = l.which, n
    end

    λ, ϕ, nconv, _, nmult = Arpack.eigs(J; nev = narpack, which = which, sigma = l.sigma, l.kwargs...)
    nconv < n && @warn "only $nconv of the $n requested eigenvalues converged using Arpack.eigs"

    values, vectors = sortselect(λ, ϕ, n, l.which)
    return values, vectors, nconv ≥ n, nmult
end

function EigenValueSolvers.geneigsolve(l::EigArpack, A, B, nev::Int; kwargs...)
    @argcheck A isa AbstractMatrix
    @argcheck B isa AbstractMatrix
    n = min(nev, size(A, 1))

    if !isnothing(l.sigma)
        λ, ϕ, nconv, nmult = shiftinvert(A, B, l.sigma, n, l.kwargs)
    elseif isrealimag(l.which, A, B) && !isposdef(B)
        λ, ϕ, nconv, nmult = inverseB(A, B, imagnev(n, size(A, 1)), l.kwargs)
    else
        which, narpack = l.which, n
        if isrealimag(l.which, A, B)
            issymmetric(A) && throw(symmetricimag(l.which))
            which, narpack = :LI, imagnev(n, size(A, 1))
        end
        λ, ϕ, nconv, _, nmult = Arpack.eigs(A, B; nev = narpack, which = which, l.kwargs...)
    end
    nconv < n && @warn "only $nconv of the $n requested eigenvalues converged using Arpack.eigs"

    values, vectors = sortselect(λ, ϕ, n, l.which)
    return values, vectors, nconv ≥ n, nmult
end

end
