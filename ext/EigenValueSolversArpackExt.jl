module EigenValueSolversArpackExt

using ArgCheck: @argcheck
using EigenValueSolvers: EigenValueSolvers, EigArpack, ShiftInvert, sortselect
using LinearAlgebra: LinearAlgebra, issymmetric, lu

import Arpack

"""
    arpacktarget(which, nev, n, isreal) -> (which, nev)

Translate the target `which` of the package into the `which`/`nev` pair handed to
`Arpack.eigs` for a problem of size `n` without a shift.

For real (nonsymmetric) problems ARPACK sorts `:LI`/`:SI` by the *magnitude* of the
imaginary part (see `dsortc`), i.e. `:LI` returns the eigenvalues farthest from the real
axis and `:SI` the (nearly) real ones. The spectrum of a real problem is symmetric with
respect to the real axis, so the `nev` eigenvalues with the largest (smallest) imaginary
part are among the `2nev` eigenvalues with the largest imaginary part in magnitude. These
are requested and [`sortselect`](@ref EigenValueSolvers.sortselect) picks the right ones.

For a real symmetric problem all eigenvalues are real, so `:LI`/`:SI` are meaningless and
throw (`issymmetric` is the check `Arpack.eigs` itself uses to pick the symmetric driver).
"""
function arpacktarget(which::Symbol, nev::Int, n::Int, isreal::Bool, issym)
    if isreal && which in (:LI, :SI)
        issym() && throw(
            ArgumentError(
                "EigArpack: the target `:$which` is meaningless for a real symmetric problem, " *
                    "all its eigenvalues are real"
            )
        )
        return :LI, min(2 * nev, n - 2)
    end
    return which, nev
end

"""
    shiftinvert(A, B, σ, nev, kwargs) -> (λ, ϕ, nconv, nmult)

Solve `A⋅x = λ⋅B⋅x` for the `nev` eigenvalues closest to `σ` with `Arpack.eigs`, doing the
shift-invert transformation explicitly:

    (A - σ⋅B)⁻¹ B⋅x = 1 / (λ - σ) ⋅ x

so the wanted eigenvalues are the dominant ones of the operator.

This is needed since `Arpack.eigs(A, B; sigma)` swaps the meaning of `:LM` and `:SM` for
generalized problems (with its default `explicittransform = :auto` it returns the
eigenvalues *farthest* from `sigma` for `which = :LM`).
"""
function shiftinvert(A, B, σ, nev::Int, kwargs)
    M = A - σ * B
    n = size(A, 1)
    op = ShiftInvert(lu(M), B, n, Vector{eltype(M)}(undef, n))
    μ, ϕ, nconv, _, nmult = Arpack.eigs(op; nev = nev, which = :LM, kwargs...)
    return (@. σ + inv(μ)), ϕ, nconv, nmult
end

function EigenValueSolvers.eigsolve(l::EigArpack, J, nev::Int; kwargs...)
    @argcheck J isa AbstractMatrix
    n = min(nev, size(J, 1))

    if isnothing(l.sigma)
        which, narpack = arpacktarget(l.which, n, size(J, 1), eltype(J) <: Real, () -> issymmetric(J))
    else
        # In shift-invert mode the target refers to the transformed problem, where the
        # eigenvalues closest to `sigma` are the dominant ones. `l.which` is applied afterwards.
        which, narpack = :LM, n
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

    if isnothing(l.sigma)
        isreal = eltype(A) <: Real && eltype(B) <: Real
        which, narpack = arpacktarget(l.which, n, size(A, 1), isreal, () -> issymmetric(A) && issymmetric(B))
        λ, ϕ, nconv, _, nmult = Arpack.eigs(A, B; nev = narpack, which = which, l.kwargs...)
    else
        λ, ϕ, nconv, nmult = shiftinvert(A, B, l.sigma, n, l.kwargs)
    end
    nconv < n && @warn "only $nconv of the $n requested eigenvalues converged using Arpack.eigs"

    values, vectors = sortselect(λ, ϕ, n, l.which)
    return values, vectors, nconv ≥ n, nmult
end

end
