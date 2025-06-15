# This is heavily inspired by https://github.com/bifurcationkit/BifurcationKit.jl/blob/master/src/EigSolver.jl 
# under the MIT "Expat" License

module EigenValueSolvers

import Arpack
using LinearAlgebra
using Parameters
using ArgCheck
import LinearMaps
import KrylovKit
import ArnoldiMethod

export DefaultEig, EigArpack, EigKrylovKit, EigArnoldiMethod

abstract type AbstractEigenSolver end
abstract type AbstractDirectEigenSolver <: AbstractEigenSolver end
abstract type AbstractIterativeEigenSolver <: AbstractEigenSolver end
abstract type AbstractMFEigenSolver <: AbstractIterativeEigenSolver end

function geteigenvector(
        eigsolve::ES, vecs, n::Union{Int, AbstractVector{Int64}}) where {ES <: AbstractEigenSolver}
    vecs[:, n]
end

# We can e.g. convert sparse matrices to dense here
__to_array_for_eig(x) = Array(x)
__to_array_for_eig(x::Array) = x

getsolver(eig::AbstractEigenSolver) = eig

# Also which = abs
@with_kw struct DefaultEig{T} <: AbstractDirectEigenSolver
    which::T = real
end

function (l::DefaultEig)(J, nev; kwargs...)
    # we convert to Array so we can call l on small sparse matrices
    F = eigen(__to_array_for_eig(J); sortby = l.which)
    nev2 = min(nev, length(F.values))

    # return F.values[end:-1:(end - nev2 + 1)], F.vectors[:, end:-1:(end - nev2 + 1)], true, 1
    return F.values[1:nev2], F.vectors[:, 1:nev2], true, 1
end

# General Eigenvalue Problem
function gev(l::DefaultEig, A, B, nev; kwargs...)
    # we convert to Array so we can call it on small sparse matrices
    F = eigen(__to_array_for_eig(A), __to_array_for_eig(B))
    return F.values, F.vectors
end

struct EigArpack{T, Tby, Tw} <: AbstractIterativeEigenSolver
    "Shift for Shift-Invert method with `(J - sigma⋅I)"
    sigma::T

    "Which eigen-element to extract :LR, :LM, ..."
    which::Symbol

    "Sorting function, default to real"
    by::Tby

    "Keyword arguments passed to EigArpack"
    kwargs::Tw
end

EigArpack(sigma = nothing, which = :LR; kwargs...) = EigArpack(sigma, which, real, kwargs)

function (l::EigArpack)(J, nev; kwargs...)
    @argcheck J isa AbstractMatrix

    λ, ϕ, ncv = Arpack.eigs(J; nev, which = l.which, sigma = l.sigma, l.kwargs...)
    Ind = sortperm(λ; by = l.by, rev = true)
    ncv < nev && @warn "$ncv eigenvalues have converged using Arpack.eigs, you requested $nev"
    return λ[Ind], ϕ[:, Ind], true, 1
end

function gev(l::EigArpack, A, B, nev; kwargs...)
    @argcheck A isa AbstractMatrix

    values, ϕ, ncv = Arpack.eigs(A, B; nev, sigma = l.sigma, which = l.which, l.kwargs...)
    return values, ϕ
end

@with_kw struct EigKrylovKit{T, vectype} <: AbstractMFEigenSolver
    "Krylov Dimension"
    dim::Int64 = KrylovDefaults.krylovdim[]

    "Tolerance"
    tol::T = 1e-4

    "Number of restarts"
    restart::Int64 = 200

    "Maximum number of iterations"
    maxiter::Int64 = KrylovDefaults.maxiter[]

    "Verbosity ∈ {0, 1, 2}"
    verbose::Int = 0

    "Which eigenvalues are looked for :LR (largest real), :LM, ..."
    which::Symbol = :LR

    "If the linear map is symmetric, only meaningful if T<:Real"
    issymmetric::Bool = false

    "If the linear map is hermitian"
    ishermitian::Bool = false

    "Example of vector to usen for Krylov iterations"
    x₀::vectype = nothing
end

function (l::EigKrylovKit{T, vectype})(J, _nev; kwargs...) where {T, vectype}
    kw = (verbosity = l.verbose,
        krylovdim = l.dim, maxiter = l.maxiter,
        tol = l.tol, issymmetric = l.issymmetric,
        ishermitian = l.ishermitian)
    if J isa AbstractMatrix && isnothing(l.x₀)
        nev = min(_nev, size(J, 1))
        vals, vec, info = KrylovKit.eigsolve(J, nev, l.which; kw...)
    else
        nev = min(_nev, length(l.x₀))
        vals, vec, info = KrylovKit.eigsolve(J, l.x₀, nev, l.which; kw...)
    end
    # (length(vals) != _nev) && (@warn "EigKrylovKit returned $(length(vals)) eigenvalues instead of the $_nev requested")
    info.converged == 0 && (@warn "KrylovKit.eigsolve solver did not converge")
    return vals, vec, info.converged > 0, info.numops
end

function geteigenvector(
        eigsolve::EigKrylovKit{T, vectype}, vecs, n::Union{Int, AbstractVector{Int64}}) where {T, vectype}
    vecs[n]
end

struct EigArnoldiMethod{T, Tby, Tw, Tkw, vectype} <: AbstractIterativeEigenSolver
    "Shift for Shift-Invert method"
    sigma::T

    "Which eigen-element to extract LR(), LM(), ..."
    which::Tw

    "How do we sort the computed eigenvalues, defaults to real"
    by::Tby

    "Key words arguments passed to EigArpack"
    kwargs::Tkw

    "Example of vector used for Krylov iterations"
    x₀::vectype
end

function EigArnoldiMethod(; sigma = nothing, which = ArnoldiMethod.LR(), x₀ = nothing, kwargs...)
    EigArnoldiMethod(sigma, which, real, kwargs, x₀)
end

function (l::EigArnoldiMethod)(J, nev; kwargs...)
    @argcheck J isa AbstractMatrix

    if isnothing(l.sigma)
        decomp, history = ArnoldiMethod.partialschur(J; nev, which = l.which,
            l.kwargs...)
    else
        F = lu(l.sigma * LinearAlgebra.I - J)
        Jmap = LinearMaps.LinearMap{eltype(J)}((y, x) -> ldiv!(y, F, x), size(J, 1);
            ismutating = true)
        decomp, history = ArnoldiMethod.partialschur(Jmap; nev, which = l.which,
            l.kwargs...)
    end

    λ, ϕ = ArnoldiMethod.partialeigen(decomp)
    # shift and invert
    if isnothing(l.sigma) == false
        λ .= @. l.sigma - 1 / λ
    end
    Ind = sortperm(λ; by = l.by, rev = true)
    ncv = length(λ)
    ncv < nev &&
        @warn "$ncv eigenvalues have converged using ArnoldiMethod.partialschur, you requested $nev"
    return λ[Ind], ϕ[:, Ind], history.converged, 1
end

function gev(l::EigArnoldiMethod, A, B, nev; kwargs...)
    @argcheck A isa AbstractMatrix

    # Solve Ax = λBx using Shift-invert method 
    # (A - σ⋅B)⁻¹ B⋅x = 1/(λ-σ)x
    σ = isnothing(l.sigma) ? 0 : l.sigma
    P = lu(A - σ * B)
    𝒯 = eltype(A)
    L = LinearMaps.LinearMap{𝒯}((y, x) -> ldiv!(y, P, B * x), size(A, 1); ismutating = true)
    decomp, history = ArnoldiMethod.partialschur(L; nev, which = l.which,
        l.kwargs...)
    vals, ϕ = ArnoldiMethod.partialeigen(decomp)
    values = @. 1 / vals + σ

    return values, ϕ
end

end
