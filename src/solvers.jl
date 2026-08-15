"""
    DefaultEig(which = :LR)
    DefaultEig(; which = :LR)

Direct eigen solver based on `LinearAlgebra.eigen`. It computes the *full* spectrum and then
returns the `nev` eigenpairs selected by `which` (one of [`TARGETS`](@ref)). Sparse matrices
are converted to dense ones first, so this solver is meant for small problems and as a
reference for the iterative solvers.

Supports the generalized eigenvalue problem through [`gev`](@ref).
"""
struct DefaultEig <: AbstractDirectEigenSolver
    "Which eigenvalues are looked for, one of [`TARGETS`](@ref)"
    which::Symbol

    DefaultEig(which::Symbol) = new(checktarget(which))
end

DefaultEig(; which::Symbol = :LR) = DefaultEig(which)

function eigsolve(l::DefaultEig, J, nev::Int; kwargs...)
    F = eigen(toarray(J))
    λ, ϕ = sortselect(F.values, F.vectors, nev, l.which)
    return λ, ϕ, length(λ) == min(nev, size(J, 1)), 1
end

function geneigsolve(l::DefaultEig, A, B, nev::Int; kwargs...)
    F = eigen(toarray(A), toarray(B))
    λ, ϕ = sortselect(F.values, F.vectors, nev, l.which)
    return λ, ϕ, length(λ) == min(nev, size(A, 1)), 1
end

"""
    EigArpack(sigma = nothing, which = :LR; kwargs...)

Iterative eigen solver based on [Arpack.jl](https://github.com/JuliaLinearAlgebra/Arpack.jl).
Requires `using Arpack`.

If `sigma` is given, the shift-invert method `(J - sigma⋅I)⁻¹` is used and the `nev`
eigenvalues *closest to* `sigma` are computed; `which` then only determines the order in
which they are returned. Without a shift, `which` selects the eigenvalues as documented in
[`TARGETS`](@ref).

Additional `kwargs` are forwarded to `Arpack.eigs`.

Supports the generalized eigenvalue problem through [`gev`](@ref).
"""
struct EigArpack{Tsigma, Tkw} <: AbstractIterativeEigenSolver
    "Shift for the shift-invert method `(J - sigma⋅I)⁻¹`, or `nothing`"
    sigma::Tsigma

    "Which eigenvalues are looked for, one of [`TARGETS`](@ref)"
    which::Symbol

    "Keyword arguments passed to `Arpack.eigs`"
    kwargs::Tkw
end

function EigArpack(sigma = nothing, which::Symbol = :LR; kwargs...)
    return EigArpack(sigma, checktarget(which), kwargs)
end

backend(::EigArpack) = :Arpack

"""
    EigArnoldiMethod(; sigma = nothing, which = :LR, x₀ = nothing, kwargs...)

Iterative eigen solver based on
[ArnoldiMethod.jl](https://github.com/JuliaLinearAlgebra/ArnoldiMethod.jl). Requires
`using ArnoldiMethod` (and, for `sigma !== nothing`, `using LinearMaps`).

If `sigma` is given, the shift-invert method `(sigma⋅I - J)⁻¹` is used and the `nev`
eigenvalues *closest to* `sigma` are computed; `which` then only determines the order in
which they are returned. Without a shift, `which` selects the eigenvalues as documented in
[`TARGETS`](@ref).

Additional `kwargs` are forwarded to `ArnoldiMethod.partialschur`.

Supports the generalized eigenvalue problem through [`gev`](@ref), which always uses the
shift-invert method with a shift of `sigma === nothing ? 0 : sigma`.
"""
struct EigArnoldiMethod{Tsigma, Tkw, Tvec} <: AbstractIterativeEigenSolver
    "Shift for the shift-invert method `(sigma⋅I - J)⁻¹`, or `nothing`"
    sigma::Tsigma

    "Which eigenvalues are looked for, one of [`TARGETS`](@ref)"
    which::Symbol

    "Keyword arguments passed to `ArnoldiMethod.partialschur`"
    kwargs::Tkw

    "Vector used to start the Krylov iterations, or `nothing`"
    x₀::Tvec
end

function EigArnoldiMethod(; sigma = nothing, which::Symbol = :LR, x₀ = nothing, kwargs...)
    return EigArnoldiMethod(sigma, checktarget(which), kwargs, x₀)
end

backend(::EigArnoldiMethod) = :ArnoldiMethod

"""
    EigKrylovKit(; kwargs...)

Matrix-free iterative eigen solver based on
[KrylovKit.jl](https://github.com/Jutho/KrylovKit.jl). Requires `using KrylovKit`.

In contrast to the other solvers, `J` does not have to be an `AbstractMatrix`: any object
`J` for which `J * x` is defined works, in which case a starting vector `x₀` has to be
supplied. The eigenvectors are then returned as a vector of vectors, use
[`geteigenvector`](@ref) to access them.

Supports the generalized eigenvalue problem through [`gev`](@ref), which requires `B` to be
positive definite.

See the fields below for the available keyword arguments.
"""
struct EigKrylovKit{T, Tvec} <: AbstractMFEigenSolver
    "Krylov dimension"
    dim::Int

    "Tolerance"
    tol::T

    "Maximum number of iterations"
    maxiter::Int

    "Verbosity ∈ {0, 1, 2, 3}"
    verbose::Int

    "Which eigenvalues are looked for, one of [`TARGETS`](@ref)"
    which::Symbol

    "Whether the linear map is symmetric, only meaningful if the eltype is real"
    issymmetric::Bool

    "Whether the linear map is hermitian"
    ishermitian::Bool

    "Whether `B` in the generalized eigenvalue problem is positive definite, see [`gev`](@ref)"
    isposdef::Bool

    "Vector used to start the Krylov iterations, required if `J` is not an `AbstractMatrix`"
    x₀::Tvec
end

function EigKrylovKit(;
        dim::Int = 30,
        tol = 1.0e-4,
        maxiter::Int = 100,
        verbose::Int = 0,
        which::Symbol = :LR,
        issymmetric::Bool = false,
        ishermitian::Bool = false,
        isposdef::Bool = false,
        x₀ = nothing,
    )
    @argcheck dim > 0
    @argcheck maxiter > 0
    @argcheck tol > 0
    return EigKrylovKit(
        dim, tol, maxiter, verbose, checktarget(which), issymmetric, ishermitian, isposdef, x₀
    )
end

backend(::EigKrylovKit) = :KrylovKit
