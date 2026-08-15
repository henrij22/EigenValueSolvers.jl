"""
    EigDefault(which = :LR)
    EigDefault(; which = :LR)

Direct eigen solver based on `LinearAlgebra.eigen`. It computes the *full* spectrum and then
returns the `nev` eigenpairs selected by `which` (one of [`TARGETS`](@ref)). Sparse matrices
are converted to dense ones first, so this solver is meant for small problems and as a
reference for the iterative solvers.

Supports the generalized eigenvalue problem through [`gev`](@ref).
"""
struct EigDefault <: AbstractDirectEigenSolver
    "Which eigenvalues are looked for, one of [`TARGETS`](@ref)"
    which::Symbol

    EigDefault(which::Symbol) = new(checktarget(EigDefault, which))
end

EigDefault(; which::Symbol = :LR) = EigDefault(which)

function eigsolve(l::EigDefault, J, nev::Int; kwargs...)
    F = eigen(toarray(J))
    λ, ϕ = sortselect(F.values, F.vectors, nev, l.which)
    return λ, ϕ, length(λ) == min(nev, size(J, 1)), 1
end

function geneigsolve(l::EigDefault, A, B, nev::Int; kwargs...)
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
    return EigArpack(sigma, checktarget(EigArpack, which), kwargs)
end

backend(::EigArpack) = :Arpack

"""
    EigArnoldiMethod(; sigma = nothing, which = :LR, x₀ = nothing, factorize = lu, kwargs...)

Iterative eigen solver based on
[ArnoldiMethod.jl](https://github.com/JuliaLinearAlgebra/ArnoldiMethod.jl). Requires
`using ArnoldiMethod`.

If `sigma` is given, the shift-invert method `(sigma⋅I - J)⁻¹` is used and the `nev`
eigenvalues *closest to* `sigma` are computed; `which` then only determines the order in
which they are returned. Without a shift, `which` selects the eigenvalues as documented in
[`TARGETS`](@ref).

The `factorize` function is what turns the shifted matrix into something that can be solved
against; see the field documentation below for how to plug in a different sparse direct
solver.

Additional `kwargs` are forwarded to `ArnoldiMethod.partialschur`.

Supports the generalized eigenvalue problem through [`gev`](@ref), which always uses the
shift-invert method with a shift of `sigma === nothing ? 0 : sigma`.
"""
struct EigArnoldiMethod{Tsigma, Tkw, Tvec, Tfac} <: AbstractIterativeEigenSolver
    "Shift for the shift-invert method `(sigma⋅I - J)⁻¹`, or `nothing`"
    sigma::Tsigma

    "Which eigenvalues are looked for, one of [`TARGETS`](@ref)"
    which::Symbol

    "Keyword arguments passed to `ArnoldiMethod.partialschur`"
    kwargs::Tkw

    "Vector used to start the Krylov iterations, or `nothing`"
    x₀::Tvec

    """
    Function used to factorize the shifted matrix, defaults to `LinearAlgebra.lu`.

    It is called as `factorize(M)` and its result `F` only has to support
    `LinearAlgebra.ldiv!(y, F, x)`, so any of the following works:

    - `cholesky`, if the shifted matrix is known to be positive definite
    - `qr`, for the rank-deficient case
    - `M -> MUMPS.Mumps(M)` or a `Pardiso.jl` wrapper for large sparse problems,
      where the SuiteSparse `lu` becomes the bottleneck
    - any closure returning a preconditioner-like object, e.g. to reuse a factorization
      across several solver calls
    """
    factorize::Tfac
end

function EigArnoldiMethod(;
        sigma = nothing, which::Symbol = :LR, x₀ = nothing, factorize = lu, kwargs...
    )
    return EigArnoldiMethod(sigma, checktarget(EigArnoldiMethod, which), kwargs, x₀, factorize)
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
positive definite and `ishermitian` and `isposdef` to be set. `A` and `B` may be arbitrary
operators there as well, again provided that `x₀` is supplied.

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
        dim, tol, maxiter, verbose, checktarget(EigKrylovKit, which),
        issymmetric, ishermitian, isposdef, x₀
    )
end

backend(::EigKrylovKit) = :KrylovKit

"""
    EigLOBPCG(; which = :SR, tol = nothing, maxiter = 200, P = nothing, C = nothing, X₀ = nothing, kwargs...)

Iterative eigen solver based on the LOBPCG implementation of
[IterativeSolvers.jl](https://github.com/JuliaLinearAlgebra/IterativeSolvers.jl). Requires
`using IterativeSolvers`.

LOBPCG is restricted to *symmetric/hermitian* problems, and to a positive definite `B` in
the generalized case, but in exchange it converges on the smallest eigenvalues without any
factorization of the matrix. That makes it the solver of choice for large sparse problems
where the shift-invert methods of [`EigArpack`](@ref) and [`EigArnoldiMethod`](@ref) run out
of memory, especially in combination with a preconditioner `P`.

Because LOBPCG selects eigenvalues by algebraic value, only the targets `:LR` and `:SR` are
supported, see [`supportedtargets`](@ref).

The matrix has to be at least three times as large as `nev`, otherwise LOBPCG is unstable
and throws; use [`EigDefault`](@ref) for problems that small.

Supports the generalized eigenvalue problem through [`gev`](@ref).
"""
struct EigLOBPCG{T, TP, TC, TX, Tkw} <: AbstractIterativeEigenSolver
    "Which eigenvalues are looked for, either `:LR` or `:SR`"
    which::Symbol

    "Tolerance, or `nothing` for the IterativeSolvers default"
    tol::T

    "Maximum number of iterations"
    maxiter::Int

    "Preconditioner, applied as `ldiv!(P, x)`, or `nothing`"
    P::TP

    "Constraint matrix the eigenvectors are kept orthogonal to, or `nothing`"
    C::TC

    "Block of vectors used to start the iterations, or `nothing`"
    X₀::TX

    "Keyword arguments passed to `IterativeSolvers.lobpcg`"
    kwargs::Tkw
end

function EigLOBPCG(;
        which::Symbol = :SR, tol = nothing, maxiter::Int = 200,
        P = nothing, C = nothing, X₀ = nothing, kwargs...
    )
    @argcheck maxiter > 0
    return EigLOBPCG(checktarget(EigLOBPCG, which), tol, maxiter, P, C, X₀, kwargs)
end

supportedtargets(::Type{<:EigLOBPCG}) = (:LR, :SR)
backend(::EigLOBPCG) = :IterativeSolvers

"""
    EigGenericArpack(; which = :LR, kwargs...)

Iterative eigen solver based on
[GenericArpack.jl](https://github.com/dgleich/GenericArpack.jl), a pure Julia translation of
ARPACK. Requires `using GenericArpack`.

Compared to [`EigArpack`](@ref) this needs no compiled ARPACK library, and it is not
restricted to the four BLAS element types: it runs in `Float32`, `BigFloat` or any other
`AbstractFloat`, which makes it useful whenever the problem is too ill-conditioned for
double precision.

The implementation only covers *symmetric/hermitian* problems, so `:LI` and `:SI` are not
supported, see [`supportedtargets`](@ref). It also has no shift, use [`EigArpack`](@ref) or
[`EigArnoldiMethod`](@ref) if you need the shift-invert method.

Additional `kwargs` are forwarded to `GenericArpack.symeigs`.

Supports the generalized eigenvalue problem through [`gev`](@ref), which requires `B` to be
positive definite.
"""
struct EigGenericArpack{Tkw} <: AbstractIterativeEigenSolver
    "Which eigenvalues are looked for, one of `:LM`, `:SM`, `:LR` and `:SR`"
    which::Symbol

    "Keyword arguments passed to `GenericArpack.symeigs`"
    kwargs::Tkw
end

function EigGenericArpack(; which::Symbol = :LR, kwargs...)
    return EigGenericArpack(checktarget(EigGenericArpack, which), kwargs)
end

supportedtargets(::Type{<:EigGenericArpack}) = (:LM, :SM, :LR, :SR)
backend(::EigGenericArpack) = :GenericArpack
