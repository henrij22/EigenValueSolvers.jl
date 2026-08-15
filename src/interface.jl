"""
    AbstractEigenSolver

Supertype of all eigenvalue solvers.

Every solver `l::AbstractEigenSolver` is callable and implements the *standard* eigenvalue
problem `J⋅x = λ⋅x`

    l(J, nev; kwargs...) -> (values, vectors, converged, numops)

and, if the underlying backend supports it, the *generalized* eigenvalue problem
`A⋅x = λ⋅B⋅x` via [`gev`](@ref)

    gev(l, A, B, nev; kwargs...) -> (values, vectors, converged, numops)

Both return the same four elements:

| element     | meaning                                                                  |
|:------------|:-------------------------------------------------------------------------|
| `values`    | at most `nev` eigenvalues, ordered according to the solver's `which`      |
| `vectors`   | the matching eigenvectors, see [`geteigenvector`](@ref)                   |
| `converged` | `true` if at least `nev` eigenpairs converged                             |
| `numops`    | number of operator applications used by the backend                       |

`vectors` is a matrix whose columns are the eigenvectors for all solvers except the
matrix-free ones (see [`AbstractMFEigenSolver`](@ref)), which may return a vector of
vectors. Use [`geteigenvector`](@ref) to access an eigenvector independently of the solver.
"""
abstract type AbstractEigenSolver end

"""
    AbstractDirectEigenSolver <: AbstractEigenSolver

Solvers that compute the full spectrum and then select `nev` eigenpairs from it, e.g.
[`DefaultEig`](@ref).
"""
abstract type AbstractDirectEigenSolver <: AbstractEigenSolver end

"""
    AbstractIterativeEigenSolver <: AbstractEigenSolver

Solvers that compute only a few eigenpairs iteratively, e.g. [`EigArpack`](@ref).
"""
abstract type AbstractIterativeEigenSolver <: AbstractEigenSolver end

"""
    AbstractMFEigenSolver <: AbstractIterativeEigenSolver

Matrix-free iterative solvers, i.e. solvers that only need the action `x -> J⋅x` instead of
an `AbstractMatrix`, e.g. [`EigKrylovKit`](@ref).
"""
abstract type AbstractMFEigenSolver <: AbstractIterativeEigenSolver end

"""
    TARGETS

The eigenvalue targets accepted by the `which` field of every solver:

| target | selects the eigenvalues with ... | ordering of the result |
|:-------|:---------------------------------|:-----------------------|
| `:LM`  | largest magnitude                | descending in `abs`    |
| `:SM`  | smallest magnitude               | ascending in `abs`     |
| `:LR`  | largest real part                | descending in `real`   |
| `:SR`  | smallest real part               | ascending in `real`    |
| `:LI`  | largest imaginary part           | descending in `imag`   |
| `:SI`  | smallest imaginary part          | ascending in `imag`    |
"""
const TARGETS = (:LM, :SM, :LR, :SR, :LI, :SI)

"""
    ordering(which::Symbol) -> (by, rev)

Translate a target from [`TARGETS`](@ref) into the `by`/`rev` pair of `sortperm` that brings
the eigenvalues into the order documented for that target.
"""
function ordering(which::Symbol)
    which === :LM && return (abs, true)
    which === :SM && return (abs, false)
    which === :LR && return (real, true)
    which === :SR && return (real, false)
    which === :LI && return (imag, true)
    which === :SI && return (imag, false)
    throw(ArgumentError("unknown target `which = :$which`, must be one of $TARGETS"))
end

checktarget(which::Symbol) = (ordering(which); return which)

# Eigenvectors are stored either as the columns of a matrix or as a vector of vectors.
selectvectors(ϕ::AbstractMatrix, idx) = ϕ[:, idx]
selectvectors(ϕ::AbstractVector, idx) = ϕ[idx]

"""
    sortselect(λ, ϕ, nev, which) -> (values, vectors)

Sort the eigenpairs `(λ, ϕ)` according to the target `which` and keep the leading
`min(nev, length(λ))` of them. This is what makes the output of all solvers consistent: a
backend may return more eigenpairs than requested, or return them in its own order.
"""
function sortselect(λ, ϕ, nev::Int, which::Symbol)
    by, rev = ordering(which)
    idx = sortperm(λ; by = by, rev = rev)[1:min(nev, length(λ))]
    return λ[idx], selectvectors(ϕ, idx)
end

"""
    geteigenvector(solver::AbstractEigenSolver, vectors, n) -> v

Return the `n`-th eigenvector out of the `vectors` returned by `solver`.

`n` may also be an `AbstractVector{Int}`, in which case the corresponding collection of
eigenvectors is returned. Going through this function makes the calling code independent of
whether the solver stores the eigenvectors as the columns of a matrix or as a vector of
vectors.
"""
geteigenvector(::AbstractEigenSolver, vectors::AbstractMatrix, n::Union{Int, AbstractVector{Int}}) = vectors[:, n]
geteigenvector(::AbstractEigenSolver, vectors::AbstractVector, n::Union{Int, AbstractVector{Int}}) = vectors[n]

"""
    getsolver(l::AbstractEigenSolver) -> AbstractEigenSolver

Return the eigen solver used by `l`. Provided so that wrapper types can forward to the
solver they hold; for a plain solver this is the identity.
"""
getsolver(l::AbstractEigenSolver) = l

# We convert e.g. sparse matrices to dense ones here, so that the direct solvers can be
# called on them as well.
toarray(x) = Array(x)
toarray(x::Array) = x

"""
    gev(l::AbstractEigenSolver, A, B, nev; kwargs...) -> (values, vectors, converged, numops)

Solve the generalized eigenvalue problem `A⋅x = λ⋅B⋅x` for `nev` eigenpairs, see
[`AbstractEigenSolver`](@ref) for the meaning of the return values.
"""
gev(l::AbstractEigenSolver, A, B, nev::Int; kwargs...) = geneigsolve(l, A, B, nev; kwargs...)

(l::AbstractEigenSolver)(J, nev::Int; kwargs...) = eigsolve(l, J, nev; kwargs...)

# `eigsolve`/`geneigsolve` are the extension points. Solvers backed by a weak dependency
# only get a method once that dependency is loaded, so the fallbacks below explain what is
# missing instead of failing with a `MethodError`.
"""
    backend(l::AbstractEigenSolver) -> Union{Nothing, Symbol}

The package that has to be loaded for `l` to work, or `nothing` if `l` needs no weak
dependency.
"""
backend(::AbstractEigenSolver) = nothing

function missingbackend(l::AbstractEigenSolver, what::String)
    pkg = backend(l)
    isnothing(pkg) && throw(ArgumentError("$(nameof(typeof(l))) does not support $what"))
    return throw(
        ArgumentError(
            "$(nameof(typeof(l))) requires $pkg.jl to be loaded, add it to your project and " *
                "run `using $pkg` to load the corresponding package extension"
        )
    )
end

"""
    EigenValueSolvers.eigsolve(l::AbstractEigenSolver, J, nev; kwargs...)

The method a solver has to implement for the standard eigenvalue problem `J⋅x = λ⋅x`. It is
what `l(J, nev; kwargs...)` forwards to, and it has to return the four elements documented
in [`AbstractEigenSolver`](@ref).

The fallback throws, either because the package extension providing the backend of `l` is
not loaded, or because `l` does not support standard eigenvalue problems at all.
"""
eigsolve(l::AbstractEigenSolver, J, nev::Int; kwargs...) = missingbackend(l, "standard eigenvalue problems")

"""
    EigenValueSolvers.geneigsolve(l::AbstractEigenSolver, A, B, nev; kwargs...)

The method a solver has to implement for the generalized eigenvalue problem `A⋅x = λ⋅B⋅x`.
It is what [`gev`](@ref) forwards to, and it has to return the four elements documented in
[`AbstractEigenSolver`](@ref).

The fallback throws, either because the package extension providing the backend of `l` is
not loaded, or because `l` does not support generalized eigenvalue problems at all.
"""
geneigsolve(l::AbstractEigenSolver, A, B, nev::Int; kwargs...) = missingbackend(l, "generalized eigenvalue problems")
