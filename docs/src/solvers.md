```@meta
CurrentModule = EigenValueSolvers
```

# Solvers

## `EigDefault`

```@docs
EigDefault
```

!!! note "Renamed in v0.2.0"
    This solver was called `DefaultEig` up to v0.1.2. There is no alias for the old name,
    so rename all uses when upgrading.

`EigDefault` computes the full spectrum with `LinearAlgebra.eigen` and then selects `nev`
eigenpairs from it. Sparse matrices are converted to dense ones first, so this solver is
meant for small problems and as a reference for the iterative solvers. It is the only solver
that needs no additional package.

## `EigArpack`

```@docs
EigArpack
```

Requires `using Arpack`. Extra keyword arguments given to the constructor are forwarded to
`Arpack.eigs`, e.g.

```julia
EigArpack(-1.0, :SR; tol = 1.0e-10, maxiter = 500)
```

## `EigArnoldiMethod`

```@docs
EigArnoldiMethod
```

Requires `using ArnoldiMethod`. Extra keyword arguments given to the constructor are
forwarded to `ArnoldiMethod.partialschur`, e.g.

```julia
EigArnoldiMethod(; sigma = -1.0, which = :SR, mindim = 40, maxdim = 80)
```

`ArnoldiMethod.jl` has no `:SM` target. Use a shift of zero instead, which computes the
eigenvalues closest to the origin.

### Plugging in another sparse direct solver

With a shift, most of the time in `EigArnoldiMethod` is spent factorizing `sigma⋅I - J` (or
`A - sigma⋅B` for [`gev`](@ref)). The factorization is done by the function in the
`factorize` field, which defaults to `LinearAlgebra.lu` (UMFPACK for sparse matrices). It is
called as `factorize(M)`, and its result `F` only has to support
`LinearAlgebra.ldiv!(y, F, x)`. To use a different solver, pass another function:

```julia
using EigenValueSolvers, ArnoldiMethod, LinearAlgebra

# a Cholesky factorization, if the shifted matrix is known to be positive definite
EigArnoldiMethod(; sigma = 1.0e5, which = :SR, factorize = cholesky)
```

Some alternatives to UMFPACK:

| package | use case |
|:--|:--|
| [`cholesky`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.cholesky) | part of `LinearAlgebra`, for a positive definite shifted matrix, e.g. a shift below the spectrum of a stiffness matrix |
| [Pardiso.jl](https://github.com/JuliaSparse/Pardiso.jl) | shared-memory parallel direct solver |
| [MUMPS.jl](https://github.com/JuliaSmoothOptimizers/MUMPS.jl) | distributed-memory (MPI) direct solver for problems that do not fit on one machine |
| [HSL.jl](https://github.com/JuliaSmoothOptimizers/HSL.jl) | `MA57` for symmetric indefinite matrices, such as `sigma⋅I - J` with a shift inside the spectrum |
| [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl) | common interface to many of the above |

Not all of these return an object that supports `ldiv!`. In that case write a small wrapper
type that implements `ldiv!(y, ::MyWrapper, x)` and pass `factorize = M -> MyWrapper(M)`.
The matrix is factorized once per solver call. To reuse a factorization across calls, return
a cached object from the closure.

[ArnoldiMethodTransformations.jl](https://github.com/wrs28/ArnoldiMethodTransformations.jl)
implements the same transformations for `ArnoldiMethod.partialschur` directly, without this
package.

## `EigKrylovKit`

```@docs
EigKrylovKit
```

Requires `using KrylovKit`. This is the only matrix-free solver: `J` can be any object that
supports `J * x`. In that case a starting vector `x₀` is required.

```julia
using EigenValueSolvers, KrylovKit

solver = EigKrylovKit(; which = :LR, ishermitian = true, x₀ = rand(n))
values, vectors, converged, numops = solver(x -> A * x, 4)
```

The eigenvectors are then returned as a vector of vectors, so use [`geteigenvector`](@ref) to
access them.

## `EigLOBPCG`

```@docs
EigLOBPCG
```

Requires `using IterativeSolvers`. Unlike the Arnoldi-based solvers, LOBPCG computes the
smallest eigenvalues without a factorization. This makes it an option when
`lu(K - sigma*M)` does not fit in memory. Its convergence depends strongly on the
preconditioner `P`, which is applied as `ldiv!(P, x)`:

```julia
using EigenValueSolvers, IterativeSolvers, LinearAlgebra

# an incomplete Cholesky or algebraic multigrid preconditioner belongs here
solver = EigLOBPCG(; which = :SR, P = Diagonal(diag(K)), tol = 1.0e-10, maxiter = 500)
values, vectors, converged, iterations = gev(solver, K, M, 10)
```

Restrictions: the problem has to be symmetric or hermitian with a positive definite `B`, only
the targets `:LR` and `:SR` are available, and the matrix must be at least three times as
large as `nev`.

## `EigGenericArpack`

```@docs
EigGenericArpack
```

Requires `using GenericArpack`. GenericArpack.jl is a pure Julia translation of ARPACK.
Compared to [`EigArpack`](@ref), it needs no compiled library and supports element types
other than the four BLAS types. Together with `GenericSchur.jl` for dense matrices, this
allows eigenvalue computations in extended precision:

```julia
using EigenValueSolvers, GenericArpack

K = big.(stiffness_matrix)
values, vectors, converged, numops = EigGenericArpack(; which = :SR)(K, 5)
```

It only handles symmetric or hermitian problems, so `:LI`/`:SI` are not available. It also
has no shift. Use [`EigArpack`](@ref) or [`EigArnoldiMethod`](@ref) for shift-invert.

## Writing a new solver

A new solver is a subtype of one of the abstract types below that implements
[`EigenValueSolvers.eigsolve`](@ref) and, optionally,
[`EigenValueSolvers.geneigsolve`](@ref). Both should pass the raw output of the backend
through [`EigenValueSolvers.sortselect`](@ref), so that the order and the number of returned
eigenpairs match the other solvers.

If the backend does not support every entry of [`TARGETS`](@ref), define
[`supportedtargets`](@ref) for the new type. An unsupported target is then rejected when the
solver is constructed, instead of returning the wrong part of the spectrum.

```@docs
AbstractEigenSolver
AbstractDirectEigenSolver
AbstractIterativeEigenSolver
AbstractMFEigenSolver
```
