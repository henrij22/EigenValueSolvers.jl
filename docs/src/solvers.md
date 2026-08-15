```@meta
CurrentModule = EigenValueSolvers
```

# Solvers

## `DefaultEig`

```@docs
DefaultEig
```

`DefaultEig` computes the full spectrum with `LinearAlgebra.eigen` and then selects `nev`
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

`ArnoldiMethod.jl` has no `:SM` target; use a shift of zero instead, which computes the
eigenvalues closest to the origin.

## `EigKrylovKit`

```@docs
EigKrylovKit
```

Requires `using KrylovKit`. This is the only matrix-free solver: `J` may be any object for
which `J * x` is defined, in which case a starting vector `x₀` has to be supplied.

```julia
using EigenValueSolvers, KrylovKit

solver = EigKrylovKit(; which = :LR, ishermitian = true, x₀ = rand(n))
values, vectors, converged, numops = solver(x -> A * x, 4)
```

Note that the eigenvectors are then returned as a vector of vectors, so
[`geteigenvector`](@ref) should be used to access them.

## Writing a new solver

A new solver is a subtype of one of the abstract types below that implements
[`EigenValueSolvers.eigsolve`](@ref) and, optionally,
[`EigenValueSolvers.geneigsolve`](@ref). Both should call
[`EigenValueSolvers.sortselect`](@ref) on the raw output of the backend, which is what keeps
the ordering and the number of returned eigenpairs consistent with the other solvers.

```@docs
AbstractEigenSolver
AbstractDirectEigenSolver
AbstractIterativeEigenSolver
AbstractMFEigenSolver
```
