```@meta
CurrentModule = EigenValueSolvers
```

# Solvers

## `EigDefault`

```@docs
EigDefault
```

!!! note "Renamed in v0.2.0"
    This solver was called `DefaultEig` up to v0.1.2, and there is no alias for the old
    name -- grep for it when upgrading.

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

`ArnoldiMethod.jl` has no `:SM` target; use a shift of zero instead, which computes the
eigenvalues closest to the origin.

### Plugging in another sparse direct solver

With a shift, the cost of `EigArnoldiMethod` is dominated by factorizing `sigma⋅I - J` (or
`A - sigma⋅B` for [`gev`](@ref)). That factorization is taken from the `factorize` field,
which defaults to `LinearAlgebra.lu` — SuiteSparse UMFPACK for sparse matrices. It is called
as `factorize(M)`, and the only thing required of the result `F` is that
`LinearAlgebra.ldiv!(y, F, x)` works, so swapping in a different solver is a one-liner:

```julia
using EigenValueSolvers, ArnoldiMethod, LinearAlgebra

# a Cholesky factorization, if the shifted matrix is known to be positive definite
EigArnoldiMethod(; sigma = 1.0e5, which = :SR, factorize = cholesky)
```

Which package is worth reaching for depends on where UMFPACK gives out:

| package | when it helps |
|:--|:--|
| [Pardiso.jl](https://github.com/JuliaSparse/Pardiso.jl) | shared-memory parallel direct solves; usually the largest single win on a multicore workstation, and the most robust choice for the indefinite matrices that shift-invert produces |
| [MUMPS.jl](https://github.com/JuliaSmoothOptimizers/MUMPS.jl) | distributed-memory (MPI) factorization, for problems that no longer fit on one machine |
| [HSL.jl](https://github.com/JuliaSmoothOptimizers/HSL.jl) | `MA57` for symmetric indefinite systems, which is exactly the structure of `sigma⋅I - J` for a shift inside the spectrum |
| [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl) | a uniform front end if you want to switch between the above by changing one algorithm argument |
| [`cholesky`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.cholesky) / [`qr`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.qr) | already in `LinearAlgebra`; roughly twice as fast as `lu` when the shifted matrix is positive definite, and `qr` copes with a shift landing on an eigenvalue |

These solvers do not all return something that supports `ldiv!` directly, so a small closure
is sometimes needed — `factorize = M -> MyWrapper(M)` where `MyWrapper` implements
`ldiv!(y, ::MyWrapper, x)`. Note that the factorization happens once per solver call; wrap a
cached object in a closure if you want to reuse it across calls.

[ArnoldiMethodTransformations.jl](https://github.com/wrs28/ArnoldiMethodTransformations.jl)
packages this same idea for `ArnoldiMethod.partialschur` directly, if you would rather not
go through this package at all.

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

## `EigLOBPCG`

```@docs
EigLOBPCG
```

Requires `using IterativeSolvers`. Unlike the Arnoldi-based solvers, LOBPCG finds the
*smallest* eigenvalues without factorizing anything, which is what makes it viable when
`lu(K - sigma*M)` no longer fits in memory. Its convergence depends strongly on the
preconditioner `P`, which is applied as `ldiv!(P, x)`:

```julia
using EigenValueSolvers, IterativeSolvers, LinearAlgebra

# an incomplete Cholesky or algebraic multigrid preconditioner belongs here
solver = EigLOBPCG(; which = :SR, P = Diagonal(diag(K)), tol = 1.0e-10, maxiter = 500)
values, vectors, converged, iterations = gev(solver, K, M, 10)
```

Restrictions to keep in mind: the problem has to be symmetric/hermitian with a positive
definite `B`, only the targets `:LR` and `:SR` are available, and the matrix must be at
least three times as large as `nev`.

## `EigGenericArpack`

```@docs
EigGenericArpack
```

Requires `using GenericArpack`. This is a pure Julia translation of ARPACK, which buys two
things over [`EigArpack`](@ref): no compiled dependency, and support for element types
beyond the four BLAS ones. Together with `GenericSchur.jl` for the dense case, it makes
extended precision eigenvalue computations possible:

```julia
using EigenValueSolvers, GenericArpack

K = big.(stiffness_matrix)
values, vectors, converged, numops = EigGenericArpack(; which = :SR)(K, 5)
```

It covers symmetric/hermitian problems only, so `:LI`/`:SI` are unavailable, and it has no
shift — use [`EigArpack`](@ref) or [`EigArnoldiMethod`](@ref) when you need shift-invert.

## Writing a new solver

A new solver is a subtype of one of the abstract types below that implements
[`EigenValueSolvers.eigsolve`](@ref) and, optionally,
[`EigenValueSolvers.geneigsolve`](@ref). Both should call
[`EigenValueSolvers.sortselect`](@ref) on the raw output of the backend, which is what keeps
the ordering and the number of returned eigenpairs consistent with the other solvers.

If the backend cannot handle every entry of [`TARGETS`](@ref), narrow
[`supportedtargets`](@ref) for the new type so that an unsupported target is rejected when
the solver is constructed rather than silently answering with the wrong part of the
spectrum.

```@docs
AbstractEigenSolver
AbstractDirectEigenSolver
AbstractIterativeEigenSolver
AbstractMFEigenSolver
```
