# EigenValueSolvers.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://henrij22.github.io/EigenValueSolvers.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://henrij22.github.io/EigenValueSolvers.jl/dev)
[![Build Status](https://github.com/henrij22/EigenValueSolvers.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/henrij22/EigenValueSolvers.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/henrij22/EigenValueSolvers.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/henrij22/EigenValueSolvers.jl)
[![code style: runic](https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black)](https://github.com/fredrikekre/Runic.jl)

A thin, uniform interface around the eigenvalue solvers of `LinearAlgebra`,
[Arpack.jl](https://github.com/JuliaLinearAlgebra/Arpack.jl),
[ArnoldiMethod.jl](https://github.com/JuliaLinearAlgebra/ArnoldiMethod.jl),
[KrylovKit.jl](https://github.com/Jutho/KrylovKit.jl),
[IterativeSolvers.jl](https://github.com/JuliaLinearAlgebra/IterativeSolvers.jl) and
[GenericArpack.jl](https://github.com/dgleich/GenericArpack.jl).

All solvers share the same call signature, the same `which` targets and the same return
values, so that a solver can be swapped for another one without touching the calling code.

```julia
using EigenValueSolvers, Arpack

solver = EigArpack(-1.0, :SR)               # the 20 eigenvalues closest to -1.0 ...
values, vectors, converged, numops = solver(K, 20)   # ... in ascending order

v = geteigenvector(solver, vectors, 1)
```

The backends are *weak* dependencies loaded through package extensions, so only the solvers
you actually use are installed and loaded:

| solver             | requires                 | problems                                 |
|:-------------------|:-------------------------|:-----------------------------------------|
| `EigDefault`       | –                        | small, dense                             |
| `EigArpack`        | `using Arpack`           | large sparse, shift-invert               |
| `EigArnoldiMethod` | `using ArnoldiMethod`    | large sparse, shift-invert               |
| `EigKrylovKit`     | `using KrylovKit`        | matrix-free                              |
| `EigLOBPCG`        | `using IterativeSolvers` | large sparse symmetric, no factorization |
| `EigGenericArpack` | `using GenericArpack`    | symmetric, non-`Float64` element types   |

Loading [GenericSchur.jl](https://github.com/RalphAS/GenericSchur.jl) additionally makes
`EigDefault` work on `BigFloat` matrices, without this package needing an extension for it.

See the [documentation](https://henrij22.github.io/EigenValueSolvers.jl/stable) for details.

This package is heavily inspired by
[BifurcationKit.jl](https://github.com/bifurcationkit/BifurcationKit.jl/blob/master/src/EigSolver.jl),
which is released under the MIT "Expat" License.
