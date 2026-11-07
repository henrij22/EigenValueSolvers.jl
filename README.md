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

All solvers have the same call signature, accept the same `which` targets and return the
same values. You can switch from one solver to another without changing the calling code.

```julia
using EigenValueSolvers, Arpack

solver = EigArpack(-1.0, :SR)               # the 20 eigenvalues closest to -1.0 ...
values, vectors, converged, numops = solver(K, 20)   # ... in ascending order

v = geteigenvector(solver, vectors, 1)
```

The backends are weak dependencies loaded through package extensions, so you only install
the packages for the solvers you use:

| solver             | requires                 | problems                                 |
|:-------------------|:-------------------------|:-----------------------------------------|
| `EigDefault`       | –                        | small, dense                             |
| `EigArpack`        | `using Arpack`           | large sparse, shift-invert               |
| `EigArnoldiMethod` | `using ArnoldiMethod`    | large sparse, shift-invert               |
| `EigKrylovKit`     | `using KrylovKit`        | matrix-free                              |
| `EigLOBPCG`        | `using IterativeSolvers` | large sparse symmetric, no factorization |
| `EigGenericArpack` | `using GenericArpack`    | symmetric, non-`Float64` element types   |

With [GenericSchur.jl](https://github.com/RalphAS/GenericSchur.jl) loaded, `EigDefault` also
works on `BigFloat` matrices.

See the [documentation](https://henrij22.github.io/EigenValueSolvers.jl/stable) for details.

## Installation (not yet registered)

```julia
using Pkg
Pkg.add("EigenValueSolvers")
```

## Development

Development started well before the advent of agent coding, where I was inspired by the EigenValue solver interface of
[BifurcationKit.jl](https://github.com/bifurcationkit/BifurcationKit.jl/blob/master/src/EigSolver.jl) 
(released under the MIT "Expat" License).
Now with agents, I could tidy up the package, add more solvers and fix long-standing bugs and inconsistencies.

