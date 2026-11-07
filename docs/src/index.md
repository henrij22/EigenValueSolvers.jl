```@meta
CurrentModule = EigenValueSolvers
```

# EigenValueSolvers.jl

A thin, uniform interface around the eigenvalue solvers of `LinearAlgebra`,
[Arpack.jl](https://github.com/JuliaLinearAlgebra/Arpack.jl),
[ArnoldiMethod.jl](https://github.com/JuliaLinearAlgebra/ArnoldiMethod.jl),
[KrylovKit.jl](https://github.com/Jutho/KrylovKit.jl),
[IterativeSolvers.jl](https://github.com/JuliaLinearAlgebra/IterativeSolvers.jl) and
[GenericArpack.jl](https://github.com/dgleich/GenericArpack.jl).

All solvers have the same call signature, accept the same `which` targets and return the
same values. You can switch from one solver to another without changing the calling code.

This package is inspired by the eigenvalue solvers of
[`BifurcationKit.jl`](https://github.com/bifurcationkit/BifurcationKit.jl/blob/master/src/EigSolver.jl)
(released under the MIT "Expat" License).

## Installation

```julia
using Pkg
Pkg.add("EigenValueSolvers")
```

The backends are weak dependencies, loaded through package extensions. Only
[`EigDefault`](@ref) works out of the box. For the other solvers, install and load the
corresponding package:

| solver                      | requires                 | problems                        |
|:----------------------------|:-------------------------|:--------------------------------|
| [`EigDefault`](@ref)        | –                        | small, dense                    |
| [`EigArpack`](@ref)         | `using Arpack`           | large sparse, shift-invert      |
| [`EigArnoldiMethod`](@ref)  | `using ArnoldiMethod`    | large sparse, shift-invert      |
| [`EigKrylovKit`](@ref)      | `using KrylovKit`        | matrix-free                     |
| [`EigLOBPCG`](@ref)         | `using IterativeSolvers` | large sparse symmetric, no factorization |
| [`EigGenericArpack`](@ref)  | `using GenericArpack`    | symmetric, non-`Float64` element types |

If the backend of a solver is not loaded, calling it throws an error that names the missing
package.

[GenericSchur.jl](https://github.com/RalphAS/GenericSchur.jl) needs no extension, but it is
useful together with this package: it adds `LinearAlgebra.eigen` methods for element types
that LAPACK does not handle. With it loaded, [`EigDefault`](@ref) also works on `BigFloat`
or `Double64` matrices.

## Usage

A solver is a callable object. Calling it with a matrix `J` and a number of requested
eigenpairs `nev` solves the standard eigenvalue problem `J⋅x = λ⋅x`:

```jldoctest quickstart
julia> using EigenValueSolvers

julia> J = [4.0 1.0 0.0; 1.0 3.0 1.0; 0.0 1.0 2.0];

julia> solver = EigDefault(:LR);

julia> values, vectors, converged, numops = solver(J, 2);

julia> round.(values; digits = 4)
2-element Vector{Float64}:
 4.7321
 3.0

julia> converged
true
```

The four return values are always the same:

| element     | meaning                                                              |
|:------------|:---------------------------------------------------------------------|
| `values`    | at most `nev` eigenvalues, ordered according to the solver's `which`  |
| `vectors`   | the matching eigenvectors                                            |
| `converged` | `true` if at least `nev` eigenpairs converged                        |
| `numops`    | number of operator applications used by the backend                  |

Matrix-free solvers return the eigenvectors as a vector of vectors instead of a matrix. Use
[`geteigenvector`](@ref) to access an eigenvector, then the code works for both:

```jldoctest quickstart
julia> v = geteigenvector(solver, vectors, 1);

julia> J * v ≈ values[1] * v
true
```

## Targets

The `which` field of every solver selects the part of the spectrum that is computed and the
order in which it is returned:

| target | selects the eigenvalues with ... | ordering of the result |
|:-------|:---------------------------------|:-----------------------|
| `:LM`  | largest magnitude                | descending in `abs`    |
| `:SM`  | smallest magnitude               | ascending in `abs`     |
| `:LR`  | largest real part                | descending in `real`   |
| `:SR`  | smallest real part               | ascending in `real`    |
| `:LI`  | largest imaginary part           | descending in `imag`   |
| `:SI`  | smallest imaginary part          | ascending in `imag`    |

Every solver sorts its result this way, so two solvers with the same `which` return
comparable results.

Not every backend supports every target. For a solver that only handles symmetric problems,
`:LI`/`:SI` make no sense, and a solver that selects by algebraic value cannot do
`:LM`/`:SM`. Constructing a solver with an unsupported target throws an error.
[`supportedtargets`](@ref) lists the targets a solver accepts:

```jldoctest
julia> using EigenValueSolvers

julia> supportedtargets(EigLOBPCG)
(:LR, :SR)
```

## Generalized eigenvalue problems

`A⋅x = λ⋅B⋅x` is solved with [`gev`](@ref), which returns the same four elements:

```jldoctest
julia> using EigenValueSolvers

julia> A = [4.0 1.0 0.0; 1.0 3.0 1.0; 0.0 1.0 2.0];

julia> B = [2.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 2.0];

julia> values, vectors, converged, numops = gev(EigDefault(:SR), A, B, 1);

julia> round.(values; digits = 4)
1-element Vector{Float64}:
 0.634
```

## Shift-invert

[`EigArpack`](@ref) and [`EigArnoldiMethod`](@ref) accept a shift `sigma`. With a shift, they
compute the `nev` eigenvalues closest to `sigma`, and `which` only sets the order of the
result. Use this to compute the smallest eigenvalues of a large sparse matrix:

```julia
using EigenValueSolvers, Arpack

solver = EigArpack(-1.0, :SR)
values, vectors, converged, numops = solver(K, 20)
```
