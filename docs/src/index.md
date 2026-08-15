```@meta
CurrentModule = EigenValueSolvers
```

# EigenValueSolvers.jl

A thin, uniform interface around the eigenvalue solvers of `LinearAlgebra`,
[Arpack.jl](https://github.com/JuliaLinearAlgebra/Arpack.jl),
[ArnoldiMethod.jl](https://github.com/JuliaLinearAlgebra/ArnoldiMethod.jl) and
[KrylovKit.jl](https://github.com/Jutho/KrylovKit.jl).

All solvers share the same call signature, the same `which` targets and the same return
values, so that a solver can be swapped for another one without touching the calling code.

This package is heavily inspired by
[`BifurcationKit.jl`](https://github.com/bifurcationkit/BifurcationKit.jl/blob/master/src/EigSolver.jl),
which is released under the MIT "Expat" License.

## Installation

```julia
using Pkg
Pkg.add("EigenValueSolvers")
```

The backends are *weak* dependencies, loaded through package extensions. Only
[`DefaultEig`](@ref) works out of the box; for the other solvers the corresponding package
has to be installed and loaded:

| solver                    | requires                    |
|:--------------------------|:----------------------------|
| [`DefaultEig`](@ref)      | –                           |
| [`EigArpack`](@ref)       | `using Arpack`              |
| [`EigArnoldiMethod`](@ref)| `using ArnoldiMethod`       |
| [`EigKrylovKit`](@ref)    | `using KrylovKit`           |

Using a solver whose backend is not loaded raises an error that says which package is
missing.

## Usage

A solver is a callable object. Calling it with a matrix `J` and a number of requested
eigenpairs `nev` solves the standard eigenvalue problem `J⋅x = λ⋅x`:

```jldoctest quickstart
julia> using EigenValueSolvers

julia> J = [4.0 1.0 0.0; 1.0 3.0 1.0; 0.0 1.0 2.0];

julia> solver = DefaultEig(:LR);

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

Use [`geteigenvector`](@ref) to access an eigenvector, since matrix-free solvers return a
vector of vectors instead of a matrix:

```jldoctest quickstart
julia> v = geteigenvector(solver, vectors, 1);

julia> J * v ≈ values[1] * v
true
```

## Targets

The `which` field of every solver selects *which* part of the spectrum is computed and in
which order it is returned:

| target | selects the eigenvalues with ... | ordering of the result |
|:-------|:---------------------------------|:-----------------------|
| `:LM`  | largest magnitude                | descending in `abs`    |
| `:SM`  | smallest magnitude               | ascending in `abs`     |
| `:LR`  | largest real part                | descending in `real`   |
| `:SR`  | smallest real part               | ascending in `real`    |
| `:LI`  | largest imaginary part           | descending in `imag`   |
| `:SI`  | smallest imaginary part          | ascending in `imag`    |

Every solver applies this ordering itself, so the results of two different solvers with the
same `which` are directly comparable.

## Generalized eigenvalue problems

`A⋅x = λ⋅B⋅x` is solved with [`gev`](@ref), which returns the same four elements:

```jldoctest
julia> using EigenValueSolvers

julia> A = [4.0 1.0 0.0; 1.0 3.0 1.0; 0.0 1.0 2.0];

julia> B = [2.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 2.0];

julia> values, vectors, converged, numops = gev(DefaultEig(:SR), A, B, 1);

julia> round.(values; digits = 4)
1-element Vector{Float64}:
 0.634
```

## Shift-invert

[`EigArpack`](@ref) and [`EigArnoldiMethod`](@ref) accept a shift `sigma`. With a shift, the
`nev` eigenvalues *closest to* `sigma` are computed and `which` only determines the order in
which they are returned. This is the way to get at the smallest eigenvalues of a large
sparse matrix:

```julia
using EigenValueSolvers, Arpack

solver = EigArpack(-1.0, :SR)
values, vectors, converged, numops = solver(K, 20)
```
