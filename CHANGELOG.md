# Changelog

## v0.2.0

### Breaking

- `Arpack.jl`, `ArnoldiMethod.jl` and `KrylovKit.jl` moved to *weak* dependencies. The solver
  types are still exported from the package itself, but `EigArpack`, `EigArnoldiMethod` and
  `EigKrylovKit` now require the corresponding package to be loaded. Using one without its
  backend raises an error naming the missing package.
- `LinearMaps.jl` and `Parameters.jl` are no longer dependencies. `ArgCheck.jl` and
  `LinearAlgebra` are the only remaining ones.
- `which` is now a `Symbol` out of `:LM`, `:SM`, `:LR`, `:SR`, `:LI`, `:SI` for *every*
  solver, and it determines both the selected part of the spectrum and the order of the
  result. In particular:
  - `DefaultEig` took a sorting function before and always returned the eigenvalues in
    ascending order; `DefaultEig(:SR)` reproduces the old behaviour.
  - `EigArnoldiMethod` took an `ArnoldiMethod.Target` before; pass the matching symbol
    instead.
  - The separate `by` field of `EigArpack` and `EigArnoldiMethod` is gone.
- All solvers now return *at most* `nev` eigenpairs, and `gev` returns the same four
  elements as a solver call (`values`, `vectors`, `converged`, `numops`) instead of only
  two.
- With a shift `sigma`, `EigArpack` and `EigArnoldiMethod` now compute the eigenvalues
  *closest to* `sigma`, and `which` only orders the result. Previously `which` was passed on
  to the shift-inverted problem, where it selected an unrelated part of the spectrum.
- `converged` is now `true` only if at least `nev` eigenpairs converged. `DefaultEig` and
  `EigArpack` previously returned `true` unconditionally.

### Fixed

- `EigKrylovKit()` threw an `UndefVarError` because `KrylovDefaults` was never imported.
- The `x₀` field of `EigArnoldiMethod` was ignored and is now used as the starting vector.
- `numops` is now the number of operator applications reported by the backend instead of a
  hard-coded `1`.

### Added

- Documentation at <https://henrij22.github.io/EigenValueSolvers.jl>.
- `gev` support for `EigKrylovKit` via `KrylovKit.geneigsolve`, along with the new
  `isposdef` field.
- `geteigenvector`, `getsolver` and `gev` are now exported.
