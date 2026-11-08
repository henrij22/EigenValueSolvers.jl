```@meta
CurrentModule = EigenValueSolvers
```

# API Reference

## Exported

```@docs
EigenValueSolvers
gev
geteigenvector
getsolver
supportedtargets
```

## Not exported

These have to be qualified with `EigenValueSolvers.`. Except for [`TARGETS`](@ref), they are
not part of the public interface, but a new solver builds on them.

```@docs
TARGETS
EigenValueSolvers.eigsolve
EigenValueSolvers.geneigsolve
EigenValueSolvers.checktarget
EigenValueSolvers.sortselect
EigenValueSolvers.ordering
EigenValueSolvers.backend
EigenValueSolvers.matrixrequired
```

## Index

```@index
```
