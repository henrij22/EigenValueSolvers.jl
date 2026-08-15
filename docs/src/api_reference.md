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

These have to be qualified with `EigenValueSolvers.`. Apart from [`TARGETS`](@ref), they are
not part of the public interface, but they are what a new solver has to hook into.

```@docs
TARGETS
EigenValueSolvers.eigsolve
EigenValueSolvers.geneigsolve
EigenValueSolvers.checktarget
EigenValueSolvers.sortselect
EigenValueSolvers.ordering
EigenValueSolvers.backend
```

## Index

```@index
```
