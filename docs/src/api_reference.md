```@meta
CurrentModule = EigenValueSolvers
```

# API Reference

## Public

```@docs
EigenValueSolvers
gev
geteigenvector
getsolver
TARGETS
```

## Internal

These are not exported and not part of the public interface, but they are what a new solver
has to hook into.

```@docs
EigenValueSolvers.eigsolve
EigenValueSolvers.geneigsolve
EigenValueSolvers.sortselect
EigenValueSolvers.ordering
EigenValueSolvers.backend
```

## Index

```@index
```
