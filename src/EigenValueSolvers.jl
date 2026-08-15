"""
    EigenValueSolvers

A thin, uniform interface around the eigenvalue solvers of `LinearAlgebra`, `Arpack.jl`,
`ArnoldiMethod.jl`, `KrylovKit.jl`, `IterativeSolvers.jl` and `GenericArpack.jl`.

This package is heavily inspired by
[`BifurcationKit.jl`](https://github.com/bifurcationkit/BifurcationKit.jl/blob/master/src/EigSolver.jl),
which is released under the MIT "Expat" License.
"""
module EigenValueSolvers

using LinearAlgebra: LinearAlgebra, eigen, lu
using ArgCheck: @argcheck

export AbstractEigenSolver, AbstractDirectEigenSolver, AbstractIterativeEigenSolver, AbstractMFEigenSolver
export EigDefault, EigArpack, EigKrylovKit, EigArnoldiMethod, EigLOBPCG, EigGenericArpack
export gev, geteigenvector, getsolver, supportedtargets

include("interface.jl")
include("solvers.jl")

end
