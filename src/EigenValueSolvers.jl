"""
    EigenValueSolvers

A thin, uniform interface around the eigenvalue solvers of `LinearAlgebra`, `Arpack.jl`,
`ArnoldiMethod.jl` and `KrylovKit.jl`.

This package is heavily inspired by
[`BifurcationKit.jl`](https://github.com/bifurcationkit/BifurcationKit.jl/blob/master/src/EigSolver.jl),
which is released under the MIT "Expat" License.
"""
module EigenValueSolvers

using LinearAlgebra: LinearAlgebra, eigen
using ArgCheck: @argcheck

export AbstractEigenSolver, AbstractDirectEigenSolver, AbstractIterativeEigenSolver, AbstractMFEigenSolver
export DefaultEig, EigArpack, EigKrylovKit, EigArnoldiMethod
export gev, geteigenvector, getsolver

include("interface.jl")
include("solvers.jl")

end
