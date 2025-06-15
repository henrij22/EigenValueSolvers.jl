using EigenValueSolvers
using Test
using DelimitedFiles
using SparseArrays

@testset "EigenValueSolvers.jl" begin
    # Read a matrix from file "matrix.txt" with space delimiter.
    mDense = readdlm("matrix.txt", '\t')
    mSparse = sparse(mDense)

    arnoldiMethodSolver = EigArnoldiMethod(; sigma = -1.0, which = :SR)
    standardEigenMethod = DefaultEig()

    arnoldiEigenvalues, _, _ = arnoldiMethodSolver(mSparse, 20)
    standardEigenValues, _, _ = standardEigenMethod(mSparse, 20)

    arnoldiEigenvalues = sort(real.(arnoldiEigenvalues))

    @test isapprox(arnoldiEigenvalues[1:6], zeros(6); atol = 1e-8)
    @test isapprox(standardEigenValues[1:6], zeros(6); atol = 1e-8)
end
