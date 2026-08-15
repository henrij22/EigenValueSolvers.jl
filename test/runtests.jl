using EigenValueSolvers
using Test

using ArnoldiMethod
using Arpack
using DelimitedFiles
using KrylovKit
using LinearAlgebra
using SparseArrays

# A stiffness matrix with six zero eigenvalues (the rigid body modes) and an otherwise
# positive spectrum.
const K = sparse(readdlm(joinpath(@__DIR__, "matrix.txt"), '\t'))
const REFERENCE = sort(eigvals(Symmetric(Matrix(K))))

# A solver without any backend, used to check the error path of the package extensions.
struct UnbackedSolver <: AbstractEigenSolver end

# `K` has eigenvalues of multiplicity two, and a Krylov method may well return only one copy
# of such a pair. Comparing element by element against `REFERENCE` would therefore be flaky,
# so we only check that every computed eigenvalue is in the spectrum.
inspectrum(λ) = all(λᵢ -> minimum(abs, REFERENCE .- λᵢ) ≤ 1.0e-6 * max(1, abs(λᵢ)), real.(λ))

@testset "EigenValueSolvers.jl" begin
    @testset "targets" begin
        for which in EigenValueSolvers.TARGETS
            @test DefaultEig(which).which === which
            @test EigArpack(nothing, which).which === which
            @test EigArnoldiMethod(; which = which).which === which
            @test EigKrylovKit(; which = which).which === which
        end
        @test_throws ArgumentError DefaultEig(:XX)
        @test_throws ArgumentError EigArpack(nothing, :XX)
        @test_throws ArgumentError EigArnoldiMethod(; which = :XX)
        @test_throws ArgumentError EigKrylovKit(; which = :XX)
    end

    @testset "sorting and selection" begin
        λ = [3.0, -1.0, 2.0]
        ϕ = Float64[1 2 3; 4 5 6; 7 8 9]

        @test EigenValueSolvers.sortselect(λ, ϕ, 2, :LR) == ([3.0, 2.0], ϕ[:, [1, 3]])
        @test EigenValueSolvers.sortselect(λ, ϕ, 2, :SR) == ([-1.0, 2.0], ϕ[:, [2, 3]])
        @test EigenValueSolvers.sortselect(λ, ϕ, 2, :LM) == ([3.0, 2.0], ϕ[:, [1, 3]])
        @test EigenValueSolvers.sortselect(λ, ϕ, 2, :SM) == ([-1.0, 2.0], ϕ[:, [2, 3]])
        # more eigenpairs requested than available
        @test first(EigenValueSolvers.sortselect(λ, ϕ, 10, :SR)) == [-1.0, 2.0, 3.0]
        # eigenvectors stored as a vector of vectors
        @test EigenValueSolvers.sortselect(λ, collect(eachcol(ϕ)), 2, :SR)[2] == [ϕ[:, 2], ϕ[:, 3]]
    end

    @testset "small dense problems" begin
        A = Symmetric(Float64[4 1 0; 1 3 1; 0 1 2])
        exact = sort(eigvals(A))
        nev = 2

        solvers = (
            DefaultEig(:SR),
            EigArpack(0.0, :SR),
            EigArnoldiMethod(; sigma = 0.0, which = :SR),
            EigKrylovKit(; which = :SR, ishermitian = true, tol = 1.0e-12),
        )

        for l in solvers
            λ, ϕ, converged, numops = l(Matrix(A), nev)
            @test length(λ) == nev
            @test converged
            @test numops ≥ 1
            @test real.(λ) ≈ exact[1:nev]
            for i in 1:nev
                v = geteigenvector(l, ϕ, i)
                @test A * v ≈ λ[i] * v
            end
            @test getsolver(l) === l
        end
    end

    @testset "generalized problem" begin
        A = Symmetric(Float64[4 1 0; 1 3 1; 0 1 2])
        B = Symmetric(Float64[2 0 0; 0 2 0; 0 0 2])
        exact = sort(eigvals(Matrix(A), Matrix(B)))
        nev = 2

        solvers = (
            DefaultEig(:SR),
            EigArnoldiMethod(; sigma = 0.0, which = :SR),
            EigKrylovKit(; which = :SR, ishermitian = true, isposdef = true, tol = 1.0e-12),
        )

        for l in solvers
            λ, ϕ, converged, numops = gev(l, Matrix(A), Matrix(B), nev)
            @test length(λ) == nev
            @test converged
            @test numops ≥ 1
            @test real.(λ) ≈ exact[1:nev]
            for i in 1:nev
                v = geteigenvector(l, ϕ, i)
                @test A * v ≈ λ[i] * (B * v)
            end
        end
    end

    @testset "stiffness matrix, smallest eigenvalues" begin
        nev = 20

        solvers = (
            EigArpack(-1.0, :SR),
            EigArnoldiMethod(; sigma = -1.0, which = :SR),
        )

        for l in solvers
            λ, ϕ, converged, _ = l(K, nev)
            @test converged
            @test length(λ) == nev
            @test issorted(real.(λ))
            @test inspectrum(λ)
            @test isapprox(real.(λ[1:6]), zeros(6); atol = 1.0e-8)
            # the shift-invert method has to find the very smallest part of the spectrum
            @test maximum(real, λ) ≤ REFERENCE[2 * nev]
        end
    end

    @testset "stiffness matrix, largest eigenvalues" begin
        nev = 5

        solvers = (
            EigArpack(nothing, :LM),
            EigArnoldiMethod(; which = :LM),
            EigKrylovKit(; which = :LM, ishermitian = true, dim = 60, maxiter = 300, tol = 1.0e-10),
        )

        for l in solvers
            λ, _, converged, _ = l(K, nev)
            @test converged
            @test length(λ) == nev
            @test issorted(abs.(λ); rev = true)
            @test inspectrum(λ)
            @test real(λ[1]) ≈ REFERENCE[end]
            @test minimum(real, λ) ≥ REFERENCE[end - 2 * nev]
        end
    end

    @testset "matrix-free" begin
        A = Float64[4 1 0; 1 3 1; 0 1 2]
        exact = sort(eigvals(Symmetric(A)); rev = true)

        l = EigKrylovKit(; which = :LR, ishermitian = true, tol = 1.0e-12, x₀ = rand(3))
        λ, ϕ, converged, _ = l(x -> A * x, 2)

        @test converged
        @test real.(λ) ≈ exact[1:2]
        # matrix-free solvers return the eigenvectors as a vector of vectors
        @test ϕ isa AbstractVector{<:AbstractVector}
        v = geteigenvector(l, ϕ, 1)
        @test A * v ≈ λ[1] * v

        # without a starting vector there is nothing for KrylovKit to iterate on
        @test_throws ArgumentError EigKrylovKit(; ishermitian = true)(x -> A * x, 2)
    end

    @testset "matrix-free generalized problem" begin
        A = Float64[4 1 0; 1 3 1; 0 1 2]
        B = Float64[2 0 0; 0 2 0; 0 0 2]
        exact = sort(eigvals(A, B))

        l = EigKrylovKit(;
            which = :SR, ishermitian = true, isposdef = true, tol = 1.0e-12, x₀ = rand(3)
        )
        λ, ϕ, converged, _ = gev(l, x -> A * x, x -> B * x, 1)

        @test converged
        @test real.(λ) ≈ exact[1:1]
        @test ϕ isa AbstractVector{<:AbstractVector}
        v = geteigenvector(l, ϕ, 1)
        @test A * v ≈ λ[1] * (B * v)

        @test_throws ArgumentError gev(
            EigKrylovKit(; ishermitian = true, isposdef = true), x -> A * x, x -> B * x, 1
        )
    end

    @testset "missing backend" begin
        # `DefaultEig` needs no weak dependency, the others report the package they need
        @test EigenValueSolvers.backend(DefaultEig()) === nothing
        @test EigenValueSolvers.backend(EigArpack()) === :Arpack
        @test EigenValueSolvers.backend(EigArnoldiMethod()) === :ArnoldiMethod
        @test EigenValueSolvers.backend(EigKrylovKit()) === :KrylovKit

        @test_throws ArgumentError UnbackedSolver()(rand(2, 2), 1)
        @test_throws ArgumentError gev(UnbackedSolver(), rand(2, 2), rand(2, 2), 1)
    end
end
