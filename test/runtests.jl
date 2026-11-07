using EigenValueSolvers
using Test

using ArnoldiMethod
using Arpack
using DelimitedFiles
using GenericArpack
using GenericSchur
using IterativeSolvers
using KrylovKit
using LinearAlgebra
using Random
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

# A dense symmetric matrix with exactly the prescribed spectrum, built from a Householder
# reflector. Doing this instead of transforming with a random orthogonal matrix keeps the
# tests reproducible: the random stream is stable neither across Julia versions nor across
# the different BLAS implementations of the CI runners, and LOBPCG in particular is
# sensitive enough to that to fail on one platform and pass on another.
function withspectrum(spectrum, seedvector)
    w = normalize(collect(float.(seedvector)))
    Q = I - 2 * w * transpose(w)
    return Matrix(Symmetric(Q * Diagonal(collect(float.(spectrum))) * transpose(Q)))
end

@testset "EigenValueSolvers.jl" begin
    @testset "targets" begin
        for which in EigenValueSolvers.TARGETS
            @test EigDefault(which).which === which
            @test EigArpack(nothing, which).which === which
            @test EigArnoldiMethod(; which = which).which === which
            @test EigKrylovKit(; which = which).which === which
        end
        @test_throws ArgumentError EigDefault(:XX)
        @test_throws ArgumentError EigArpack(nothing, :XX)
        @test_throws ArgumentError EigArnoldiMethod(; which = :XX)
        @test_throws ArgumentError EigKrylovKit(; which = :XX)
        @test_throws ArgumentError EigLOBPCG(; which = :XX)
        @test_throws ArgumentError EigGenericArpack(; which = :XX)

        # keywords the backend call sets itself
        @test_throws ArgumentError EigArpack(; nev = 3)
        @test_throws ArgumentError EigArpack(nothing, :LR; ritzvec = false)
        @test_throws ArgumentError EigArpack(; explicittransform = :none)
        @test_throws ArgumentError EigArpack(1.0; sigma = 2.0)
        @test_throws ArgumentError EigArpack(nothing, :SR; which = :LR)
        @test EigArpack(1.0; sigma = 1.0).sigma == 1.0
        @test_throws ArgumentError EigArnoldiMethod(; v1 = ones(3))
        @test_throws ArgumentError EigArnoldiMethod(; nev = 3)
        @test_throws ArgumentError EigGenericArpack(; stats = nothing)
        @test EigArpack(; maxiter = 10).kwargs[:maxiter] == 10
    end

    @testset "supported targets" begin
        @test supportedtargets(EigDefault) == EigenValueSolvers.TARGETS
        @test supportedtargets(EigDefault()) == EigenValueSolvers.TARGETS
        @test supportedtargets(EigLOBPCG) == (:LR, :SR)
        @test supportedtargets(EigGenericArpack) == (:LM, :SM, :LR, :SR)

        # a solver has to reject the targets it cannot honour instead of silently
        # returning the wrong part of the spectrum
        for which in (:LM, :SM, :LI, :SI)
            @test_throws ArgumentError EigLOBPCG(; which = which)
        end
        for which in (:LI, :SI)
            @test_throws ArgumentError EigGenericArpack(; which = which)
        end
        for which in supportedtargets(EigLOBPCG)
            @test EigLOBPCG(; which = which).which === which
        end
        for which in supportedtargets(EigGenericArpack)
            @test EigGenericArpack(; which = which).which === which
        end
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
            EigDefault(:SR),
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
            EigDefault(:SR),
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
            EigGenericArpack(; which = :LM),
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

        l = EigKrylovKit(; which = :LR, ishermitian = true, tol = 1.0e-12, x₀ = [0.3, 0.7, 0.5])
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
            which = :SR, ishermitian = true, isposdef = true, tol = 1.0e-12, x₀ = [0.3, 0.7, 0.5]
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

    @testset "symmetric solvers" begin
        # LOBPCG needs the matrix to be at least three times the block size.
        n = 60
        nev = 3
        spectrum = collect(1.0:n)
        A = withspectrum(spectrum, sqrt.(1.0:n))
        B = withspectrum(range(1.0, 3.0; length = n), n:-1.0:1)
        exactgen = sort(eigvals(A, B))

        # A deterministic, well conditioned starting block. Without one, LOBPCG draws a
        # `rand` (not `randn`) block, whose columns all cluster around the all-ones vector.
        X₀ = [cospi(i * j / n) for i in 1:n, j in 1:nev]

        # without `X₀`, LOBPCG starts from `rand` and fails in about 1 of 300 runs
        lobpcgseed = 1

        # `tol` stays clear of 1e-9, where LOBPCG's Rayleigh-Ritz subproblem starts to fail
        # with a `PosDefException` once the search space becomes numerically rank deficient.
        lobpcgsolver(; kwargs...) = EigLOBPCG(; tol = 1.0e-8, maxiter = 500, kwargs...)

        @testset "standard, $(nameof(typeof(l))) $(l.which)" for l in (
                lobpcgsolver(; which = :SR),
                lobpcgsolver(; which = :LR),
                lobpcgsolver(; which = :SR, X₀ = X₀),
                lobpcgsolver(; which = :LR, X₀ = X₀),
                EigGenericArpack(; which = :SR),
                EigGenericArpack(; which = :LR),
                EigGenericArpack(; which = :LM),
            )
            Random.seed!(lobpcgseed)
            λ, ϕ, converged, numops = l(A, nev)
            by, rev = EigenValueSolvers.ordering(l.which)
            exact = sort(spectrum; by = by, rev = rev)[1:nev]

            # the return types are part of the contract, and LOBPCG reports these
            # differently depending on whether it was given a starting block
            @test converged isa Bool
            @test numops isa Integer

            @test converged
            @test numops ≥ 1
            @test length(λ) == nev
            @test isapprox(λ, exact; rtol = 1.0e-6)
            for i in 1:nev
                v = geteigenvector(l, ϕ, i)
                @test isapprox(A * v, λ[i] * v; rtol = 1.0e-6)
            end
        end

        @testset "generalized, $(nameof(typeof(l)))" for l in (
                lobpcgsolver(; which = :SR),
                lobpcgsolver(; which = :SR, X₀ = X₀),
                EigGenericArpack(; which = :SR),
            )
            Random.seed!(lobpcgseed)
            λ, ϕ, converged, numops = gev(l, A, B, nev)

            @test converged isa Bool
            @test numops isa Integer
            @test converged
            @test isapprox(λ, exactgen[1:nev]; rtol = 1.0e-6)
            for i in 1:nev
                v = geteigenvector(l, ϕ, i)
                @test isapprox(A * v, λ[i] * (B * v); rtol = 1.0e-6)
            end
        end

        # LOBPCG is unstable for matrices that are small relative to the block size
        @test_throws ArgumentError EigLOBPCG()(withspectrum(1.0:3.0, 1.0:3.0), 2)
        @test_throws ArgumentError EigLOBPCG(; X₀ = ones(6, 3))(withspectrum(1.0:6.0, 1.0:6.0), 1)
    end

    @testset "Arpack honours every target" begin
        # complex conjugate pairs and real eigenvalues, distinct in real part, imag part and abs
        m = 10
        pairs = [complex(-2.0 + 0.45k, 0.3k + 0.02k^2) for k in 1:m]
        reals = [-3.1 + 0.73k for k in 1:m]
        D = zeros(3m, 3m)
        for (k, z) in enumerate(pairs)
            D[(2k - 1):2k, (2k - 1):2k] = [real(z) imag(z); -imag(z) real(z)]
        end
        for (k, r) in enumerate(reals)
            D[2m + k, 2m + k] = r
        end
        w = normalize(sqrt.(1.0:(3m)))
        Q = I - 2 * w * transpose(w)
        A = Q * D * transpose(Q)
        spectrum = vcat(pairs, conj.(pairs), reals)
        nev = 4

        for which in EigenValueSolvers.TARGETS
            by, rev = EigenValueSolvers.ordering(which)
            expected = by.(sort(spectrum; by = by, rev = rev)[1:nev])
            λ, _, converged, _ = EigArpack(nothing, which; maxiter = 1000)(A, nev)
            @test converged
            @test by.(λ) ≈ expected atol = 1.0e-8

            # ARPACK's "SM" mode is unreliable for generalized problems
            which === :SM && continue
            d = 1.0 .+ (1:(3m)) ./ (3m)
            λ, _, converged, _ = gev(EigArpack(nothing, which; maxiter = 1000), Diagonal(d) * A, Matrix(Diagonal(d)), nev)
            @test converged
            @test by.(λ) ≈ expected atol = 1.0e-8
        end

        S = withspectrum(1.0:20, 1.0:20)
        @test_throws ArgumentError EigArpack(nothing, :LI)(S, 2)
        @test_throws ArgumentError EigArpack(nothing, :SI)(S, 2)

        # `sigma`/`which` as keywords must not end up in the kwargs for `Arpack.eigs`
        l = EigArpack(; sigma = 0.0, which = :SR, maxiter = 500)
        @test l.sigma == 0.0
        @test l.which === :SR
        @test NamedTuple(l.kwargs) == (; maxiter = 500)
        @test EigArpack(which = :SR).which === :SR

        # ARPACK computes at most n - 2 eigenvalues, :LI/:SI of a real problem need twice nev
        @test_throws ArgumentError EigArpack(nothing, :LI)(A, 15)
        @test length(EigArpack(nothing, :LI)(A, 14)[1]) == 14

        # the second `B` is indefinite
        n = 30
        Ks = withspectrum(1.0:n, sqrt.(1.0:n))
        for B in (Matrix(Diagonal(1.0 .+ (1:n) ./ n)), Matrix(Diagonal([isodd(i) ? 1.0 : -2.0 for i in 1:n])))
            exact = eigvals(Ks, B)
            for σ in (0.0, 4.2, -3.7)
                λ, ϕ, converged, _ = gev(EigArpack(σ, :LR), Ks, B, 3)
                @test converged
                @test sort(real.(λ)) ≈ sort(real.(sort(exact; by = x -> abs(x - σ))[1:3]))
                @test issorted(real.(λ); rev = true)
                for i in 1:3
                    @test Ks * ϕ[:, i] ≈ λ[i] * (B * ϕ[:, i])
                end
                isposdef(B) && @test eltype(λ) <: Real
            end
        end
        @test_throws ArgumentError gev(EigArpack(1.0im, :LR), Ks, Matrix(Diagonal(1.0 .+ (1:n) ./ n)), 3)

        # all eigenvalues are real for a symmetric `A` and a positive definite `B`
        @test_throws ArgumentError gev(EigArpack(nothing, :LI), Ks, Matrix(Diagonal(1.0 .+ (1:n) ./ n)), 2)

        # a symmetric `A` with an indefinite `B`, as in buckling, has complex eigenvalues ±i⋅c
        npairs, nreal = 5, 20
        np = 2npairs + nreal
        Ap, Bp = zeros(np, np), zeros(np, np)
        for k in 1:npairs
            i = 2k - 1
            Ap[i:(i + 1), i:(i + 1)] = (k / 2) * [0 1; 1 0]
            Bp[i:(i + 1), i:(i + 1)] = [1 0; 0 -1]
        end
        for j in 1:nreal
            Ap[2npairs + j, 2npairs + j] = 1 + j
            Bp[2npairs + j, 2npairs + j] = isodd(j) ? 1.0 : -2.0
        end
        wp = normalize(sqrt.(1.0:np))
        Qp = I - 2 * wp * transpose(wp)
        Ap, Bp = Matrix(Symmetric(transpose(Qp) * Ap * Qp)), Matrix(Symmetric(transpose(Qp) * Bp * Qp))
        @test !isposdef(Bp)
        for (which, expected) in ((:LI, [2.5, 2.0, 1.5]), (:SI, [-2.5, -2.0, -1.5]))
            λ, _, converged, _ = gev(EigArpack(nothing, which; maxiter = 1000), Ap, Bp, 3)
            @test converged
            @test imag.(λ) ≈ expected atol = 1.0e-8
        end
    end

    @testset "custom factorization" begin
        # `factorize` only has to return something supporting `ldiv!`, so a Cholesky
        # factorization works for the positive definite shifted matrix below.
        n = 40
        spectrum = collect(1.0:n)
        A = withspectrum(spectrum, sqrt.(1.0:n))
        nev = 3

        # `cholesky` needs (sigma⋅I - A) to be positive definite, so the shift has to sit
        # above the whole spectrum -- which means the eigenvalues closest to it are the
        # largest ones, returned in ascending order because `which = :SR`.
        l = EigArnoldiMethod(; sigma = 100.0, which = :SR, factorize = cholesky)
        λ, _, converged, _ = l(A, nev)

        @test converged
        @test λ ≈ spectrum[(end - nev + 1):end]

        # the factorization is genuinely taken from the field, not hard-coded
        called = Ref(0)
        counting = M -> (called[] += 1; lu(M))
        EigArnoldiMethod(; sigma = 100.0, which = :SR, factorize = counting)(A, nev)
        @test called[] == 1
    end

    @testset "GenericSchur for non-BLAS eltypes" begin
        # GenericSchur adds `LinearAlgebra.eigen` methods for element types LAPACK cannot
        # handle, which `EigDefault` then picks up without any extension of its own.
        A = BigFloat[4 1 0; 1 3 1; 0 1 2]
        λ, ϕ, converged, _ = EigDefault(:SR)(A, 2)

        @test converged
        @test eltype(λ) <: Union{BigFloat, Complex{BigFloat}}
        @test real.(λ) ≈ sort(eigvals(Symmetric(Float64.(A))))[1:2] rtol = 1.0e-12
        v = geteigenvector(EigDefault(:SR), ϕ, 1)
        @test A * v ≈ λ[1] * v
    end

    @testset "missing backend" begin
        # `EigDefault` needs no weak dependency, the others report the package they need
        @test EigenValueSolvers.backend(EigDefault()) === nothing
        @test EigenValueSolvers.backend(EigArpack()) === :Arpack
        @test EigenValueSolvers.backend(EigArnoldiMethod()) === :ArnoldiMethod
        @test EigenValueSolvers.backend(EigKrylovKit()) === :KrylovKit
        @test EigenValueSolvers.backend(EigLOBPCG()) === :IterativeSolvers
        @test EigenValueSolvers.backend(EigGenericArpack()) === :GenericArpack

        @test_throws ArgumentError UnbackedSolver()(rand(2, 2), 1)
        @test_throws ArgumentError gev(UnbackedSolver(), rand(2, 2), rand(2, 2), 1)
    end
end
