module EigenValueSolversArnoldiMethodExt

using ArgCheck: @argcheck
using EigenValueSolvers: EigenValueSolvers, EigArnoldiMethod, sortselect
using LinearAlgebra: LinearAlgebra, ldiv!, mul!

import ArnoldiMethod

"""
    ShiftInvert(F, B, n, tmp)

The linear map `x -> F \\ (B * x)`, with `B === nothing` standing for the identity. This
implements just enough of the array interface (`eltype`, `size` and `mul!`) for
`ArnoldiMethod.partialschur`, which saves a dependency on a dedicated linear map package.
"""
struct ShiftInvert{T, TF, TB}
    F::TF
    B::TB
    n::Int
    tmp::Vector{T}
end

Base.eltype(::ShiftInvert{T}) where {T} = T
Base.size(op::ShiftInvert) = (op.n, op.n)
Base.size(op::ShiftInvert, i::Int) = i ≤ 2 ? op.n : 1

LinearAlgebra.mul!(y, op::ShiftInvert{T, TF, Nothing}, x) where {T, TF} = ldiv!(y, op.F, x)

function LinearAlgebra.mul!(y, op::ShiftInvert, x)
    mul!(op.tmp, op.B, x)
    return ldiv!(y, op.F, op.tmp)
end

# ArnoldiMethod has no `:SM` target, but the shift-invert method with a shift of zero does
# exactly that, which is what a user asking for `:SM` most likely wants.
function checkarnoldiwhich(which::Symbol)
    which === :SM && throw(
        ArgumentError(
            "ArnoldiMethod.jl does not support the target `:SM`, use a shift instead, e.g. " *
                "`EigArnoldiMethod(; sigma = 0, which = :SM)`"
        )
    )
    return which
end

function EigenValueSolvers.eigsolve(l::EigArnoldiMethod, J, nev::Int; kwargs...)
    @argcheck J isa AbstractMatrix
    n = min(nev, size(J, 1))

    if isnothing(l.sigma)
        decomp, history = ArnoldiMethod.partialschur(
            J; nev = n, which = checkarnoldiwhich(l.which), v1 = l.x₀, l.kwargs...
        )
        λ, ϕ = ArnoldiMethod.partialeigen(decomp)
    else
        # (sigma⋅I - J)⁻¹ has the eigenvalues μ = 1 / (sigma - λ), so the eigenvalues closest
        # to `sigma` are the dominant ones and `:LM` is the right target here.
        M = l.sigma * LinearAlgebra.I - J
        op = ShiftInvert(l.factorize(M), nothing, size(J, 1), eltype(M)[])
        decomp, history = ArnoldiMethod.partialschur(op; nev = n, which = :LM, v1 = l.x₀, l.kwargs...)
        μ, ϕ = ArnoldiMethod.partialeigen(decomp)
        λ = @. l.sigma - inv(μ)
    end

    history.nconverged < n &&
        @warn "only $(history.nconverged) of the $n requested eigenvalues converged using ArnoldiMethod.partialschur"

    values, vectors = sortselect(λ, ϕ, n, l.which)
    return values, vectors, history.converged, history.mvproducts
end

function EigenValueSolvers.geneigsolve(l::EigArnoldiMethod, A, B, nev::Int; kwargs...)
    @argcheck A isa AbstractMatrix
    @argcheck B isa AbstractMatrix
    n = min(nev, size(A, 1))

    # A⋅x = λ⋅B⋅x is solved with the shift-invert method
    # (A - sigma⋅B)⁻¹ B⋅x = 1 / (λ - sigma) ⋅ x
    σ = isnothing(l.sigma) ? zero(eltype(A)) : l.sigma
    M = A - σ * B
    T = eltype(M)
    op = ShiftInvert(l.factorize(M), B, size(A, 1), Vector{T}(undef, size(A, 1)))

    decomp, history = ArnoldiMethod.partialschur(op; nev = n, which = :LM, v1 = l.x₀, l.kwargs...)
    μ, ϕ = ArnoldiMethod.partialeigen(decomp)
    λ = @. σ + inv(μ)

    history.nconverged < n &&
        @warn "only $(history.nconverged) of the $n requested eigenvalues converged using ArnoldiMethod.partialschur"

    values, vectors = sortselect(λ, ϕ, n, l.which)
    return values, vectors, history.converged, history.mvproducts
end

end
