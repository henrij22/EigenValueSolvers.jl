using Documenter, DocumenterCodeBlocks, EigenValueSolvers

# Load the weak dependencies so that the package extensions are available
using ArnoldiMethod, Arpack, KrylovKit

const liveserver = "liveserver" in ARGS

if liveserver
    using Revise
    Revise.revise()
end

DocMeta.setdocmeta!(
    EigenValueSolvers, :DocTestSetup, :(using EigenValueSolvers, LinearAlgebra); recursive = true
)

makedocs(;
    format = Documenter.HTML(;
        canonical = "https://henrij22.github.io/EigenValueSolvers.jl/stable",
        collapselevel = 1,
    ),
    repo = Documenter.Remotes.GitHub("henrij22", "EigenValueSolvers.jl"),
    plugins = [CodeBlocks()],
    modules = [EigenValueSolvers],
    sitename = "EigenValueSolvers.jl",
    checkdocs = :exports,
    pages = [
        "Home" => "index.md",
        "Solvers" => "solvers.md",
        "API Reference" => "api_reference.md",
    ],
)

if !liveserver
    deploydocs(;
        repo = "github.com/henrij22/EigenValueSolvers.jl.git",
        push_preview = true,
        versions = [
            "stable" => "v^",
            "dev" => "dev",
        ],
    )
end
