using Documenter, GPUArrays

makedocs(
    modules = [GPUShowcases],
    format = :html,
    sitename = "GPU Showcases for Julia",
    pages = [
        "Home"    => "index.md",
        "Showcases"  => [
            "Convolution/convolution.md",
            "PDE/pde.md",
            "Poincare/poincare.md",
            "SmokeSimulation/smokesimulation.md"
        ],
    ],
    doctest = test
)

deploydocs(
    deps   = Deps.pip("mkdocs", "python-markdown-math", "mkdocs-cinder"),
    repo   = "github.com/JuliaGPU/GPUShowcases.jl.git",
    julia  = "0.6",
    osname = "linux"
)
